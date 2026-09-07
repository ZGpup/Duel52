//! The lane-equivariant forward pass.
//!
//! `PLAN.md` §4.2b. `FINDINGS.md` F4.3 measured the flat MLP not knowing that Duel 52's three
//! lanes are interchangeable — an opening prior of .320 / .277 / .403 where it must be .333
//! each — and F4.5 showed data augmentation shrinking that defect without removing it (policy
//! TV 0.152 → 0.039, argmax agreement stuck at 114/128). Augmentation can only *ask* the
//! network to be symmetric. This makes it symmetric by construction:
//!
//! ```text
//! u_l = W_lane · x[lane_obs[l]] + b_in           per lane, ONE shared matrix
//! v   = W_glob · x[global_obs]                   lane-invariant, no bias of its own
//! h_l = relu(ln_in(u_l + v))
//!
//! repeat blocks:
//!     n_l = ln(h_l)
//!     m   = mean_l n_l                           the only channel between lanes
//!     h_l = h_l + W2 · relu(W1·n_l + Wm·m + b1) + b2
//!
//! h_l = ln_out(h_l)
//! p   = mean_l h_l
//!
//! logits[lane_action[l][k]] = W_pl[k] · h_l + b_pl[k]
//! logits[global_action[k]]  = W_pg[k] · p   + b_pg[k]
//! value                     = tanh(W_v2 · relu(W_v1·p + b_v1) + b_v2)
//! ```
//!
//! # Why this is exactly equivariant
//!
//! Relabelling the lanes permutes the `h_l` and leaves `m` and `p` **unchanged**, because a
//! mean does not care about order. So the lane-owned logits permute exactly with the lanes,
//! the global logits and the value do not move at all, and no weight anywhere is indexed by a
//! lane. There is no parameter that *could* encode a lane preference, which is a stronger
//! statement than F4.5's "the preference got small".
//!
//! `phase4_the_lane_network_is_exactly_equivariant` is the guard, and it asserts equality
//! rather than a tolerance: the same weights on a relabelled observation must produce the
//! permuted logits bit-for-bit, up to the f32 reassociation a different summation order
//! causes.
//!
//! # Which indices belong to which lane
//!
//! [`crate::encode::lane_structure`], and nowhere else. `CLAUDE.md`: there is exactly one
//! encoder and a table of "which lane owns this float" is a reading of it. The reverse maps
//! built here ([`Owner`]) are inverted from that table at construction, not derived
//! independently.
//!
//! # Cost
//!
//! The trunk runs `lanes` times, but the input projection — 58% of the flat network's
//! parameters — does not grow: three lanes of `lane_obs × width` is the same work as one
//! `obs_dim × width`. The policy head *shrinks* by a factor of `lanes`, because one shared
//! `width × lane_action` matrix replaces `width × action_dim`.

use super::mlp::{layer_norm, matvec, matvec_add, relu};
use super::weights::{Arch, Weights};
use crate::config::GameConfig;
use crate::encode::lane_structure;

/// Which lane owns one index of the observation or policy vector, and where it sits in that
/// lane's list.
///
/// `lane == GLOBAL` means the index is owned by no lane: a scalar the relabelling cannot
/// move, or a `CHOOSE_RANK` logit.
#[derive(Clone, Copy)]
struct Owner {
    lane: u32,
    k: u32,
}

const GLOBAL: u32 = u32::MAX;

/// Resolved tensor positions for [`super::weights::ArchKind::Lane`].
///
/// Built from [`Arch::params`], so it cannot drift from the checkpoint's `param_order`.
struct Index {
    lane_in_w: usize,
    lane_in_b: usize,
    glob_in_w: usize,
    ln_in: usize,
    /// Block `i` starts at `blocks_at + i * 7`; the seven are
    /// `ln.weight, ln.bias, fc1.weight, fcm.weight, fc1.bias, fc2.weight, fc2.bias`.
    blocks_at: usize,
    ln_out: usize,
    policy_lane_w: usize,
    policy_lane_b: usize,
    policy_glob_w: usize,
    policy_glob_b: usize,
    value1_w: usize,
    value1_b: usize,
    value2_w: usize,
    value2_b: usize,
}

impl Index {
    fn new(arch: &Arch) -> Index {
        let after = 5 + arch.blocks * 7;
        Index {
            lane_in_w: 0,
            lane_in_b: 1,
            glob_in_w: 2,
            ln_in: 3, // .weight; .bias is ln_in + 1
            blocks_at: 5,
            ln_out: after, // .weight; .bias is ln_out + 1
            policy_lane_w: after + 2,
            policy_lane_b: after + 3,
            policy_glob_w: after + 4,
            policy_glob_b: after + 5,
            value1_w: after + 6,
            value1_b: after + 7,
            value2_w: after + 8,
            value2_b: after + 9,
        }
    }
}

/// Everything the lane forward pass needs beyond the weights themselves.
pub(super) struct LaneBody {
    idx: Index,
    /// `W_lane` transposed to `[lane_obs × width]`, so the input layer can walk the
    /// observation's non-zeros and add whole contiguous rows — the same trick the flat
    /// network uses, and it matters more here because ~205 of 4290 floats are non-zero.
    lane_in_wt: Vec<f32>,
    /// `W_glob` transposed to `[global_obs × width]`.
    glob_in_wt: Vec<f32>,
    /// `obs_dim` long: which lane owns each observation float.
    obs_owner: Vec<Owner>,
    /// `action_dim` long: which lane owns each logit.
    action_owner: Vec<Owner>,
}

impl LaneBody {
    pub(super) fn new(weights: &Weights, config: &GameConfig) -> LaneBody {
        let arch = weights.arch;
        let idx = Index::new(&arch);
        let structure = lane_structure(config);

        let transpose = |w: &[f32], in_dim: usize| {
            let mut t = vec![0.0f32; in_dim * arch.width];
            for i in 0..arch.width {
                for j in 0..in_dim {
                    t[j * arch.width + i] = w[i * in_dim + j];
                }
            }
            t
        };

        // Invert `lane_structure`'s forward tables. A gather table says "lane l's k-th float
        // is at index i"; the input layer walks `x` and needs the other direction.
        let invert = |lanes: &[Vec<u32>], global: &[u32], total: usize| {
            let mut owner = vec![Owner { lane: GLOBAL, k: 0 }; total];
            for (l, list) in lanes.iter().enumerate() {
                for (k, &i) in list.iter().enumerate() {
                    owner[i as usize] = Owner { lane: l as u32, k: k as u32 };
                }
            }
            for (k, &i) in global.iter().enumerate() {
                owner[i as usize] = Owner { lane: GLOBAL, k: k as u32 };
            }
            owner
        };

        LaneBody {
            lane_in_wt: transpose(&weights.params[idx.lane_in_w], arch.lane_obs),
            glob_in_wt: transpose(&weights.params[idx.glob_in_w], arch.global_obs()),
            obs_owner: invert(&structure.lane_obs, &structure.global_obs, arch.obs_dim),
            action_owner: invert(
                &structure.lane_action,
                &structure.global_action,
                arch.action_dim,
            ),
            idx,
        }
    }

    /// `h_l` for every lane, then `p`, left in `scratch`.
    pub(super) fn trunk(&self, weights: &Weights, x: &[f32], scratch: &mut LaneScratch) {
        let arch = &weights.arch;
        let w = &weights.params;
        let i = &self.idx;
        let (width, lanes) = (arch.width, arch.lanes);

        // --- input ------------------------------------------------------------------
        // Each lane starts at the shared bias; the global projection accumulates separately
        // and is added to every lane afterwards, so `glob_in` needs no bias of its own.
        for l in 0..lanes {
            scratch.h[l * width..(l + 1) * width].copy_from_slice(&w[i.lane_in_b]);
        }
        scratch.g.iter_mut().for_each(|v| *v = 0.0);

        // One ascending pass over the observation. Ascending because the accumulation order
        // is part of the contract (`mlp.rs`'s module header): every accumulator here is
        // built in increasing index order, exactly as the flat network's is.
        for (index, &value) in x.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let owner = self.obs_owner[index];
            if owner.lane == GLOBAL {
                let row = &self.glob_in_wt[owner.k as usize * width..][..width];
                for (acc, &wj) in scratch.g.iter_mut().zip(row) {
                    *acc += value * wj;
                }
            } else {
                let row = &self.lane_in_wt[owner.k as usize * width..][..width];
                let h = &mut scratch.h[owner.lane as usize * width..][..width];
                for (acc, &wj) in h.iter_mut().zip(row) {
                    *acc += value * wj;
                }
            }
        }

        for l in 0..lanes {
            let h = &mut scratch.h[l * width..(l + 1) * width];
            for (acc, &gj) in h.iter_mut().zip(scratch.g.iter()) {
                *acc += gj;
            }
            layer_norm(h, &w[i.ln_in], &w[i.ln_in + 1]);
            relu(h);
        }

        // --- blocks -----------------------------------------------------------------
        let scale = 1.0 / lanes as f32;
        for b in 0..arch.blocks {
            let at = i.blocks_at + b * 7;

            // n_l = ln(h_l), then m = mean_l n_l. The mean is the only path between lanes,
            // and being a mean is exactly why the block stays equivariant.
            for l in 0..lanes {
                let (from, to) = (l * width, (l + 1) * width);
                scratch.n[from..to].copy_from_slice(&scratch.h[from..to]);
                layer_norm(&mut scratch.n[from..to], &w[at], &w[at + 1]);
            }
            scratch.m.iter_mut().for_each(|v| *v = 0.0);
            for l in 0..lanes {
                for (acc, &nj) in scratch.m.iter_mut().zip(&scratch.n[l * width..][..width]) {
                    *acc += nj;
                }
            }
            scratch.m.iter_mut().for_each(|v| *v *= scale);

            // The mixing term is computed once and reused by every lane: it does not depend
            // on l, and recomputing it per lane would only cost time.
            matvec(&w[at + 3], None, &scratch.m, width, &mut scratch.mix);

            for l in 0..lanes {
                let n = &scratch.n[l * width..][..width];
                matvec(&w[at + 2], Some(&w[at + 4]), n, width, &mut scratch.t);
                matvec_add(&scratch.mix, &mut scratch.t);
                relu(&mut scratch.t);
                matvec(&w[at + 5], Some(&w[at + 6]), &scratch.t, width, &mut scratch.r);
                let h = &mut scratch.h[l * width..][..width];
                for (acc, &rj) in h.iter_mut().zip(scratch.r.iter()) {
                    *acc += rj;
                }
            }
        }

        // --- output norm and the invariant pooled state ------------------------------
        for l in 0..lanes {
            layer_norm(
                &mut scratch.h[l * width..(l + 1) * width],
                &w[i.ln_out],
                &w[i.ln_out + 1],
            );
        }
        scratch.p.iter_mut().for_each(|v| *v = 0.0);
        for l in 0..lanes {
            for (acc, &hj) in scratch.p.iter_mut().zip(&scratch.h[l * width..][..width]) {
                *acc += hj;
            }
        }
        scratch.p.iter_mut().for_each(|v| *v *= scale);
    }

    /// One logit, from a trunk output already in `scratch`.
    ///
    /// Shared by the dense and the masked paths so that the two cannot disagree: a masked
    /// logit is bit-identical to the dense one because it is the same function.
    #[inline]
    fn logit(&self, weights: &Weights, a: usize, scratch: &LaneScratch) -> f32 {
        let arch = &weights.arch;
        let w = &weights.params;
        let i = &self.idx;
        let owner = self.action_owner[a];
        let (mat, bias, state) = if owner.lane == GLOBAL {
            (&w[i.policy_glob_w], &w[i.policy_glob_b], &scratch.p[..])
        } else {
            (
                &w[i.policy_lane_w],
                &w[i.policy_lane_b],
                &scratch.h[owner.lane as usize * arch.width..][..arch.width],
            )
        };
        let row = &mat[owner.k as usize * arch.width..][..arch.width];
        let mut acc = bias[owner.k as usize];
        for (&wj, &hj) in row.iter().zip(state) {
            acc += wj * hj;
        }
        acc
    }

    pub(super) fn policy(&self, weights: &Weights, logits: &mut [f32], scratch: &LaneScratch) {
        for a in 0..weights.arch.action_dim {
            logits[a] = self.logit(weights, a, scratch);
        }
    }

    pub(super) fn policy_masked(
        &self,
        weights: &Weights,
        mask: &[bool],
        logits: &mut [f32],
        scratch: &LaneScratch,
    ) {
        for (a, &allowed) in mask.iter().enumerate() {
            if allowed {
                logits[a] = self.logit(weights, a, scratch);
            }
        }
    }

    /// The value head, read off the pooled state — so it is lane-invariant by construction.
    pub(super) fn value(&self, weights: &Weights, scratch: &mut LaneScratch) -> f32 {
        let w = &weights.params;
        let i = &self.idx;
        matvec(
            &w[i.value1_w],
            Some(&w[i.value1_b]),
            &scratch.p,
            weights.arch.width,
            &mut scratch.v,
        );
        relu(&mut scratch.v);
        let mut acc = w[i.value2_b][0];
        for (&wj, &vj) in w[i.value2_w].iter().zip(scratch.v.iter()) {
            acc += wj * vj;
        }
        acc.tanh()
    }
}

/// Working buffers for the lane forward pass.
pub(super) struct LaneScratch {
    /// `lanes × width` — the residual stream, one row per lane.
    h: Vec<f32>,
    /// `lanes × width` — the normalised copy a block reads.
    n: Vec<f32>,
    /// `width` — the global projection, added to every lane.
    g: Vec<f32>,
    /// `width` — the mean of the normalised lane states.
    m: Vec<f32>,
    /// `width` — `Wm · m`, computed once per block.
    mix: Vec<f32>,
    t: Vec<f32>,
    r: Vec<f32>,
    /// `width` — the pooled, lane-invariant state the value head and `CHOOSE_RANK` read.
    p: Vec<f32>,
    v: Vec<f32>,
}

impl LaneScratch {
    pub(super) fn new(arch: &Arch) -> LaneScratch {
        let w = arch.width;
        LaneScratch {
            h: vec![0.0; arch.lanes * w],
            n: vec![0.0; arch.lanes * w],
            g: vec![0.0; w],
            m: vec![0.0; w],
            mix: vec![0.0; w],
            t: vec![0.0; w],
            r: vec![0.0; w],
            p: vec![0.0; w],
            v: vec![0.0; arch.value_hidden],
        }
    }
}
