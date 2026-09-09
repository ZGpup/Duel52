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

use super::mlp::{layer_norm, matvec, matvec_add, relu, LN_EPS};
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

    /// The input projection for one observation: `h` holds `lanes × width`, normalised and
    /// rectified, ready for the blocks.
    ///
    /// Split out of [`Self::trunk`] so that [`Self::trunk_batch`] runs **exactly this code**
    /// per row before it transposes into the batched layout. The input layer is the one part
    /// of the trunk that does not batch — it walks each observation's own ~205 non-zeros
    /// (`FINDINGS.md` F3.3) — and sharing the function is what stops the two paths drifting.
    fn input_row(&self, weights: &Weights, x: &[f32], h: &mut [f32], g: &mut [f32]) {
        let arch = &weights.arch;
        let w = &weights.params;
        let i = &self.idx;
        let (width, lanes) = (arch.width, arch.lanes);

        // Each lane starts at the shared bias; the global projection accumulates separately
        // and is added to every lane afterwards, so `glob_in` needs no bias of its own.
        for l in 0..lanes {
            h[l * width..(l + 1) * width].copy_from_slice(&w[i.lane_in_b]);
        }
        g.iter_mut().for_each(|v| *v = 0.0);

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
                for (acc, &wj) in g.iter_mut().zip(row) {
                    *acc += value * wj;
                }
            } else {
                let row = &self.lane_in_wt[owner.k as usize * width..][..width];
                let hl = &mut h[owner.lane as usize * width..][..width];
                for (acc, &wj) in hl.iter_mut().zip(row) {
                    *acc += value * wj;
                }
            }
        }

        for l in 0..lanes {
            let hl = &mut h[l * width..(l + 1) * width];
            for (acc, &gj) in hl.iter_mut().zip(g.iter()) {
                *acc += gj;
            }
            layer_norm(hl, &w[i.ln_in], &w[i.ln_in + 1]);
            relu(hl);
        }
    }

    /// `h_l` for every lane, then `p`, left in `scratch`.
    pub(super) fn trunk(&self, weights: &Weights, x: &[f32], scratch: &mut LaneScratch) {
        let arch = &weights.arch;
        let w = &weights.params;
        let i = &self.idx;
        let (width, lanes) = (arch.width, arch.lanes);

        // --- input ------------------------------------------------------------------
        self.input_row(weights, x, &mut scratch.h, &mut scratch.g);

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

    /// The trunk for `bs` observations at once.
    ///
    /// # Why this exists
    ///
    /// The trunk is ~89% of self-play's CPU time, and at one position per call it runs at
    /// about 14% of what the chip can do. The reason is the shape of [`matvec`]: a dot
    /// product is a *reduction*, every step needing the previous step's accumulator, so a
    /// single position cannot fill the machine's four-wide f32 units however it is written.
    /// Breaking the reduction into several accumulators would change the summation order,
    /// which `mlp.rs`'s module header makes a contract.
    ///
    /// The batch index does not have that problem. `out[i][b]` for different `b` are
    /// completely independent, so laying the activations out `[feature][batch]` and putting
    /// the batch in the inner loop fills the units without touching the reduction at all.
    ///
    /// # Why it is bit-identical
    ///
    /// **This is the property the whole design rests on.** [`matmat`] accumulates over `j`
    /// in the same ascending order as [`matvec`], starting from the same bias, so every
    /// `(i, b)` sees the identical sequence of f32 operations it would have seen alone.
    /// Nothing here depends on `bs` or on which other positions share the batch — so a game
    /// evaluated in a batch of 64 produces the same bits as the same game evaluated alone,
    /// and self-play stays reproducible from its seed whatever `--eval-batch` is set to.
    /// `phase4_batched_trunk_is_bit_identical_to_the_single_row_trunk` is the guard.
    pub(super) fn trunk_batch(
        &self,
        weights: &Weights,
        xs: &[f32],
        bs: usize,
        scratch: &mut LaneBatchScratch,
    ) {
        let arch = &weights.arch;
        let w = &weights.params;
        let i = &self.idx;
        let (width, lanes) = (arch.width, arch.lanes);
        assert!(bs > 0, "an empty batch has nothing to evaluate");
        assert!(bs <= scratch.cap, "batch is larger than the scratch was built for");
        assert_eq!(
            xs.len(),
            bs * arch.obs_dim,
            "observation batch is the wrong length"
        );
        scratch.rows = bs;
        let ls = width * bs;

        // --- input ------------------------------------------------------------------
        // Per row, through the same `input_row` the unbatched path uses, then transposed
        // into `[feature][batch]`. The input layer is sparse per observation and does not
        // batch; it is ~6% of the trunk, so it is not worth contorting.
        for b in 0..bs {
            let x = &xs[b * arch.obs_dim..(b + 1) * arch.obs_dim];
            self.input_row(weights, x, &mut scratch.row_h, &mut scratch.row_g);
            for (j, &v) in scratch.row_h.iter().enumerate() {
                scratch.h[j * bs + b] = v;
            }
        }

        // --- blocks -----------------------------------------------------------------
        let scale = 1.0 / lanes as f32;
        for blk in 0..arch.blocks {
            let at = i.blocks_at + blk * 7;

            scratch.n[..lanes * ls].copy_from_slice(&scratch.h[..lanes * ls]);
            for l in 0..lanes {
                layer_norm_batch(
                    &mut scratch.n[l * ls..l * ls + ls],
                    &w[at],
                    &w[at + 1],
                    bs,
                    width,
                    &mut scratch.mean,
                    &mut scratch.inv,
                );
            }
            scratch.m[..ls].iter_mut().for_each(|v| *v = 0.0);
            for l in 0..lanes {
                let src = &scratch.n[l * ls..l * ls + ls];
                for (acc, &nj) in scratch.m[..ls].iter_mut().zip(src) {
                    *acc += nj;
                }
            }
            scratch.m[..ls].iter_mut().for_each(|v| *v *= scale);

            matmat(&w[at + 3], None, &scratch.m, width, width, &mut scratch.mix, bs);

            for l in 0..lanes {
                let src = &scratch.n[l * ls..l * ls + ls];
                matmat(&w[at + 2], Some(&w[at + 4]), src, width, width, &mut scratch.t, bs);
                for k in 0..ls {
                    let v = scratch.t[k] + scratch.mix[k];
                    scratch.t[k] = if v < 0.0 { 0.0 } else { v };
                }
                matmat(
                    &w[at + 5],
                    Some(&w[at + 6]),
                    &scratch.t,
                    width,
                    width,
                    &mut scratch.r,
                    bs,
                );
                let dst = &mut scratch.h[l * ls..l * ls + ls];
                for (acc, &rj) in dst.iter_mut().zip(scratch.r[..ls].iter()) {
                    *acc += rj;
                }
            }
        }

        // --- output norm and the invariant pooled state ------------------------------
        for l in 0..lanes {
            layer_norm_batch(
                &mut scratch.h[l * ls..l * ls + ls],
                &w[i.ln_out],
                &w[i.ln_out + 1],
                bs,
                width,
                &mut scratch.mean,
                &mut scratch.inv,
            );
        }
        scratch.p[..ls].iter_mut().for_each(|v| *v = 0.0);
        for l in 0..lanes {
            let src = &scratch.h[l * ls..l * ls + ls];
            for (acc, &hj) in scratch.p[..ls].iter_mut().zip(src) {
                *acc += hj;
            }
        }
        scratch.p[..ls].iter_mut().for_each(|v| *v *= scale);
    }

    /// The value head for every row of a finished [`Self::trunk_batch`].
    ///
    /// Batched for the same reason the trunk is, and it became worth doing only once the
    /// trunk was: at one position per call it is ~4% of self-play, and with the trunk 2×
    /// faster it is ~11% of what is left. Same arithmetic as [`Self::value`], so the same
    /// bits — `out` receives the network's own `(-1, 1)` output, rescaling is the caller's.
    pub(super) fn value_batch(
        &self,
        weights: &Weights,
        scratch: &mut LaneBatchScratch,
        out: &mut [f32],
    ) {
        let arch = &weights.arch;
        let w = &weights.params;
        let i = &self.idx;
        let bs = scratch.rows;
        debug_assert!(out.len() >= bs);

        matmat(
            &w[i.value1_w],
            Some(&w[i.value1_b]),
            &scratch.p,
            arch.width,
            arch.value_hidden,
            &mut scratch.v,
            bs,
        );
        for x in scratch.v[..arch.value_hidden * bs].iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        // One output row: the same `bias + Σⱼ w·v` the single-row head runs.
        matmat(
            &w[i.value2_w],
            Some(&w[i.value2_b]),
            &scratch.v,
            arch.value_hidden,
            1,
            out,
            bs,
        );
        for x in out[..bs].iter_mut() {
            *x = x.tanh();
        }
    }

    /// Lift row `b` of a finished [`Self::trunk_batch`] into a single-row [`LaneScratch`].
    ///
    /// The heads then run through [`Self::policy_masked`] and [`Self::value`] unchanged,
    /// which is deliberate: they are ~4% of the cost and every position has a different
    /// legal mask, so batching them would buy nothing and give the two paths a second place
    /// to disagree.
    pub(super) fn unpack_row(
        &self,
        weights: &Weights,
        batch: &LaneBatchScratch,
        b: usize,
        out: &mut LaneScratch,
    ) {
        let arch = &weights.arch;
        let (width, lanes) = (arch.width, arch.lanes);
        let bs = batch.rows;
        debug_assert!(b < bs, "row {b} is not in a batch of {bs}");
        for j in 0..lanes * width {
            out.h[j] = batch.h[j * bs + b];
        }
        for j in 0..width {
            out.p[j] = batch.p[j * bs + b];
        }
    }
}

/// `out[i][b] = bias[i] + Σⱼ w[i][j] · x[j][b]`, with `x` and `out` laid out
/// `[feature][batch]`.
///
/// The batched twin of [`matvec`], and deliberately the same arithmetic: the accumulator
/// starts at the bias and takes `j` in ascending order, so each `(i, b)` gets bit-for-bit
/// what [`matvec`] would have given it. The inner loop is over the batch, which is the
/// dimension that carries no dependency and therefore vectorises.
/// Batch columns accumulated in one pass, and the reason it is a fixed-size array.
///
/// ⚠️ **The accumulators must live in a stack array of compile-time size, not in a slice of
/// `out`.** Written the obvious way — accumulating straight into `out[i*bs..]` — the compiler
/// cannot prove the output does not alias `w` or `x`, so it reloads and restores the
/// accumulator on every one of the `in_dim` iterations and the kernel runs at roughly the
/// speed of the unbatched one. Measured on an M2 at width 128: 1.9 GMAC/s accumulating into
/// the output slice against 11.9 GMAC/s into a stack array. A local array cannot alias
/// anything, so it stays in vector registers for the whole reduction.
///
/// 64 f32 is 16 NEON registers of the 32 an ARM64 core has, which leaves room for the
/// weight broadcast and the activation loads.
const TILE: usize = 64;

fn matmat(
    w: &[f32],
    bias: Option<&[f32]>,
    x: &[f32],
    in_dim: usize,
    out_dim: usize,
    out: &mut [f32],
    bs: usize,
) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert!(x.len() >= in_dim * bs);
    debug_assert!(out.len() >= out_dim * bs);
    for i in 0..out_dim {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let start = bias.map_or(0.0, |b| b[i]);
        let mut lo = 0usize;
        while lo < bs {
            let n = TILE.min(bs - lo);
            let mut acc = [start; TILE];
            let a = &mut acc[..n];
            for (j, &wj) in row.iter().enumerate() {
                let xr = &x[j * bs + lo..j * bs + lo + n];
                for (v, &xb) in a.iter_mut().zip(xr) {
                    *v += wj * xb;
                }
            }
            out[i * bs + lo..i * bs + lo + n].copy_from_slice(a);
            lo += n;
        }
    }
}

/// [`layer_norm`] over `width` features for each of `bs` rows laid out `[feature][batch]`.
///
/// Same order as the single-row version — mean over ascending `j`, then the biased variance,
/// then `1/√(var + ε)` — so the result is bit-identical per row. `mean` and `inv` are
/// caller-owned so nothing allocates in the hot loop.
fn layer_norm_batch(
    x: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    bs: usize,
    width: usize,
    mean: &mut [f32],
    inv: &mut [f32],
) {
    let n = width as f32;
    let mean = &mut mean[..bs];
    let inv = &mut inv[..bs];

    mean.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..width {
        let xr = &x[j * bs..(j + 1) * bs];
        for (acc, &v) in mean.iter_mut().zip(xr) {
            *acc += v;
        }
    }
    mean.iter_mut().for_each(|v| *v /= n);

    inv.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..width {
        let xr = &x[j * bs..(j + 1) * bs];
        for ((acc, &v), &mu) in inv.iter_mut().zip(xr).zip(mean.iter()) {
            let d = v - mu;
            *acc += d * d;
        }
    }
    inv.iter_mut()
        .for_each(|v| *v = 1.0 / (*v / n + LN_EPS).sqrt());

    for j in 0..width {
        let (g, be) = (gamma[j], beta[j]);
        let xr = &mut x[j * bs..(j + 1) * bs];
        for ((v, &mu), &iv) in xr.iter_mut().zip(mean.iter()).zip(inv.iter()) {
            *v = (*v - mu) * iv * g + be;
        }
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

/// Working buffers for [`LaneBody::trunk_batch`], sized for a batch of `cap` rows.
///
/// Everything from `h` down to `p` is laid out `[feature][batch]` with a stride of the
/// batch's *current* size, so a short final batch does no work on unused columns. `row_h`
/// and `row_g` are the single-row buffers the input projection fills before the transpose.
pub(super) struct LaneBatchScratch {
    /// Rows this was allocated for.
    cap: usize,
    /// Rows the last [`LaneBody::trunk_batch`] filled — the stride `unpack_row` reads with.
    rows: usize,
    h: Vec<f32>,
    n: Vec<f32>,
    m: Vec<f32>,
    mix: Vec<f32>,
    t: Vec<f32>,
    r: Vec<f32>,
    p: Vec<f32>,
    /// `value_hidden × batch` — the value head's hidden layer.
    v: Vec<f32>,
    row_h: Vec<f32>,
    row_g: Vec<f32>,
    /// Per-row LayerNorm intermediates, `cap` long.
    mean: Vec<f32>,
    inv: Vec<f32>,
}

impl LaneBatchScratch {
    pub(super) fn new(arch: &Arch, cap: usize) -> LaneBatchScratch {
        let w = arch.width;
        LaneBatchScratch {
            cap,
            rows: 0,
            h: vec![0.0; arch.lanes * w * cap],
            n: vec![0.0; arch.lanes * w * cap],
            m: vec![0.0; w * cap],
            mix: vec![0.0; w * cap],
            t: vec![0.0; w * cap],
            r: vec![0.0; w * cap],
            p: vec![0.0; w * cap],
            v: vec![0.0; arch.value_hidden * cap],
            row_h: vec![0.0; arch.lanes * w],
            row_g: vec![0.0; w],
            mean: vec![0.0; cap],
            inv: vec![0.0; cap],
        }
    }
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
