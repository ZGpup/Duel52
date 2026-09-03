//! The reference forward pass.
//!
//! `PHASE3_STEP1.md` §1.5, which pins `DESIGN.md` §5's residual MLP exactly:
//!
//! ```text
//! x                     obs_dim floats
//! h = relu(ln_in(W_in·x + b_in))                              width
//! repeat blocks:
//!     r = W2·relu(W1·ln_i(h) + b1) + b2
//!     h = h + r
//! h = ln_out(h)
//! policy_logits = W_p·h + b_p                                 raw, unmasked
//! value         = tanh(W_v2·relu(W_v1·h + b_v1) + b_v2)       scalar
//! ```
//!
//! Pre-norm: the LayerNorm inside a block runs *before* the block's first linear, and the
//! residual add is unnormalised. That is what lets the trunk go deep without the residual
//! stream drifting, and it is the arrangement `py/duel52/nn/model.py` mirrors.
//!
//! # Determinism, and why it is spelled out
//!
//! `CLAUDE.md`: "Everything is seeded and deterministic. Non-reproducible results are bugs."
//! For a forward pass that means four commitments, all of them testable:
//!
//! - **f32 throughout.** No f64 accumulator, because PyTorch will not use one either and the
//!   parity thresholds are set against f32 behaviour.
//! - **A fixed accumulation order.** Every dot product runs `j` from `0` to `in_dim − 1`,
//!   starting from the bias. Floating-point addition is not associative, so this ordering is
//!   part of the contract, not an implementation detail.
//! - **No parallelism inside the reference implementation.** A batch is evaluated row by
//!   row; if a future backend parallelises, it parallelises *across* rows, which cannot
//!   change any single row's arithmetic.
//! - **No fast-math.** Nothing here asks the optimiser to reassociate.
//!
//! `phase3_forward_pass_is_deterministic` checks all four by asserting bit-identical output
//! across repeated calls and across threads.
//!
//! Bit-exactness against PyTorch is *not* claimed and is not achievable — PyTorch reduces in
//! a different order, and on a different device again. `py/tests/test_parity.py` therefore
//! asserts agreement to `1e-3` on logits and `1e-4` on values, which is loose enough to
//! survive accumulation order and tight enough that any transcription bug — a transposed
//! weight, a swapped gamma and beta, a missing residual — fails immediately, because those
//! produce `O(1)` differences rather than `O(1e-6)` ones.

use super::weights::{Arch, Weights};
use super::Evaluator;

/// LayerNorm epsilon. Matches PyTorch's `nn.LayerNorm` default; the parity test is sensitive
/// to this, because a different epsilon shifts every activation slightly.
const LN_EPS: f32 = 1e-5;

/// The reference [`Evaluator`]: plain loops over [`Weights`].
pub struct MlpEvaluator {
    /// Shared, because a ladder builds one agent per game and the default architecture is
    /// ~20 MB — cloning it per game would cost more than the games.
    weights: std::sync::Arc<Weights>,
    /// Index of each tensor in `weights.params`, resolved once so the hot loop does no
    /// string comparison.
    idx: Index,
    /// `W_in` transposed to `[obs_dim × width]`, so the input layer can walk the
    /// observation's non-zeros and add whole contiguous rows. See [`input_layer`].
    ///
    /// Costs `obs_dim × width` floats — 8.8 MB at the default architecture — which is why
    /// [`crate::nn::evaluator_for`] caches evaluators rather than weights.
    in_wt: Vec<f32>,
}

/// Resolved tensor positions. Built from [`Arch::params`], so it cannot drift from the
/// checkpoint's `param_order`.
struct Index {
    in_w: usize,
    in_b: usize,
    ln_in: usize,
    /// First tensor of block `i` is at `blocks_at + i * 6`; the six are
    /// `ln.weight, ln.bias, fc1.weight, fc1.bias, fc2.weight, fc2.bias`.
    blocks_at: usize,
    ln_out: usize,
    policy_w: usize,
    policy_b: usize,
    value1_w: usize,
    value1_b: usize,
    value2_w: usize,
    value2_b: usize,
}

impl Index {
    fn new(arch: &Arch) -> Index {
        let after_blocks = 4 + arch.blocks * 6;
        Index {
            in_w: 0,
            in_b: 1,
            ln_in: 2, // .weight; .bias is ln_in + 1
            blocks_at: 4,
            ln_out: after_blocks, // .weight; .bias is ln_out + 1
            policy_w: after_blocks + 2,
            policy_b: after_blocks + 3,
            value1_w: after_blocks + 4,
            value1_b: after_blocks + 5,
            value2_w: after_blocks + 6,
            value2_b: after_blocks + 7,
        }
    }
}

impl MlpEvaluator {
    pub fn new(weights: Weights) -> MlpEvaluator {
        MlpEvaluator::shared(std::sync::Arc::new(weights))
    }

    /// Build over weights someone else already holds — how a cached checkpoint reaches an
    /// agent without a copy.
    ///
    /// Not cheap: it transposes the input matrix. Build one per checkpoint per process
    /// ([`crate::nn::evaluator_for`]), not one per game.
    pub fn shared(weights: std::sync::Arc<Weights>) -> MlpEvaluator {
        let idx = Index::new(&weights.arch);
        let arch = weights.arch;
        let w_in = &weights.params[idx.in_w];
        let mut in_wt = vec![0.0f32; arch.obs_dim * arch.width];
        for i in 0..arch.width {
            for j in 0..arch.obs_dim {
                in_wt[j * arch.width + i] = w_in[i * arch.obs_dim + j];
            }
        }
        MlpEvaluator {
            weights,
            idx,
            in_wt,
        }
    }

    /// Working buffers, so a search loop allocates once rather than once per evaluation.
    pub fn scratch(&self) -> Scratch {
        Scratch::new(&self.weights.arch)
    }

    pub fn arch(&self) -> Arch {
        self.weights.arch
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// Everything up to and including `ln_out`, leaving the trunk output in `scratch.h`.
    fn trunk(&self, x: &[f32], scratch: &mut Scratch) {
        let arch = &self.weights.arch;
        let w = &self.weights.params;
        let i = &self.idx;

        // h = relu(ln_in(W_in·x + b_in))
        input_layer(&self.in_wt, &w[i.in_b], x, arch.width, &mut scratch.h);
        layer_norm(&mut scratch.h, &w[i.ln_in], &w[i.ln_in + 1]);
        relu(&mut scratch.h);

        // h = h + W2·relu(W1·ln(h) + b1) + b2, per block.
        for b in 0..arch.blocks {
            let at = i.blocks_at + b * 6;
            scratch.a.copy_from_slice(&scratch.h);
            layer_norm(&mut scratch.a, &w[at], &w[at + 1]);
            matvec(&w[at + 2], &w[at + 3], &scratch.a, arch.width, &mut scratch.b);
            relu(&mut scratch.b);
            matvec(&w[at + 4], &w[at + 5], &scratch.b, arch.width, &mut scratch.a);
            for (h, r) in scratch.h.iter_mut().zip(&scratch.a) {
                *h += *r;
            }
        }

        layer_norm(&mut scratch.h, &w[i.ln_out], &w[i.ln_out + 1]);
    }

    /// The value head, read off a trunk output already in `scratch.h`.
    fn value_head(&self, scratch: &mut Scratch) -> f32 {
        let arch = &self.weights.arch;
        let w = &self.weights.params;
        let i = &self.idx;

        // A two-layer head ending in tanh, so it is bounded to the zero-sum range.
        matvec(
            &w[i.value1_w],
            &w[i.value1_b],
            &scratch.h,
            arch.width,
            &mut scratch.v,
        );
        relu(&mut scratch.v);
        // The final row is a single dot product, so it is written out rather than routed
        // through `matvec` for a one-element output — same ascending order either way.
        let mut acc = w[i.value2_b][0];
        for (&wj, &vj) in w[i.value2_w].iter().zip(scratch.v.iter()) {
            acc += wj * vj;
        }
        acc.tanh()
    }

    /// One row. `logits` is `action_dim` long; the value is returned.
    fn forward_one(&self, x: &[f32], logits: &mut [f32], scratch: &mut Scratch) -> f32 {
        let arch = &self.weights.arch;
        let w = &self.weights.params;
        let i = &self.idx;
        self.trunk(x, scratch);
        // Policy: raw logits, unmasked. Masking and softmax are the caller's job.
        matvec(&w[i.policy_w], &w[i.policy_b], &scratch.h, arch.width, logits);
        self.value_head(scratch)
    }

    /// One row, computing **only** the logits `mask` selects.
    ///
    /// The search path. `mask` is the legal-action mask, and a Duel 52 position offers
    /// ~21 of 2195 encoded actions (`FINDINGS.md` F3.3), so skipping the rest removes
    /// almost the whole policy head. Entries where `mask` is false are left untouched —
    /// the caller never reads them, and writing a placeholder would cost the same
    /// `action_dim` pass this exists to avoid.
    ///
    /// Every logit it *does* write is bit-identical to [`Self::forward_one`]'s: same
    /// weights, same row, same ascending accumulation order.
    pub fn eval_masked_with(
        &self,
        x: &[f32],
        mask: &[bool],
        logits: &mut [f32],
        scratch: &mut Scratch,
    ) -> f32 {
        let arch = &self.weights.arch;
        let w = &self.weights.params;
        let i = &self.idx;
        assert_eq!(x.len(), arch.obs_dim, "observation is the wrong length");
        assert_eq!(mask.len(), arch.action_dim, "mask is the wrong length");
        assert_eq!(logits.len(), arch.action_dim, "logit buffer is the wrong length");

        self.trunk(x, scratch);

        let pw = &w[i.policy_w];
        let pb = &w[i.policy_b];
        for (a, &allowed) in mask.iter().enumerate() {
            if !allowed {
                continue;
            }
            let row = &pw[a * arch.width..(a + 1) * arch.width];
            let mut acc = pb[a];
            for (&wj, &hj) in row.iter().zip(scratch.h.iter()) {
                acc += wj * hj;
            }
            logits[a] = acc;
        }

        self.value_head(scratch)
    }
}

impl Evaluator for MlpEvaluator {
    fn eval_batch(&self, obs: &[f32], n: usize, logits_out: &mut [f32], values_out: &mut [f32]) {
        let arch = &self.weights.arch;
        assert_eq!(obs.len(), n * arch.obs_dim, "observation batch is the wrong length");
        assert_eq!(
            logits_out.len(),
            n * arch.action_dim,
            "logit buffer is the wrong length"
        );
        assert_eq!(values_out.len(), n, "value buffer is the wrong length");

        let mut scratch = Scratch::new(arch);
        for row in 0..n {
            let x = &obs[row * arch.obs_dim..(row + 1) * arch.obs_dim];
            let logits = &mut logits_out[row * arch.action_dim..(row + 1) * arch.action_dim];
            values_out[row] = self.forward_one(x, logits, &mut scratch);
        }
    }

    fn obs_dim(&self) -> usize {
        self.weights.arch.obs_dim
    }

    fn action_dim(&self) -> usize {
        self.weights.arch.action_dim
    }
}

/// Per-call working buffers, allocated once per `eval_batch` rather than per row.
///
/// Public so a search loop can hold one across millions of evaluations; get one from
/// [`MlpEvaluator::scratch`].
pub struct Scratch {
    h: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    v: Vec<f32>,
}

impl Scratch {
    fn new(arch: &Arch) -> Scratch {
        Scratch {
            h: vec![0.0; arch.width],
            a: vec![0.0; arch.width],
            b: vec![0.0; arch.width],
            v: vec![0.0; arch.value_hidden],
        }
    }
}

/// `out = W_in·x + b`, walking only the non-zeros of `x`.
///
/// `wt` is `W_in` transposed to `[in_dim × out_dim]`, so one non-zero input contributes a
/// contiguous `out_dim`-long row. That is the whole reason the transpose is kept: the
/// observation is ~5% dense (205 of 4290 features, `FINDINGS.md` F3.3), and the input layer
/// is otherwise the largest matrix in the network.
///
/// **Accumulation order is preserved.** For each output `i` the terms still arrive with `j`
/// ascending, starting from the bias — the loop is transposed, not reordered — so this
/// computes bit-identically to [`matvec`] over the untransposed matrix, with one
/// footnote: a term whose input is exactly zero is skipped rather than added, and
/// `acc + 0.0` differs from `acc` only when `acc` is `-0.0`, which then becomes `+0.0`.
/// Nothing downstream can distinguish those two. `sparse_input_matches_the_dense_matvec`
/// checks the equality on random weights.
#[inline]
fn input_layer(wt: &[f32], bias: &[f32], x: &[f32], out_dim: usize, out: &mut [f32]) {
    debug_assert_eq!(wt.len(), x.len() * out_dim);
    debug_assert_eq!(bias.len(), out_dim);
    out.copy_from_slice(bias);
    for (j, &xj) in x.iter().enumerate() {
        if xj == 0.0 {
            continue;
        }
        let row = &wt[j * out_dim..(j + 1) * out_dim];
        for (o, &wij) in out.iter_mut().zip(row.iter()) {
            *o += wij * xj;
        }
    }
}

/// `out = W·x + b`, with `W` stored row-major as `[out_dim × in_dim]`.
///
/// The accumulation order — bias first, then `j` ascending — is part of the reproducibility
/// contract. See the module docs.
#[inline]
fn matvec(w: &[f32], bias: &[f32], x: &[f32], in_dim: usize, out: &mut [f32]) {
    debug_assert_eq!(w.len(), out.len() * in_dim);
    debug_assert_eq!(bias.len(), out.len());
    debug_assert_eq!(x.len(), in_dim);
    for (i, o) in out.iter_mut().enumerate() {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let mut acc = bias[i];
        // Ascending `j`, one term at a time. `zip` rather than an index because it is
        // clearer, not because it is faster — the *order* is the contract here, and any
        // rewrite must preserve it.
        for (&wj, &xj) in row.iter().zip(x.iter()) {
            acc += wj * xj;
        }
        *o = acc;
    }
}

/// LayerNorm over the whole vector, with elementwise affine. Biased variance (divide by `n`),
/// matching PyTorch.
#[inline]
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len() as f32;
    let mut mean = 0.0f32;
    for &v in x.iter() {
        mean += v;
    }
    mean /= n;
    let mut var = 0.0f32;
    for &v in x.iter() {
        let d = v - mean;
        var += d * d;
    }
    var /= n;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv * gamma[i] + beta[i];
    }
}

#[inline]
fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GameConfig;
    use crate::encode::{action_dim, obs_dim};

    fn tiny() -> Arch {
        Arch {
            obs_dim: 12,
            action_dim: 5,
            width: 8,
            blocks: 2,
            value_hidden: 4,
        }
    }

    #[test]
    fn output_shapes_and_ranges_are_what_the_trait_promises() {
        let e = MlpEvaluator::new(Weights::random(1, tiny()));
        let n = 3;
        let obs = vec![0.25f32; n * 12];
        let mut logits = vec![0.0; n * 5];
        let mut values = vec![0.0; n];
        e.eval_batch(&obs, n, &mut logits, &mut values);
        assert!(values.iter().all(|v| v.abs() < 1.0));
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    /// The residual add is what makes this a residual trunk rather than a plain stack. Zero
    /// the second linear of every block and each block computes `h + 0`, so depth must stop
    /// mattering: a one-block and a two-block network with the same stem and the same tail
    /// have to produce identical output. Drop the `h +` from the block and this fails,
    /// because the trunk would then collapse to zero instead of passing `h` through.
    #[test]
    fn with_each_block_zeroed_the_residual_trunk_is_the_identity() {
        /// A network whose blocks are all identity: `fc2` zeroed, so `r = 0`.
        fn identity_blocks(blocks: usize) -> Weights {
            let mut w = Weights::random(4, Arch { blocks, ..tiny() });
            for b in 0..blocks {
                let at = 4 + b * 6;
                w.params[at + 4].fill(0.0); // fc2.weight
                w.params[at + 5].fill(0.0); // fc2.bias
            }
            w
        }
        // `Arch::params` puts the stem first and the tail last, and both are the same shape
        // at either depth — so seeding both from 4 gives an identical stem, and copying the
        // eight tail tensors across makes the block count the only difference.
        let one = identity_blocks(1);
        let mut two = identity_blocks(2);
        let (one_tail, two_tail) = (one.params.len() - 8, two.params.len() - 8);
        for i in 0..8 {
            two.params[two_tail + i] = one.params[one_tail + i].clone();
        }

        let obs = vec![0.1f32, -0.2, 0.3, 0.4, 0.0, 1.0, -1.0, 0.5, 0.25, 0.75, -0.5, 0.6];
        let run = |w: Weights| {
            let e = MlpEvaluator::new(w);
            let mut logits = vec![0.0; 5];
            let mut value = vec![0.0; 1];
            e.eval_batch(&obs, 1, &mut logits, &mut value);
            (logits, value[0])
        };
        assert_eq!(run(one), run(two));
    }

    /// The transposed, zero-skipping input layer is an optimisation, not a different
    /// function. A dense `matvec` over the untransposed matrix has to give the same bits —
    /// on a sparse input, which is what it is for, and on a dense one, which is the case
    /// where the two loops must agree term for term.
    #[test]
    fn sparse_input_matches_the_dense_matvec() {
        let arch = Arch {
            obs_dim: 37,
            ..tiny()
        };
        let weights = Weights::random(9, arch);
        let e = MlpEvaluator::shared(std::sync::Arc::new(weights.clone()));

        let mut rng = crate::rng::Rng::new(11);
        for sparsity in [0usize, 3, 30, 37] {
            let mut x = vec![0.0f32; arch.obs_dim];
            for _ in 0..sparsity {
                x[rng.index(arch.obs_dim)] = (rng.below(9) as f32 - 4.0) / 4.0;
            }
            let mut dense = vec![0.0f32; arch.width];
            matvec(
                &weights.params[0],
                &weights.params[1],
                &x,
                arch.obs_dim,
                &mut dense,
            );
            let mut sparse = vec![0.0f32; arch.width];
            input_layer(&e.in_wt, &weights.params[1], &x, arch.width, &mut sparse);
            assert_eq!(dense, sparse, "sparsity {sparsity}");
        }
    }

    /// The masked path exists to skip 99% of the policy head. It must not change the logits
    /// it does compute, or search and training would be reading different functions.
    #[test]
    fn masked_evaluation_agrees_with_the_full_forward_pass() {
        let arch = Arch {
            obs_dim: 37,
            action_dim: 23,
            ..tiny()
        };
        let e = MlpEvaluator::new(Weights::random(5, arch));
        let mut rng = crate::rng::Rng::new(2);
        let mut x = vec![0.0f32; arch.obs_dim];
        for _ in 0..12 {
            x[rng.index(arch.obs_dim)] = 1.0;
        }
        let mask: Vec<bool> = (0..arch.action_dim).map(|i| i % 3 == 0).collect();

        let mut full = vec![0.0f32; arch.action_dim];
        let mut value = vec![0.0f32; 1];
        e.eval_batch(&x, 1, &mut full, &mut value);

        let mut masked = vec![f32::NAN; arch.action_dim];
        let mut scratch = e.scratch();
        let v = e.eval_masked_with(&x, &mask, &mut masked, &mut scratch);

        assert_eq!(v, value[0]);
        for (i, &allowed) in mask.iter().enumerate() {
            if allowed {
                assert_eq!(masked[i], full[i], "logit {i}");
            } else {
                assert!(masked[i].is_nan(), "logit {i} was written but not asked for");
            }
        }
    }

    #[test]
    fn layer_norm_standardises_before_the_affine() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.0f32; 4];
        layer_norm(&mut x, &gamma, &beta);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6, "mean {mean} is not zero");
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-4, "variance {var} is not one");
    }

    /// The default architecture has to accept the default encoder's output, or nothing else
    /// in the phase lines up.
    #[test]
    fn the_default_architecture_matches_the_default_encoder() {
        let config = GameConfig::default();
        let arch = Arch::default_for(&config);
        assert_eq!(arch.obs_dim, obs_dim(&config));
        assert_eq!(arch.action_dim, action_dim(&config));
    }
}
