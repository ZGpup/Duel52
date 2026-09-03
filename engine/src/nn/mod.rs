//! The reference network: weights, the checkpoint format, and a forward pass.
//!
//! # Why inference lives in Rust
//!
//! `DESIGN.md` §9 implied the training loop would own the network and reach the engine
//! through PyO3. That is the wrong seam for Phase 3, for one decisive reason: **the phase's
//! deliverable is an Elo table**, and that table comes out of `duel52 ladder`, which is Rust
//! and takes an [`crate::AgentSpec`]. So do `match`, `probe` and `play --opponent`, and
//! `FINDINGS.md` F2.4, F2.5, F2.7 and F2.8 all ask Phase 3 to re-run those measurements
//! against the trained agent. A Python-side agent can use none of it.
//!
//! So: **search and inference in Rust, training in Python.** PyTorch still owns the
//! architecture, the weights and the gradients; Rust gets a frozen snapshot and runs forward
//! passes. They meet at a checkpoint file, and [`Weights::load`] refuses one whose layout
//! hashes do not match this build.
//!
//! # Why there is a hand-rolled matmul here
//!
//! The workspace `Cargo.toml` is emphatic that `engine` has no dependencies, and gives the
//! reason: reproducibility, because third-party RNGs do not promise stability across
//! versions. A five-layer f32 MLP does not threaten that reason at all — it threatens the
//! opposite, since a BLAS would introduce exactly the accumulation-order variability the
//! project is trying to avoid. So the reference forward pass is ~60 lines of plain loops
//! with a documented accumulation order, and this step adds no crates.
//!
//! A GPU backend (ONNX Runtime, or `tch`) belongs at the CUDA handoff, in a separate `nn`
//! crate alongside a `cli` crate. [`Evaluator`] is the seam that makes that swap cheap; see
//! `DESIGN.md` §9.

mod mlp;
mod weights;

pub use mlp::MlpEvaluator;
pub use weights::{Arch, Weights, CHECKPOINT_MAGIC, CHECKPOINT_VERSION};

/// Something that can turn observations into raw policy logits and values.
///
/// **Batch-shaped from the start, deliberately.** The self-play loop this is built for will
/// keep `G` games in flight per worker and advance one simulation in each per round,
/// evaluating the whole round as a single batch — no virtual loss, no search distortion, and
/// every game still reproducible from its own seed. Retrofitting a batch interface onto a
/// single-position one later would touch the whole loop, so it is locked now even though the
/// only consumer in this step evaluates one position at a time.
///
/// Flat slices with caller-allocated outputs, so nothing allocates in the hot loop.
pub trait Evaluator: Send + Sync {
    /// Evaluate `n` observations.
    ///
    /// `obs` is a flat `n × obs_dim` batch, row-major. `logits_out` receives `n ×
    /// action_dim` **raw, unmasked** logits and `values_out` receives `n` values in
    /// `(-1, 1)`.
    ///
    /// Masking and softmax are the caller's job: PUCT needs the masked distribution anyway,
    /// and a masked softmax inside the network would have to be mirrored exactly in PyTorch
    /// for the parity test to mean anything.
    fn eval_batch(&self, obs: &[f32], n: usize, logits_out: &mut [f32], values_out: &mut [f32]);

    fn obs_dim(&self) -> usize;
    fn action_dim(&self) -> usize;
}
