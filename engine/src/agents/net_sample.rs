//! The sampling network agent — `PLAN.md` item 8.
//!
//! An R-NaD checkpoint is a **mixed** strategy: its policy is the thing that was trained, and
//! the probabilities are the point, so playing it by argmax (`netpolicy`) would play a
//! different strategy than the one learned. This agent samples instead.
//!
//! # Post-processing
//!
//! By default the policy is post-processed exactly as the R-NaD reference implementation
//! plays it (OpenSpiel `rnad.py` at `d1dcdf5d`, `FineTuning.post_process_policy`):
//!
//! 1. **Threshold.** Every legal action under [`THRESHOLD`] is dropped and the rest
//!    renormalised — unless *every* action is under it, in which case the policy is kept.
//! 2. **Discretise.** Probabilities are rounded to multiples of `1 / GRID`: each action, in
//!    descending order of probability, takes `ceil(p · GRID)` units until `GRID` are handed
//!    out, and any units left over go to the most likely action.
//!
//! A softmax leaves a little mass on every action, including the blunders; without search
//! nothing else removes it. `netsample:<path>@raw` skips both steps and samples the softmax.
//!
//! # Deterministic under its seed
//!
//! Exactly **one uniform draw per decision**, from the agent's own stream, whatever the
//! position. Two states in one information set encode identically, give the same policy and
//! draw the same number, so they pick the same action — which is what keeps
//! `engine/tests/agents.rs::phase2_no_agent_reads_hidden_information` exact for an agent that
//! samples. Actions are accumulated in encoded-index order, not in `legal` order, so the pick
//! does not depend on how the legal list happens to be enumerated.

use std::path::PathBuf;
use std::sync::Arc;

use crate::action::Action;
use crate::agents::Agent;
use crate::config::GameConfig;
use crate::encode::{action_dim, encode_action, encode_observation, obs_dim};
use crate::nn::{MlpEvaluator, Scratch};
use crate::rng::Rng;
use crate::state::GameState;

/// The reference's `FineTuning.policy_threshold`.
pub const THRESHOLD: f64 = 0.03;
/// The reference's `FineTuning.policy_discretization`.
pub const GRID: i64 = 32;

/// A checkpoint whose policy is sampled.
pub struct NetSampleAgent {
    checkpoint: PathBuf,
    raw: bool,
    rng: Rng,
    /// Resolved on the first decision, as in `netpolicy`: the layout hashes are config-derived.
    evaluator: Option<Arc<MlpEvaluator>>,
    scratch: Option<Scratch>,
    obs: Vec<f32>,
    logits: Vec<f32>,
    mask: Vec<bool>,
}

impl NetSampleAgent {
    pub fn derived(checkpoint: impl Into<PathBuf>, seed: u64, stream: u64, raw: bool) -> Self {
        NetSampleAgent {
            checkpoint: checkpoint.into(),
            raw,
            rng: Rng::derive(seed, stream),
            evaluator: None,
            scratch: None,
            obs: Vec::new(),
            logits: Vec::new(),
            mask: Vec::new(),
        }
    }

    fn ensure_loaded(&mut self, config: &GameConfig) {
        if self.evaluator.is_some() && self.obs.len() == obs_dim(config) {
            return;
        }
        let evaluator = crate::nn::evaluator_for(&self.checkpoint, config)
            .unwrap_or_else(|e| panic!("netsample: {e}"));
        self.scratch = Some(evaluator.scratch());
        self.obs = vec![0.0; obs_dim(config)];
        self.logits = vec![0.0; action_dim(config)];
        self.mask = vec![false; action_dim(config)];
        self.evaluator = Some(evaluator);
    }
}

/// The reference post-processing over one policy, in place.
///
/// `probs[k]` belongs to the action with encoded index `index[k]`; `index` breaks ties in the
/// descending sort, lowest index first, which is what the reference's `argsort(-p)` does.
pub fn post_process(probs: &mut [f64], index: &[usize]) {
    debug_assert_eq!(probs.len(), index.len());
    if probs.is_empty() {
        return;
    }

    // 1. Threshold.
    let max = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max >= THRESHOLD {
        let kept: f64 = probs.iter().filter(|&&p| p >= THRESHOLD).sum();
        for p in probs.iter_mut() {
            *p = if *p >= THRESHOLD { *p / kept } else { 0.0 };
        }
    }

    // 2. Discretise.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(index[a].cmp(&index[b]))
    });
    let mut units = vec![0i64; probs.len()];
    let mut left = GRID;
    for &k in &order {
        let x = ((probs[k] * GRID as f64).ceil() as i64).min(left);
        units[k] += x;
        left -= x;
    }
    if left > 0 {
        units[order[0]] += left;
    }
    for (p, u) in probs.iter_mut().zip(&units) {
        *p = *u as f64 / GRID as f64;
    }
}

impl Agent for NetSampleAgent {
    fn choose(&mut self, state: &GameState, legal: &[Action]) -> Action {
        self.ensure_loaded(&state.config);
        let evaluator = self.evaluator.as_ref().expect("loaded above");
        let scratch = self.scratch.as_mut().expect("loaded above");

        encode_observation(state, state.acting_player(), &mut self.obs);
        // Mask from the actions we were handed, so only something offered can come back.
        self.mask.fill(false);
        let mut offered: Vec<(usize, Action)> =
            legal.iter().map(|a| (encode_action(a, state), *a)).collect();
        offered.sort_by_key(|(index, _)| *index);
        for (index, _) in &offered {
            self.mask[*index] = true;
        }
        evaluator.eval_masked_with(&self.obs, &self.mask, &mut self.logits, scratch);

        // Softmax over the legal logits, in f64 and shifted by the maximum for stability.
        let top = offered
            .iter()
            .map(|(i, _)| self.logits[*i] as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> =
            offered.iter().map(|(i, _)| (self.logits[*i] as f64 - top).exp()).collect();
        let total: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= total);
        if !self.raw {
            let index: Vec<usize> = offered.iter().map(|(i, _)| *i).collect();
            post_process(&mut probs, &index);
        }

        // One draw per decision, always.
        let u = self.rng.unit() * probs.iter().sum::<f64>();
        let mut acc = 0.0;
        for (k, p) in probs.iter().enumerate() {
            acc += p;
            if *p > 0.0 && u < acc {
                return offered[k].1;
            }
        }
        // Floating-point residue: the last action that has any probability.
        offered[probs.iter().rposition(|p| *p > 0.0).unwrap_or(offered.len() - 1)].1
    }

    fn name(&self) -> String {
        let suffix = if self.raw { "@raw" } else { "" };
        format!("netsample:{}{suffix}", self.checkpoint.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processed(p: &[f64]) -> Vec<f64> {
        let mut v = p.to_vec();
        let index: Vec<usize> = (0..p.len()).collect();
        post_process(&mut v, &index);
        v
    }

    /// Hand-checked against the reference algorithm: 0.02 falls under the threshold, the
    /// rest renormalise to 0.5102 / 0.3061 / 0.1837, which take ceil(·32) = 17, 10, 6 units —
    /// 33, so the last action is cut to the 5 that are left.
    #[test]
    fn rnad_post_processing_thresholds_then_rounds_to_the_grid() {
        let out = processed(&[0.5, 0.3, 0.18, 0.02]);
        assert_eq!(out, vec![17.0 / 32.0, 10.0 / 32.0, 5.0 / 32.0, 0.0]);
    }

    /// Every action under the threshold: the reference keeps the policy rather than dividing
    /// by zero, and the grid still hands out all 32 units.
    #[test]
    fn rnad_post_processing_keeps_a_policy_that_is_under_the_threshold_everywhere() {
        let n = 40;
        let out = processed(&vec![1.0 / n as f64; n]);
        let units: f64 = out.iter().map(|p| p * 32.0).sum();
        assert_eq!(units, 32.0);
        // Ties break lowest index first: ceil(0.8) = 1 unit each to the first 32.
        assert!(out[..32].iter().all(|&p| p == 1.0 / 32.0));
        assert!(out[32..].iter().all(|&p| p == 0.0));
    }

    /// Rounding up over-allocates, so the least likely action is the one cut short, and the
    /// result always sums to exactly one.
    #[test]
    fn rnad_post_processing_cuts_the_least_likely_action_to_fit_the_grid() {
        assert_eq!(processed(&[1.0]), vec![1.0]);
        // ceil(16.32) = 17, then ceil(15.68) = 16 against 15 units left.
        assert_eq!(processed(&[0.51, 0.49]), vec![17.0 / 32.0, 15.0 / 32.0]);
    }
}
