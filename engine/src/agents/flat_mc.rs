//! Flat Monte Carlo.
//!
//! `PLAN.md` Phase 2, rung three. The simplest thing that can be called search: score every
//! legal action by the average result of random playouts that begin with it, and take the
//! best average. No tree, no selectivity, no reuse between decisions.
//!
//! # Two details that matter more than the algorithm
//!
//! **It determinizes.** A playout needs concrete hidden cards — the opponent's hand, the
//! pile order — and reading the engine's real ones is the cheat `DESIGN.md` §6 exists to
//! prevent. So each sweep samples a world from the acting player's information set and rolls
//! out inside it.
//!
//! **Sweeps are paired.** All actions in one sweep are evaluated against the *same* sampled
//! world, rather than each drawing its own. Common random numbers: the world is by far the
//! largest source of variance in the estimate, and pairing removes it from the *differences*
//! between actions, which is the only thing an argmax cares about. The same total playout
//! budget then separates actions several times more sharply.
//!
//! # What it is for
//!
//! Flat MC is the control that isolates what a *tree* buys. It sees the same information as
//! ISMCTS and spends a comparable budget; the gap between them on the Phase 2 ladder is the
//! value of selectivity, with the evaluation function held out of the comparison entirely
//! (neither uses one).

use crate::action::Action;
use crate::agents::{pick_best, random_playout, Agent};
use crate::rng::Rng;
use crate::state::GameState;

/// Scores each legal action by the mean outcome of random playouts that start with it.
#[derive(Clone, Debug)]
pub struct FlatMcAgent {
    rng: Rng,
    /// Total playouts per decision, spread evenly over the legal actions.
    playouts: usize,
}

impl FlatMcAgent {
    pub const DEFAULT_PLAYOUTS: usize = 600;

    pub fn new(seed: u64, playouts: usize) -> FlatMcAgent {
        FlatMcAgent {
            rng: Rng::new(seed),
            playouts,
        }
    }

    pub fn derived(seed: u64, stream: u64, playouts: usize) -> FlatMcAgent {
        FlatMcAgent {
            rng: Rng::derive(seed, stream),
            playouts,
        }
    }
}

impl Agent for FlatMcAgent {
    fn choose(&mut self, state: &GameState, legal: &[Action]) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let me = state.acting_player();

        // At least one sweep, so a wide decision node still gets every action sampled once
        // rather than silently sampling none of them.
        let sweeps = (self.playouts / legal.len()).max(1);
        let mut totals = vec![0.0f64; legal.len()];

        for _ in 0..sweeps {
            let world = state.determinize(me, &mut self.rng);
            for (i, &action) in legal.iter().enumerate() {
                let mut playout = world.clone();
                playout.apply_trusted(action);
                random_playout(&mut playout, &mut self.rng);
                totals[i] += playout.outcome.value_for(me) as f64;
            }
        }

        let means: Vec<f32> = totals.iter().map(|&t| (t / sweeps as f64) as f32).collect();
        legal[pick_best(&means, &mut self.rng)]
    }

    fn name(&self) -> String {
        format!("flatmc:{}", self.playouts)
    }
}
