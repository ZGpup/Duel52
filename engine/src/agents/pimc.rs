//! Perfect-information Monte Carlo — the control, not the target.
//!
//! `PLAN.md` Phase 2, rung four; `DESIGN.md` §6: "Baseline to beat: **PIMC** (perfect-
//! information Monte Carlo) — cheaper, and known to suffer strategy fusion, which makes it a
//! useful control rather than a target."
//!
//! The algorithm is three lines: sample `worlds` determinizations of the acting player's
//! information set; inside each one run an ordinary alpha–beta search as though the game
//! were open information; average each action's value over the worlds and play the best
//! average.
//!
//! # Strategy fusion, and why we want an agent that has it
//!
//! Averaging *after* solving each world means PIMC implicitly assumes it will know the
//! hidden cards by the time it has to act again. So it will happily pick a line that wins
//! against world A by flipping the King and wins against world B by holding it — even though
//! at the real decision node it has to do one or the other. It also never pays for
//! information and never conceals any, because in a world where everything is visible
//! neither has value.
//!
//! That is a *named, well-understood* failure, which is exactly what makes it worth
//! building. ISMCTS differs from PIMC in precisely one respect — it keeps statistics at the
//! information-set level instead of solving worlds independently — so the Elo gap between
//! them on the Phase 2 ladder measures the cost of strategy fusion in this game, on this
//! engine, with everything else held fixed. If that gap turns out to be near zero, that is a
//! finding about Duel 52 worth knowing before Phase 3 spends compute on belief modelling.
//!
//! # Depth, and the node budget
//!
//! Depth counts **decision nodes**, not turns, and a turn is three actions
//! (`game_rules.md` §4) plus whatever free sub-decisions the powers open. The default of 1
//! is therefore "my action, then the best answer available at the next node".
//!
//! A full search costs `worlds · b^(depth+1)` where `b` is the branching factor, and in this
//! game `b` is not a constant: `FINDINGS.md` F1.7 records **20 cards on one side of one
//! lane** under random play, and a lane that wide offers hundreds of attack actions alone.
//! Left uncapped, one such node costs more than the rest of the game put together, which
//! makes a ladder unrunnable and — worse — makes PIMC's effective budget depend on how
//! sprawling its *opponent* likes to leave the board.
//!
//! So the search takes a [`PimcAgent::NODE_BUDGET`] on expansions per decision and trims the
//! number of worlds to fit, never below one. It is a function of the branching factor alone,
//! so it stays deterministic. At ordinary widths it does not bind at all; it exists to bound
//! the tail.

use crate::action::Action;
use crate::agents::eval::{evaluate, EvalWeights, View};
use crate::agents::{pick_best, Agent};
use crate::player::Player;
use crate::rng::Rng;
use crate::state::GameState;

/// Samples worlds, solves each one shallowly as though it were open information, and plays
/// the action with the best average.
#[derive(Clone, Debug)]
pub struct PimcAgent {
    rng: Rng,
    worlds: usize,
    depth: u32,
    weights: EvalWeights,
}

impl PimcAgent {
    /// Time-matched to `flatmc:600` and `ismcts:800` within a factor of about two, which is
    /// what makes the ladder's ISMCTS−PIMC gap readable as strategy fusion rather than as a
    /// compute difference. See `FINDINGS.md` F2.2 for the budget-scaling measurement that
    /// picked it.
    pub const DEFAULT_WORLDS: usize = 32;
    pub const DEFAULT_DEPTH: u32 = 1;
    /// Ceiling on child expansions per decision. Roughly the number of engine steps
    /// `ismcts:800` spends, so the cap is generous at ordinary branching factors and only
    /// bites on the very wide nodes described in the module docs.
    pub const NODE_BUDGET: usize = 250_000;

    pub fn new(seed: u64, worlds: usize, depth: u32) -> PimcAgent {
        PimcAgent {
            rng: Rng::new(seed),
            worlds,
            depth,
            weights: EvalWeights::default(),
        }
    }

    pub fn derived(seed: u64, stream: u64, worlds: usize, depth: u32) -> PimcAgent {
        PimcAgent {
            rng: Rng::derive(seed, stream),
            worlds,
            depth,
            weights: EvalWeights::default(),
        }
    }
}

impl Agent for PimcAgent {
    fn choose(&mut self, state: &GameState, legal: &[Action]) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let me = state.acting_player();
        let mut totals = vec![0.0f64; legal.len()];
        let worlds = self.affordable_worlds(legal.len());

        for _ in 0..worlds {
            let world = state.determinize(me, &mut self.rng);
            for (i, &action) in legal.iter().enumerate() {
                let mut child = world.clone();
                child.apply_trusted(action);
                let value = alpha_beta(
                    &child,
                    me,
                    self.depth,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    &self.weights,
                );
                totals[i] += value as f64;
            }
        }

        let means: Vec<f32> = totals.iter().map(|&t| (t / worlds as f64) as f32).collect();
        legal[pick_best(&means, &mut self.rng)]
    }

    fn name(&self) -> String {
        format!("pimc:{}x{}", self.worlds, self.depth)
    }
}

impl PimcAgent {
    /// How many worlds [`PimcAgent::NODE_BUDGET`] leaves room for at this branching factor.
    ///
    /// The estimate `b^(depth + 1)` ignores alpha–beta cutoffs, so it is an upper bound on
    /// the real cost and the trim is conservative. Never returns zero: one world of PIMC is
    /// still PIMC, and a decision has to be made.
    fn affordable_worlds(&self, branching: usize) -> usize {
        let per_world = (branching as u64)
            .saturating_pow(self.depth + 1)
            .max(1);
        let affordable = (PimcAgent::NODE_BUDGET as u64 / per_world) as usize;
        affordable.clamp(1, self.worlds)
    }
}

/// Alpha–beta over decision nodes, valued from `me`'s point of view.
///
/// **Only sound on a determinized world.** The leaf evaluation runs under
/// [`View::Omniscient`], which reads every rank on the board — legitimate here because the
/// hidden ones were *sampled* rather than observed, and a lie anywhere else.
///
/// The player to move alternates irregularly rather than every node: a turn is three
/// actions, and the free sub-decisions a power opens all belong to the player whose action
/// opened them (`DESIGN.md` §4). So the search reads `acting_player()` at every node instead
/// of assuming strict alternation, and maximises or minimises accordingly.
fn alpha_beta(
    state: &GameState,
    me: Player,
    depth: u32,
    mut alpha: f32,
    mut beta: f32,
    weights: &EvalWeights,
) -> f32 {
    if state.outcome.is_over() {
        return state.outcome.value_for(me);
    }
    if depth == 0 {
        return evaluate(state, me, View::Omniscient, weights);
    }

    let maximizing = state.acting_player() == me;
    let legal = state.legal_actions();

    // Expand once and keep the static score alongside each child: at depth 1 that score
    // *is* the answer, and deeper it is the move ordering that makes the cutoffs work.
    let mut children: Vec<(f32, GameState)> = legal
        .into_iter()
        .map(|action| {
            let mut child = state.clone();
            child.apply_trusted(action);
            let score = evaluate(&child, me, View::Omniscient, weights);
            (score, child)
        })
        .collect();

    if children.is_empty() {
        // Unreachable while the game is running — `legal_actions` is empty only once the
        // game is over — but a bottomed-out search must still return something finite.
        return evaluate(state, me, View::Omniscient, weights);
    }

    if depth == 1 {
        return children
            .iter()
            .map(|(score, _)| *score)
            .fold(if maximizing { f32::MIN } else { f32::MAX }, |acc, s| {
                if maximizing {
                    acc.max(s)
                } else {
                    acc.min(s)
                }
            });
    }

    children.sort_by(|a, b| {
        if maximizing {
            b.0.partial_cmp(&a.0).expect("evaluation is never NaN")
        } else {
            a.0.partial_cmp(&b.0).expect("evaluation is never NaN")
        }
    });

    let mut best = if maximizing {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    for (_, child) in &children {
        let value = alpha_beta(child, me, depth - 1, alpha, beta, weights);
        if maximizing {
            best = best.max(value);
            alpha = alpha.max(best);
        } else {
            best = best.min(value);
            beta = beta.min(best);
        }
        if beta <= alpha {
            break;
        }
    }
    best
}
