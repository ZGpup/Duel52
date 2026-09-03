//! A one-ply greedy agent.
//!
//! `PLAN.md` Phase 2, rung two: "Greedy heuristic agent (hand-written evaluation)."
//!
//! It applies every legal action, resolves whatever sub-decisions that action opens (also
//! greedily), and keeps the one whose resulting position scores highest under
//! [`crate::agents::eval`]. Ties break at random, so two greedy agents facing each other do
//! not play the same game forever.
//!
//! # Why it resolves sub-decisions before scoring
//!
//! `DESIGN.md` §4 makes the sub-choices a power opens into separate decision nodes, which is
//! right for search but hostile to naive one-ply evaluation: flipping a 5 changes *nothing*
//! on the board until its flip list is worked through, so an agent that scored the position
//! immediately after the `Flip` would rate every 5 as worthless. Playing the action out to
//! the next real decision is what makes "one ply" mean one *action* rather than one node.
//!
//! # Why even one-ply lookahead has to determinize
//!
//! It is tempting to think a greedy agent needs no sampled world, because
//! [`View::Informed`] already refuses to read a rank the acting player does not know. That
//! is not enough, and the reason is worth stating because it is easy to get wrong: greedy
//! does not evaluate the *current* position, it evaluates the position **after** applying a
//! candidate action — and applying an action to the engine's real state reveals things.
//!
//! - Flipping your own base card turns it face-up, and §3 says you did not know what it was.
//! - Killing a face-down card sends its rank to the discard pile, which §5 makes public.
//! - Killing a face-down 3 springs the Trap and reveals it (§6).
//!
//! One-ply lookahead on ground truth therefore lets the agent see the answer before
//! committing to the question. Sampling a world first replaces that with a draw from its
//! belief, which is what a player actually has. This was caught by
//! `phase2_no_agent_reads_hidden_information` rather than by inspection, which is the whole
//! argument for having that test.
//!
//! One world per decision, not several: greedy is the cheap rung, and averaging over worlds
//! is what the rung above it is for. The consequence is that greedy's read of an *unknown*
//! flip is noisy but unbiased.

use crate::action::Action;
use crate::agents::eval::{evaluate, EvalWeights, View};
use crate::agents::{pick_best, Agent};
use crate::player::Player;
use crate::rng::Rng;
use crate::state::GameState;

/// A sanity bound on how many free sub-decisions one action can open.
///
/// A 5 flipping a King that re-empowers the lane is the deep case the rules actually
/// produce, and it is nowhere near this. The cap exists so that a rules bug which left a
/// sub-decision permanently on the stack fails loudly in a search agent rather than hanging
/// a training run.
const MAX_SUB_DECISIONS: usize = 256;

/// Picks the action whose immediate result scores best under the hand-written evaluation.
#[derive(Clone, Debug)]
pub struct GreedyAgent {
    rng: Rng,
    weights: EvalWeights,
}

impl GreedyAgent {
    pub fn new(seed: u64) -> GreedyAgent {
        GreedyAgent {
            rng: Rng::new(seed),
            weights: EvalWeights::default(),
        }
    }

    pub fn derived(seed: u64, stream: u64) -> GreedyAgent {
        GreedyAgent {
            rng: Rng::derive(seed, stream),
            weights: EvalWeights::default(),
        }
    }

    pub fn with_weights(mut self, weights: EvalWeights) -> GreedyAgent {
        self.weights = weights;
        self
    }
}

impl Agent for GreedyAgent {
    fn choose(&mut self, state: &GameState, legal: &[Action]) -> Action {
        let me = state.acting_player();
        let view = View::Informed(me);
        // See the module docs: the lookahead, not the evaluation, is what would leak.
        let world = state.determinize(me, &mut self.rng);
        let scores: Vec<f32> = legal
            .iter()
            .map(|&action| {
                let mut next = world.clone();
                next.apply_trusted(action);
                resolve_sub_decisions_greedily(&mut next, me, view, &self.weights);
                evaluate(&next, me, view, &self.weights)
            })
            .collect();
        legal[pick_best(&scores, &mut self.rng)]
    }

    fn name(&self) -> String {
        "greedy".to_string()
    }
}

/// Answer every free sub-decision `me` currently owes, taking the locally best answer each
/// time.
///
/// Stops as soon as control returns to a real action, passes to the opponent, or the game
/// ends. Each answer is chosen by immediate evaluation without looking further ahead, which
/// matches `game_rules.md` §8's *adaptive* resolution order: the choice is made after seeing
/// the previous power land, not planned as a permutation up front.
pub(crate) fn resolve_sub_decisions_greedily(
    state: &mut GameState,
    me: Player,
    view: View,
    weights: &EvalWeights,
) {
    let mut guard = 0;
    while !state.outcome.is_over() && state.in_sub_decision() && state.acting_player() == me {
        guard += 1;
        assert!(
            guard <= MAX_SUB_DECISIONS,
            "one action opened more than {MAX_SUB_DECISIONS} sub-decisions, which the rules \
             cannot produce: {}",
            state.header()
        );

        let legal = state.legal_actions();
        debug_assert!(
            !legal.is_empty(),
            "a live sub-decision with no answer: {}",
            state.header()
        );

        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, &action) in legal.iter().enumerate() {
            let mut probe = state.clone();
            probe.apply_trusted(action);
            let score = evaluate(&probe, me, view, weights);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        state.apply_trusted(legal[best]);
    }
}
