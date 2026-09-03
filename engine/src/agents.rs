//! Agents.
//!
//! Phase 1 needs exactly one: a uniform random player, for the random-vs-random statistics
//! that are this phase's deliverable and for the CLI's practice opponent. The [`Agent`]
//! trait is the seam Phase 2's greedy / flat-MC / PIMC / ISMCTS players will slot into
//! (`PLAN.md` Phase 2), and it is deliberately minimal: an agent sees the state and the
//! legal actions and returns one of them.
//!
//! # A note on what an agent is allowed to see
//!
//! `choose` receives the full [`GameState`], which is engine-side ground truth — it
//! contains the opponent's hand and the draw pile order. A *correct* agent must not read
//! those fields. Phase 3's search will be handed a per-observer view instead
//! (`DESIGN.md` §6, determinization), but until that exists the honour system plus this
//! comment is the guard. `RandomAgent` reads nothing at all, so it cannot cheat.

use crate::action::Action;
use crate::rng::Rng;
use crate::state::GameState;

/// Something that picks an action.
pub trait Agent {
    /// Pick one of `legal`. `legal` is never empty when the game is still running.
    fn choose(&mut self, state: &GameState, legal: &[Action]) -> Action;

    /// Name for Elo tables and logs.
    fn name(&self) -> String;
}

/// Picks uniformly at random from the legal actions.
///
/// The reference baseline for the Phase 2 ladder, and the source of the Phase 1
/// statistics. Seeded, so a random-vs-random game is as reproducible as any other.
///
/// One consequence worth keeping in mind when reading Phase 1 numbers: `Pass` is one legal
/// action among typically dozens, so a random agent forfeits the rest of its turn a few
/// percent of the time, and it attacks far more often than a human would. Random-play
/// statistics describe the *game tree*, not the game as played.
#[derive(Clone, Debug)]
pub struct RandomAgent {
    rng: Rng,
}

impl RandomAgent {
    pub fn new(seed: u64) -> RandomAgent {
        RandomAgent {
            rng: Rng::new(seed),
        }
    }

    /// Build from a game seed plus a stream tag, so both players' choices and the deal are
    /// independent streams of one seed.
    pub fn derived(seed: u64, stream: u64) -> RandomAgent {
        RandomAgent {
            rng: Rng::derive(seed, stream),
        }
    }
}

impl Agent for RandomAgent {
    fn choose(&mut self, _state: &GameState, legal: &[Action]) -> Action {
        *self
            .rng
            .choose(legal)
            .expect("legal_actions is non-empty while the game is running")
    }

    fn name(&self) -> String {
        "random".to_string()
    }
}

/// Play one full game between two agents and return the final state.
///
/// Uses [`GameState::apply_trusted`], so the actions must come from
/// [`GameState::legal_actions`] — which they do, since that is what is handed to the agent.
pub fn play_game(state: &mut GameState, p0: &mut dyn Agent, p1: &mut dyn Agent) {
    while !state.outcome.is_over() {
        let legal = state.legal_actions();
        debug_assert!(
            !legal.is_empty(),
            "no legal actions but the game is not over: {}",
            state.header()
        );
        let action = match state.to_move {
            crate::player::Player::P0 => p0.choose(state, &legal),
            crate::player::Player::P1 => p1.choose(state, &legal),
        };
        state.apply_trusted(action);
    }
}
