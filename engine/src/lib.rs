//! # Duel 52 — rules-exact engine
//!
//! An engine for the card game **Duel 52** (Judd Madden & Nina Riddell, 2017), built to
//! answer a question nobody has published an answer to: *what does optimal play actually
//! look like?*
//!
//! The specification this implements is `game_rules.md` at the repository root. That
//! document, not this code, is the authority: it is a disambiguated, engine-ready version
//! of the published rules with the project owner's rulings folded in. Every non-obvious
//! branch below cites the section it comes from.
//!
//! ## The five things that are easy to get wrong
//!
//! Repeated from `CLAUDE.md`, because they shape the whole design:
//!
//! 1. **Base cards are hidden from their owner too.** That is why a 4's Foresight can
//!    usefully target your own base cards. See [`card::Card::known_to`].
//! 2. **Lane wins are endgame-only.** A lane cannot be won until the draw pile *and* the
//!    opponent's hand are both empty, so the entire draw phase is positioning. See
//!    [`state::GameState::lanes_won_by`].
//! 3. **10 cards are removed unseen at setup.** Belief over hidden cards never fully
//!    resolves. See [`state::GameState::removed`].
//! 4. **Suits are mechanically irrelevant** — collapsed to rank everywhere. See [`rank`].
//! 5. **The split-deck variant is the default configuration.** See [`config::Variant`].
//!
//! ## Tour of the code
//!
//! | Module | What lives there |
//! |---|---|
//! | [`rank`] | Ranks, hit points, which powers are constant or King-reactivatable |
//! | [`player`] | The two players |
//! | [`config`] | [`config::GameConfig`] — every tunable, and the three variant presets |
//! | [`rng`] | The frozen, self-contained PRNG that makes seeds reproducible forever |
//! | [`card`] | [`card::Card`] — one card instance on the table |
//! | [`state`] | [`state::GameState`], and the queries the rules are written in terms of |
//! | [`action`] | [`action::Action`] — what a player can choose, including sub-decisions |
//! | [`outcome`] | How a game ends, and why a draw was declared |
//! | `setup` | Dealing, per variant (`impl GameState`) |
//! | `legal` | Legal-action enumeration (`impl GameState`) |
//! | `apply` | Powers, combat, and the turn machinery (`impl GameState`) |
//! | [`display`] | Rendering a board for a specific observer, without leaking |
//! | [`agents`] | A random agent, and the trait a Phase 2 agent implements |
//! | [`stats`] | Random-vs-random measurement — the Phase 1 deliverable |
//! | [`testkit`] | Building positions by hand, for the rules tests and Phase 4 probes |
//!
//! ## Playing a game
//!
//! ```
//! use duel52_engine::{Agent, GameConfig, GameState, RandomAgent};
//!
//! // Same seed + same config always produces the same game.
//! let mut state = GameState::new(GameConfig::split_deck(), 42);
//! let mut p0 = RandomAgent::new(1);
//! let mut p1 = RandomAgent::new(2);
//!
//! while !state.outcome.is_over() {
//!     let actions = state.legal_actions();
//!     let choice = match state.to_move {
//!         duel52_engine::Player::P0 => p0.choose(&state, &actions),
//!         duel52_engine::Player::P1 => p1.choose(&state, &actions),
//!     };
//!     state.apply_trusted(choice);
//! }
//! assert!(state.outcome.is_over());
//! ```

pub mod action;
pub mod agents;
pub mod card;
pub mod config;
pub mod display;
pub mod outcome;
pub mod player;
pub mod rank;
pub mod rng;
pub mod state;
pub mod stats;
pub mod testkit;

// These modules are `impl GameState` blocks rather than new types, so they have nothing to
// export. They stay private and their contents surface as methods on `GameState`.
mod apply;
mod legal;
mod setup;

pub use action::{Action, IllegalAction, Phase, Side};
pub use agents::{Agent, RandomAgent};
pub use card::{Card, CardId, PairId};
pub use config::{GameConfig, TwoPower, Variant};
pub use outcome::{DrawReason, Outcome};
pub use player::Player;
pub use rank::{rank_counts, Rank, RankCounts};
pub use rng::Rng;
pub use state::{GameState, Lane, Pending, Pile, ResolveKind};
pub use stats::{run_random_games, GameSummary, RandomPlayStats};

/// The engine version, as reported by the CLI and stamped into results files.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
