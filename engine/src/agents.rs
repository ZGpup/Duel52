//! Agents, and the frozen Phase 2 ladder.
//!
//! `PLAN.md` Phase 2 names five rungs, in increasing order of what they are allowed to
//! think with:
//!
//! | Rung | Module | Sees | Spends |
//! |---|---|---|---|
//! | [`RandomAgent`] | here | nothing | nothing |
//! | [`greedy::GreedyAgent`] | [`greedy`] | its own information set | one ply, a hand-written evaluation |
//! | [`flat_mc::FlatMcAgent`] | [`flat_mc`] | sampled worlds | random playouts, no tree |
//! | [`pimc::PimcAgent`] | [`pimc`] | sampled worlds, *fully* | alpha–beta per world |
//! | [`ismcts::IsmctsAgent`] | [`ismcts`] | sampled worlds | a tree over information sets |
//!
//! The rungs are chosen so that each pair differs in exactly one thing, which is what makes
//! the Elo table in [`crate::ladder`] readable as an experiment rather than a leaderboard:
//! greedy over random isolates a hand-written evaluation, flat MC over greedy isolates
//! playouts, ISMCTS over flat MC isolates the tree, and ISMCTS over PIMC isolates
//! information-set reasoning from strategy fusion.
//!
//! # A note on what an agent is allowed to see
//!
//! [`Agent::choose`] receives the full [`GameState`], which is engine-side ground truth — it
//! contains the opponent's hand and the draw pile order. A *correct* agent must not read
//! those fields; the search agents here reach hidden state only through
//! [`GameState::determinize`], which resamples it from the acting player's information set.
//!
//! This used to be an honour system. It is now a test: because a determinized state is by
//! construction in the same information set as the real one, an honest agent handed either
//! must make the same decision. `engine/tests/agents.rs` checks exactly that for every rung.

pub mod eval;
pub mod flat_mc;
pub mod greedy;
pub mod ismcts;
pub mod net_mcts;
pub mod net_policy;
pub mod pimc;

use crate::action::Action;
use crate::rng::Rng;
use crate::state::GameState;

pub use flat_mc::FlatMcAgent;
pub use greedy::GreedyAgent;
pub use ismcts::IsmctsAgent;
pub use net_mcts::{NetMctsAgent, RootNoise, SearchResult};
pub use net_policy::NetPolicyAgent;
pub use pimc::PimcAgent;

/// Something that picks an action.
pub trait Agent {
    /// Pick one of `legal`. `legal` is never empty when the game is still running.
    fn choose(&mut self, state: &GameState, legal: &[Action]) -> Action;

    /// Name for Elo tables and logs. Includes the search budget where there is one, so a
    /// result row identifies the agent that produced it rather than just its family.
    fn name(&self) -> String;
}

/// Picks uniformly at random from the legal actions.
///
/// The reference baseline for the Phase 2 ladder, the source of the Phase 1 statistics, and
/// the rollout policy inside [`flat_mc`] and [`ismcts`]. Seeded, so a random-vs-random game
/// is as reproducible as any other.
///
/// One consequence worth keeping in mind when reading Phase 1 numbers: it attacks far more
/// often than a human would, and plays out its hand for no reason. Random-play statistics
/// describe the *game tree*, not the game as played. It cannot, however, stall: §4 makes
/// acting mandatory, so a random agent spends every action it has.
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

/// Play out a position to the end with uniformly random moves on both sides.
///
/// The rollout policy for [`flat_mc`] and [`ismcts`]. `PLAN.md` Phase 2 specifies "SO-ISMCTS
/// with random rollouts" deliberately: a heuristic rollout policy would fold the
/// hand-written evaluation back into the two rungs that are supposed to be free of it, and
/// the ladder would stop measuring what it is built to measure.
pub fn random_playout(state: &mut GameState, rng: &mut Rng) {
    while !state.outcome.is_over() {
        let legal = state.legal_actions();
        let action = *rng
            .choose(&legal)
            .expect("legal_actions is non-empty while the game is running");
        state.apply_trusted(action);
    }
}

/// Index of the highest score, breaking ties uniformly at random.
///
/// Random tie-breaking is not cosmetic. Duel 52 positions are full of exactly-equivalent
/// actions — three lanes that are mirror images early, two identical face-down cards — and
/// a first-index tie-break makes a deterministic agent play the same game against itself
/// every time, which would quietly turn a self-play measurement into a sample of one.
pub(crate) fn pick_best(scores: &[f32], rng: &mut Rng) -> usize {
    debug_assert!(!scores.is_empty(), "pick_best on an empty score list");
    let mut best = f32::NEG_INFINITY;
    for &s in scores {
        if s > best {
            best = s;
        }
    }
    // Reservoir sampling over the tied maxima: one pass, no allocation.
    let mut chosen = 0usize;
    let mut seen = 0u64;
    for (i, &s) in scores.iter().enumerate() {
        if s >= best {
            seen += 1;
            if rng.below(seen) == 0 {
                chosen = i;
            }
        }
    }
    chosen
}

/// A named agent configuration, parseable from the command line.
///
/// Exists so that the ladder, the `probe` command and `play --opponent` all name agents the
/// same way, and so a result row in `FINDINGS.md` can be re-run by pasting its agent name
/// back into the CLI.
/// Not `Copy`: [`AgentSpec::NetPolicy`] carries a checkpoint path. `Clone` is enough
/// everywhere — a spec is cloned once per game at most, next to the cost of playing it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AgentSpec {
    Random,
    Greedy,
    FlatMc { playouts: usize },
    Pimc { worlds: usize, depth: u32 },
    Ismcts { iterations: usize },
    /// Phase 3's policy-only rung: a checkpoint played by argmax, no search.
    NetPolicy { checkpoint: String },
    /// Phase 3's real rung: net-guided ISMCTS. PUCT over the policy prior, the value head in
    /// place of rollouts.
    NetMcts { checkpoint: String, sims: usize },
}

impl AgentSpec {
    /// **The frozen Phase 2 benchmark ladder.**
    ///
    /// `PLAN.md`: "Round-robin Elo ladder, frozen as the permanent benchmark." Every later
    /// phase measures against *these* configurations, so changing a budget here invalidates
    /// every Elo number already recorded. Add a rung rather than editing one.
    pub const LADDER: [AgentSpec; 5] = [
        AgentSpec::Random,
        AgentSpec::Greedy,
        AgentSpec::FlatMc {
            playouts: FlatMcAgent::DEFAULT_PLAYOUTS,
        },
        AgentSpec::Pimc {
            worlds: PimcAgent::DEFAULT_WORLDS,
            depth: PimcAgent::DEFAULT_DEPTH,
        },
        AgentSpec::Ismcts {
            iterations: IsmctsAgent::DEFAULT_ITERATIONS,
        },
    ];

    /// Parse `family` or `family:budget`, e.g. `greedy`, `flatmc:600`, `pimc:8x2`,
    /// `ismcts:1600`, `netpolicy:runs/gen3.d52nn`.
    ///
    /// Only the **family** is lowercased. It used to be the whole string, which was
    /// harmless while every budget was a number — and would silently corrupt a checkpoint
    /// path the moment `netpolicy:` arrived. A path is taken verbatim, colons included, so
    /// an absolute Windows-style path survives too.
    pub fn parse(text: &str) -> Result<AgentSpec, String> {
        let text = text.trim();
        let (family, budget) = match text.split_once(':') {
            Some((f, b)) => (f.to_ascii_lowercase(), Some(b)),
            None => (text.to_ascii_lowercase(), None),
        };
        let family = family.as_str();

        fn number(what: &str, value: &str) -> Result<usize, String> {
            value
                .parse::<usize>()
                .map_err(|_| format!("{what}: `{value}` is not a number"))
        }

        match family {
            "random" => Ok(AgentSpec::Random),
            "greedy" => Ok(AgentSpec::Greedy),
            "flatmc" | "flat" | "mc" => Ok(AgentSpec::FlatMc {
                playouts: match budget {
                    Some(b) => number("flatmc playouts", b)?,
                    None => FlatMcAgent::DEFAULT_PLAYOUTS,
                },
            }),
            "pimc" => {
                let (worlds, depth) = match budget {
                    None => (PimcAgent::DEFAULT_WORLDS, PimcAgent::DEFAULT_DEPTH),
                    Some(b) => match b.split_once('x') {
                        Some((w, d)) => (number("pimc worlds", w)?, number("pimc depth", d)? as u32),
                        None => (number("pimc worlds", b)?, PimcAgent::DEFAULT_DEPTH),
                    },
                };
                if worlds == 0 {
                    return Err("pimc needs at least one world".into());
                }
                Ok(AgentSpec::Pimc { worlds, depth })
            }
            "ismcts" | "is" => Ok(AgentSpec::Ismcts {
                iterations: match budget {
                    Some(b) => number("ismcts iterations", b)?,
                    None => IsmctsAgent::DEFAULT_ITERATIONS,
                },
            }),
            "netpolicy" | "net" => match budget {
                Some(path) if !path.trim().is_empty() => Ok(AgentSpec::NetPolicy {
                    checkpoint: path.to_string(),
                }),
                _ => Err(
                    "netpolicy needs a checkpoint path, e.g. `netpolicy:checkpoints/init.d52nn`"
                        .to_string(),
                ),
            },
            // `netmcts:<path>@<sims>`. Split from the *right*, because the path is taken
            // verbatim and may itself contain an `@` — a name like `gen7@2.d52nn` is not
            // the caller's mistake to pay for.
            "netmcts" | "nmcts" => {
                let budget = budget.filter(|b| !b.trim().is_empty()).ok_or_else(|| {
                    "netmcts needs a checkpoint path, e.g. \
                     `netmcts:checkpoints/gen7.d52nn@128`"
                        .to_string()
                })?;
                let (checkpoint, sims) = match budget.rsplit_once('@') {
                    Some((path, s)) => (path.to_string(), number("netmcts sims", s)?),
                    None => (budget.to_string(), NetMctsAgent::DEFAULT_SIMS),
                };
                if checkpoint.trim().is_empty() {
                    return Err("netmcts needs a checkpoint path before the `@`".to_string());
                }
                if sims == 0 {
                    return Err("netmcts needs at least one simulation".to_string());
                }
                Ok(AgentSpec::NetMcts { checkpoint, sims })
            }
            other => Err(format!(
                "unknown agent `{other}` — expected random, greedy, flatmc, pimc, ismcts, \
                 netpolicy or netmcts"
            )),
        }
    }

    /// The name this configuration reports, without building it.
    pub fn name(&self) -> String {
        match self {
            AgentSpec::Random => "random".to_string(),
            AgentSpec::Greedy => "greedy".to_string(),
            AgentSpec::FlatMc { playouts } => format!("flatmc:{playouts}"),
            AgentSpec::Pimc { worlds, depth } => format!("pimc:{worlds}x{depth}"),
            AgentSpec::Ismcts { iterations } => format!("ismcts:{iterations}"),
            AgentSpec::NetPolicy { checkpoint } => format!("netpolicy:{checkpoint}"),
            AgentSpec::NetMcts { checkpoint, sims } => format!("netmcts:{checkpoint}@{sims}"),
        }
    }

    /// Build an instance whose randomness is derived from `(seed, stream)`, so a whole
    /// match is reproducible from the game seed alone.
    ///
    /// `Send + Sync` so an agent can be owned by a ladder worker thread or by a Python
    /// object without the caller having to prove it. Every agent here is plain data plus an
    /// [`Rng`], so both hold trivially; an agent that needed interior mutability would have
    /// to use a lock rather than a `RefCell`, which is the right trade for this project.
    pub fn build(&self, seed: u64, stream: u64) -> Box<dyn Agent + Send + Sync> {
        match self {
            AgentSpec::Random => Box::new(RandomAgent::derived(seed, stream)),
            AgentSpec::Greedy => Box::new(GreedyAgent::derived(seed, stream)),
            AgentSpec::FlatMc { playouts } => {
                Box::new(FlatMcAgent::derived(seed, stream, *playouts))
            }
            AgentSpec::Pimc { worlds, depth } => {
                Box::new(PimcAgent::derived(seed, stream, *worlds, *depth))
            }
            AgentSpec::Ismcts { iterations } => {
                Box::new(IsmctsAgent::derived(seed, stream, *iterations))
            }
            // No seed: the policy is played by argmax, so it consumes no randomness. The
            // checkpoint is read on the first decision, once per process — see
            // `net_policy::load_cached`.
            AgentSpec::NetPolicy { checkpoint } => Box::new(NetPolicyAgent::new(checkpoint)),
            // No root noise: this is the evaluation build. Self-play adds it explicitly, in
            // `selfplay.rs`, because a benchmark agent must play its best move.
            AgentSpec::NetMcts { checkpoint, sims } => {
                Box::new(NetMctsAgent::derived(checkpoint, seed, stream, *sims))
            }
        }
    }
}

impl std::fmt::Display for AgentSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
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
