//! Net-guided SO-ISMCTS — `PLAN.md` Phase 3, step 2.
//!
//! [`super::ismcts`] with the two AlphaZero substitutions, and nothing else changed:
//!
//! | | rollout ISMCTS | this |
//! |---|---|---|
//! | which action to try | UCB1 over availability | **PUCT over a network prior** |
//! | what a leaf is worth | a uniform-random playout | **the value head** |
//!
//! Everything that makes the Phase 2 agent an *information-set* search is kept verbatim: a
//! fresh determinization per simulation, one tree keyed by action sequence, per-edge
//! availability counts, and a reward banked for both players because the mover at a node is
//! a property of the sampled world rather than of the tree. [`super::ismcts`]'s module docs
//! are the argument for all four and are not repeated here.
//!
//! # PUCT, with availability in place of parent visits
//!
//! ```text
//! score(e) = Q(e)  +  c_puct · P(e) · sqrt(availability(e)) / (1 + visits(e))
//! ```
//!
//! Standard PUCT puts `sqrt(N_parent)` in the numerator: the number of chances this edge has
//! had to be chosen. Under determinization that number is not the parent's visit count —
//! an action legal in only half the sampled worlds was never on the menu for the other half,
//! and charging it for those visits is the same mistake UCB1 makes without an availability
//! count. `availability(e)` *is* the number of chances, so it is what goes in the numerator.
//! When every action is legal in every world the two are equal and this reduces exactly to
//! AlphaZero's rule. `rule_6_netmcts_puct_uses_availability_not_parent_visits` pins it.
//!
//! # Priors are stored as logits, and renormalised over what is legal
//!
//! An edge remembers the **raw logit** the network gave its action, not a probability. The
//! legal set at a node changes from one determinization to the next, so a probability
//! normalised at expansion time would be normalised over the wrong support on most later
//! visits. Softmaxing the available edges' logits at selection time is exact for whatever
//! subset shows up.
//!
//! The honest caveat: logits recorded under different determinizations come from different
//! observations, because a node deeper than the root can have the *opponent* to move and
//! their observation depends on hidden cards. They are the same network on nearly the same
//! position, so mixing them adds noise to the prior and nothing worse — and the alternative,
//! storing a whole `action_dim` policy vector per node, costs 8.8 KB a node.
//!
//! # Where the evaluations go
//!
//! One forward pass per simulation, at the leaf, and none anywhere else — no rollout, so
//! this is *cheaper* per simulation than `ismcts` once the network is small. The pass is
//! [`MlpEvaluator::eval_masked_with`], which computes only the ~21 logits the position
//! actually offers out of the 2195 the head could produce.
//!
//! # The value head is zero-sum and the game, once stalemates are penalised, is not
//!
//! [`GameConfig::learning_value`] gives a stalemate the same low value to *both* players
//! (`FINDINGS.md` F3.6), so a terminal backup here is deliberately not zero-sum — each edge
//! banks a reward per player, which handles it exactly. A *leaf* estimate cannot: the network
//! emits one scalar for the player to move, and this file completes it as `1 − v` for the
//! opponent because that is the only thing one number can say.
//!
//! The approximation is one-sided in the useful direction. The value head is trained on the
//! true targets, so it learns to report a low value at the decision nodes of *whoever is to
//! move* in a stall-prone position — and every node's own mover gets that estimate directly.
//! It is only the complement that is wrong, and terminal stalemates inside the search are
//! exact. A second value head would remove the approximation; it is not worth a checkpoint
//! format change until a measurement says it is.

use std::path::PathBuf;
use std::sync::Arc;

use crate::action::Action;
use crate::agents::Agent;
use crate::config::GameConfig;
use crate::encode::{action_dim, encode_action, encode_observation, obs_dim};
use crate::nn::{MlpEvaluator, Scratch};
use crate::player::Player;
use crate::rng::Rng;
use crate::state::GameState;

/// One action out of one information set.
#[derive(Clone, Debug)]
struct Edge {
    action: Action,
    child: usize,
    /// Simulations that traversed this edge.
    visits: u32,
    /// Simulations in which this action was legal at this node, taken or not.
    availability: u32,
    /// The raw policy logit, recorded when the edge was created. See the module docs for
    /// why this is a logit and not a probability.
    logit: f32,
    /// An independent Gamma(α) draw, used only at the root and only when root noise is on.
    /// Normalising these over the available edges gives a true Dirichlet(α) on that subset —
    /// see [`Rng::gamma`].
    gamma: f32,
    /// Total backed-up outcome in `0.0..=1.0`, indexed by player.
    reward: [f64; 2],
}

#[derive(Clone, Debug, Default)]
struct Node {
    edges: Vec<Edge>,
}

impl Node {
    fn edge_for(&self, action: Action) -> Option<usize> {
        self.edges.iter().position(|e| e.action == action)
    }
}

/// Root exploration noise: `P ← (1 − weight) · P + weight · Dirichlet(alpha)`.
///
/// Self-play only. `PLAN.md` Phase 3 step 3 needs it — without noise the search is a
/// deterministic function of the position and self-play collects the same game repeatedly —
/// and evaluation must not have it, because a benchmark agent should play its best move.
#[derive(Clone, Copy, Debug)]
pub struct RootNoise {
    pub alpha: f64,
    pub weight: f32,
}

impl RootNoise {
    /// AlphaZero's chess settings. `alpha` is usually tuned to roughly `10 / branching
    /// factor`; Duel 52 offers ~21 actions at a decision node (`FINDINGS.md` F3.3), which
    /// puts it in the same range as chess's 0.3.
    pub const DEFAULT: RootNoise = RootNoise {
        alpha: 0.3,
        weight: 0.25,
    };
}

/// What one search produced. More than `choose` needs, because self-play trains on it.
pub struct SearchResult {
    /// Visit count per entry of the `legal` slice the search was given. **This is the
    /// policy target** — `DESIGN.md` §6.
    pub visits: Vec<u32>,
    /// The root's backed-up value for the player to move, in `0.0..=1.0`.
    pub root_value: f32,
    /// Nodes in the tree, for instrumentation.
    pub nodes: usize,
}

/// Net-guided information-set MCTS.
pub struct NetMctsAgent {
    checkpoint: PathBuf,
    sims: usize,
    c_puct: f32,
    noise: Option<RootNoise>,
    rng: Rng,
    /// Resolved on the first decision: [`crate::AgentSpec::build`] does not know which
    /// configuration the agent will play under, and the layout hashes are config-derived.
    evaluator: Option<Arc<MlpEvaluator>>,
}

impl NetMctsAgent {
    pub const DEFAULT_SIMS: usize = 128;
    /// PUCT's exploration constant. AlphaZero's 1.25 in a `0..=1` reward scale, where the
    /// exploitation term spans half the range the `-1..=1` convention gives it.
    pub const DEFAULT_C_PUCT: f32 = 1.25;

    pub fn new(checkpoint: impl Into<PathBuf>, sims: usize) -> NetMctsAgent {
        NetMctsAgent {
            checkpoint: checkpoint.into(),
            sims,
            c_puct: NetMctsAgent::DEFAULT_C_PUCT,
            noise: None,
            rng: Rng::new(0),
            evaluator: None,
        }
    }

    /// Build with randomness derived from `(seed, stream)`, so a whole match reproduces from
    /// the game seed.
    pub fn derived(checkpoint: impl Into<PathBuf>, seed: u64, stream: u64, sims: usize) -> NetMctsAgent {
        NetMctsAgent {
            rng: Rng::derive(seed, stream),
            ..NetMctsAgent::new(checkpoint, sims)
        }
    }

    pub fn with_c_puct(mut self, c: f32) -> NetMctsAgent {
        self.c_puct = c;
        self
    }

    /// Turn on root Dirichlet noise. Self-play only — see [`RootNoise`].
    pub fn with_root_noise(mut self, noise: Option<RootNoise>) -> NetMctsAgent {
        self.noise = noise;
        self
    }

    /// Load the checkpoint if it is not loaded, or if the layout moved under us.
    fn evaluator(&mut self, config: &GameConfig) -> Arc<MlpEvaluator> {
        let wanted = obs_dim(config);
        if let Some(e) = &self.evaluator {
            if e.arch().obs_dim == wanted {
                return e.clone();
            }
        }
        let evaluator = crate::nn::evaluator_for(&self.checkpoint, config).unwrap_or_else(|e| {
            // `Agent::choose` cannot fail, and a checkpoint that will not load is a setup
            // error rather than a position to cope with — panic with the diagnosis in it
            // rather than falling back to something that plays badly for no visible reason.
            panic!("netmcts: {e}")
        });
        self.evaluator = Some(evaluator.clone());
        evaluator
    }

    /// Run the search. `legal` must be [`GameState::legal_actions`] for `state`.
    pub fn search(&mut self, state: &GameState, legal: &[Action]) -> SearchResult {
        let config = state.config;
        let evaluator = self.evaluator(&config);
        let me = state.acting_player();

        let mut buf = Buffers {
            obs: vec![0.0; obs_dim(&config)],
            logits: vec![0.0; action_dim(&config)],
            mask: vec![false; action_dim(&config)],
            set: Vec::with_capacity(64),
            scratch: evaluator.scratch(),
        };

        let mut tree: Vec<Node> = vec![Node::default()];
        let mut root_value_total = 0.0f64;
        let mut root_value_count = 0u32;
        // (node, edge) pairs traversed, reused across simulations.
        let mut path: Vec<(usize, usize)> = Vec::with_capacity(16);

        for _ in 0..self.sims {
            let mut world = state.determinize(me, &mut self.rng);
            let mut node = 0usize;
            path.clear();

            let reward = loop {
                if world.outcome.is_over() {
                    // `learning_value`, not `value_for`: an engine-declared stalemate is
                    // worth `config.stalemate_value` to *both* players rather than half a
                    // point each, so a search cannot treat "neither side attacks" as a safe
                    // half. `FINDINGS.md` F3.6 is what happens when it can. Not zero-sum,
                    // which is why each edge banks a reward per player.
                    break [
                        config.learning_value(world.outcome, Player::P0) as f64,
                        config.learning_value(world.outcome, Player::P1) as f64,
                    ];
                }
                let available = world.legal_actions();

                // Availability is credited to every legal action, taken or not, before
                // selection — so an edge created on this visit is counted for this visit.
                for &action in &available {
                    if let Some(i) = tree[node].edge_for(action) {
                        tree[node].edges[i].availability += 1;
                    }
                }

                let missing = available
                    .iter()
                    .any(|&a| tree[node].edge_for(a).is_none());
                if missing {
                    // Expansion: one forward pass, edges for every action this node has not
                    // seen before, and the value head in place of a rollout.
                    let value = evaluate(&evaluator, &world, &available, &mut buf);
                    for &action in &available {
                        if tree[node].edge_for(action).is_some() {
                            continue;
                        }
                        let child = tree.len();
                        tree.push(Node::default());
                        let logit = buf.logits[encode_action(&action, &world)];
                        let gamma = match (node, self.noise) {
                            (0, Some(n)) => self.rng.gamma(n.alpha) as f32,
                            _ => 0.0,
                        };
                        tree[node].edges.push(Edge {
                            action,
                            child,
                            visits: 0,
                            availability: 1,
                            logit,
                            gamma,
                            reward: [0.0, 0.0],
                        });
                    }
                    let actor = world.acting_player();
                    if node == 0 {
                        root_value_total += value as f64;
                        root_value_count += 1;
                    }
                    let mut reward = [0.0f64; 2];
                    reward[actor.idx()] = value as f64;
                    reward[actor.other().idx()] = 1.0 - value as f64;
                    break reward;
                }

                let mover = world.acting_player();
                let edge = self.select(&tree[node], &available, mover, node == 0);
                path.push((node, edge));
                let action = tree[node].edges[edge].action;
                node = tree[node].edges[edge].child;
                world.apply_trusted(action);
            };

            for &(n, e) in &path {
                let edge = &mut tree[n].edges[e];
                edge.visits += 1;
                edge.reward[0] += reward[0];
                edge.reward[1] += reward[1];
            }
        }

        let visits = legal
            .iter()
            .map(|&action| {
                tree[0]
                    .edge_for(action)
                    .map(|i| tree[0].edges[i].visits)
                    .unwrap_or(0)
            })
            .collect();

        // The root's own value, preferring what the tree backed up over the raw net call.
        let root_value = {
            let root = &tree[0];
            let total: f64 = root.edges.iter().map(|e| e.reward[me.idx()]).sum();
            let n: u32 = root.edges.iter().map(|e| e.visits).sum();
            if n > 0 {
                (total / n as f64) as f32
            } else if root_value_count > 0 {
                (root_value_total / root_value_count as f64) as f32
            } else {
                0.5
            }
        };

        SearchResult {
            visits,
            root_value,
            nodes: tree.len(),
        }
    }

    /// PUCT over the edges legal in this determinization.
    fn select(&self, node: &Node, available: &[Action], mover: Player, is_root: bool) -> usize {
        // Softmax over the available edges' logits: the prior, normalised over the support
        // that actually exists in this world.
        let mut max_logit = f32::NEG_INFINITY;
        for edge in &node.edges {
            if available.contains(&edge.action) && edge.logit > max_logit {
                max_logit = edge.logit;
            }
        }
        let mut prior_sum = 0.0f32;
        let mut gamma_sum = 0.0f32;
        for edge in &node.edges {
            if available.contains(&edge.action) {
                prior_sum += (edge.logit - max_logit).exp();
                gamma_sum += edge.gamma;
            }
        }
        let noise = if is_root { self.noise } else { None };

        let mut best = usize::MAX;
        let mut best_score = f32::NEG_INFINITY;
        for (i, edge) in node.edges.iter().enumerate() {
            if !available.contains(&edge.action) {
                continue;
            }
            let mut prior = (edge.logit - max_logit).exp() / prior_sum;
            if let Some(n) = noise {
                // Dirichlet over exactly the available subset — see `Edge::gamma`. A
                // degenerate `gamma_sum` (every draw underflowed) falls back to no noise
                // rather than to a division by zero.
                if gamma_sum > 0.0 {
                    prior = (1.0 - n.weight) * prior + n.weight * (edge.gamma / gamma_sum);
                }
            }
            // First-play urgency: an unvisited edge is scored at the neutral value, so the
            // prior and the exploration term decide the order in which edges are opened.
            let q = if edge.visits == 0 {
                0.5
            } else {
                (edge.reward[mover.idx()] / edge.visits as f64) as f32
            };
            let score =
                q + self.c_puct * prior * (edge.availability as f32).sqrt() / (1.0 + edge.visits as f32);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        debug_assert!(
            best != usize::MAX,
            "no edge matched a legal action, but every legal action was supposed to have one"
        );
        best
    }
}

/// Per-decision working buffers. Allocated once per search, not once per simulation.
struct Buffers {
    obs: Vec<f32>,
    logits: Vec<f32>,
    mask: Vec<bool>,
    /// The indices set in `mask`, so clearing it costs the same as setting it rather than a
    /// pass over all `action_dim` entries — and `encode_action` runs once per action per
    /// evaluation rather than twice.
    set: Vec<usize>,
    scratch: Scratch,
}

/// One forward pass on `world`, writing the logits of exactly `available` into `buf.logits`.
///
/// Returns the value head's output rescaled from the network's `(-1, 1)` to the engine's
/// `0.0..=1.0` outcome convention ([`crate::Outcome::value_for`]), so a backed-up net value
/// and a backed-up terminal result are the same quantity.
fn evaluate(
    evaluator: &MlpEvaluator,
    world: &GameState,
    available: &[Action],
    buf: &mut Buffers,
) -> f32 {
    encode_observation(world, world.acting_player(), &mut buf.obs);
    // Set the mask from the actions we already have rather than through `legal_mask`, which
    // would re-enumerate them and clear all `action_dim` entries. Cleared by index
    // afterwards, so the buffer is all-false again for the next simulation.
    buf.set.clear();
    for &action in available {
        let index = encode_action(&action, world);
        buf.mask[index] = true;
        buf.set.push(index);
    }
    let value = evaluator.eval_masked_with(&buf.obs, &buf.mask, &mut buf.logits, &mut buf.scratch);
    for &index in &buf.set {
        buf.mask[index] = false;
    }
    0.5 * (value + 1.0)
}

impl Agent for NetMctsAgent {
    fn choose(&mut self, state: &GameState, legal: &[Action]) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let result = self.search(state, legal);
        // Most-visited, the standard robust-child rule: a rarely visited edge can hold a
        // high average off two lucky evaluations.
        let mut best = 0usize;
        for i in 1..result.visits.len() {
            if result.visits[i] > result.visits[best] {
                best = i;
            }
        }
        legal[best]
    }

    fn name(&self) -> String {
        format!("netmcts:{}@{}", self.checkpoint.display(), self.sims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent() -> NetMctsAgent {
        NetMctsAgent::new("unused-in-these-tests", 1)
    }

    /// Distinct actions to hang edges off. Nothing here applies them — these tests exercise
    /// [`NetMctsAgent::select`], which only ever compares actions for equality.
    fn action(i: u8) -> Action {
        Action::Attack {
            lane: 0,
            attacker: i,
            target: 0,
        }
    }

    fn edge(action: Action, visits: u32, availability: u32, reward: f64) -> Edge {
        Edge {
            action,
            child: 0,
            visits,
            availability,
            logit: 0.0,
            gamma: 0.0,
            reward: [reward, 0.0],
        }
    }

    /// The one substitution that makes this PUCT rather than AlphaZero's PUCT, and the
    /// easiest thing to drop by accident: the exploration numerator is the edge's own
    /// **availability**, not the parent's visit count.
    ///
    /// Two edges with identical priors, identical visit counts and identical rewards, but
    /// one of which has been on the menu twice as often. Under the availability rule the
    /// one with more availability scores higher — it has had more chances to be tried and
    /// still has the same statistics, so it is the less explored of the two relative to its
    /// opportunities. Under `sqrt(N_parent)` the numerator is shared and the two tie, which
    /// a first-index tie-break would resolve to edge 0.
    #[test]
    fn rule_6_netmcts_puct_uses_availability_not_parent_visits() {
        let node = Node {
            edges: vec![
                edge(action(0), 4, 10, 2.0),
                edge(action(1), 4, 40, 2.0),
            ],
        };
        let available = vec![action(0), action(1)];
        assert_eq!(
            agent().select(&node, &available, Player::P0, false),
            1,
            "the edge that has been available more often must win when everything else ties"
        );
    }

    /// Selection has to read the reward of whoever is to move in *this* determinization.
    /// `DESIGN.md` §4: a turn is three actions plus free sub-decisions, so the mover does
    /// not alternate per node, and it can differ between determinizations at the same node.
    #[test]
    fn rule_6_netmcts_selection_reads_the_current_movers_reward() {
        let mut good_for_p1 = edge(action(1), 4, 10, 0.0);
        good_for_p1.reward = [0.0, 4.0];
        let node = Node {
            edges: vec![edge(action(0), 4, 10, 4.0), good_for_p1],
        };
        let available = vec![action(0), action(1)];
        assert_eq!(agent().select(&node, &available, Player::P0, false), 0);
        assert_eq!(agent().select(&node, &available, Player::P1, false), 1);
    }

    /// An edge that is not legal in this determinization must not be selectable, however
    /// good it looks. This is the subset-armed bandit's other half — availability corrects
    /// the *statistics*, and this corrects the *choice*.
    #[test]
    fn rule_6_netmcts_never_selects_an_edge_that_is_not_available() {
        let node = Node {
            edges: vec![
                edge(action(0), 1, 1, 1.0),
                edge(action(1), 1, 1, 0.0),
            ],
        };
        // Only the second action is legal in this world, even though the first has a
        // perfect record.
        assert_eq!(agent().select(&node, &[action(1)], Player::P0, false), 1);
    }

    /// Priors are stored as logits and softmaxed over whatever subset is available, so the
    /// same edge gets a different prior depending on which siblings are legal. Two logits of
    /// 0 and `ln 3` share the mass 1:3 together, and the second takes all of it alone.
    #[test]
    fn phase3_priors_renormalise_over_the_available_subset() {
        let mut node = Node {
            edges: vec![
                edge(action(0), 1, 1, 0.5),
                edge(action(1), 1, 1, 0.5),
            ],
        };
        node.edges[1].logit = 3.0f32.ln();
        // With both available the higher logit wins; with only the lower one available it is
        // the only thing that can be chosen. The point is that neither call panics on a
        // partial support and both normalise against what they can see.
        assert_eq!(agent().select(&node, &[action(0), action(1)], Player::P0, false), 1);
        assert_eq!(agent().select(&node, &[action(0)], Player::P0, false), 0);
    }

    /// The fixture must build distinct actions, or every test above is vacuous — an edge
    /// lookup is by action equality.
    #[test]
    fn phase3_the_test_fixture_builds_distinct_actions() {
        assert_ne!(action(0), action(1));
    }
}
