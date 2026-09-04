//! SO-ISMCTS — single-observer information-set Monte Carlo tree search.
//!
//! `PLAN.md` Phase 2, rung five; `DESIGN.md` §6 specifies it. Cowling, Powley & Whitehouse
//! (2012), "Information Set Monte Carlo Tree Search". Random rollouts, as `PLAN.md` asks —
//! no evaluation function is involved anywhere in this file, which is what makes the ladder
//! gap against greedy and PIMC attributable to search rather than to hand-written weights.
//!
//! # The one idea
//!
//! A tree node is not a position — it is an **information set**, identified by the sequence
//! of actions taken to reach it. Every iteration samples a fresh determinization and walks
//! that same tree, so statistics gathered under many different hidden worlds accumulate on
//! shared edges. That is the whole difference from PIMC, which solves each world in
//! isolation and averages afterwards, and it is why ISMCTS can value *not knowing*: an
//! action that is good in only half the sampled worlds accumulates a mediocre average on one
//! edge, instead of winning half the independent searches outright.
//!
//! # Subset-armed bandits
//!
//! Because each iteration sees a different world, the actions legal at a node vary between
//! visits — a 5 in one sampled world is a Queen in another, and the sub-decisions they open
//! are not the same. UCB1 divides by the parent's visit count, which is wrong here: it
//! punishes an action for the visits during which it was not even on the menu.
//!
//! SO-ISMCTS fixes this with an **availability count** per edge, incremented on every
//! iteration in which that action was legal, and the exploration term uses that instead:
//!
//! ```text
//! reward[mover] / visits  +  c * sqrt( ln(availability) / visits )
//! ```
//!
//! Dropping this is the single easiest way to end up with an ISMCTS that is quietly just a
//! worse flat Monte Carlo. `rule_6_ismcts_tracks_availability_not_parent_visits` pins it.
//!
//! # Whose reward?
//!
//! Each edge banks the outcome for *both* players, and selection reads the entry belonging
//! to whoever is to move in the current determinization. Storing a single number and
//! negating by depth would be wrong twice over: a turn is three actions plus free
//! sub-decisions, so the mover does not alternate per node (`DESIGN.md` §4), and the mover
//! at a given node can genuinely *differ* between determinizations, because whether an action
//! ends a turn can depend on a hidden rank.
//!
//! # The single-observer limitation, stated plainly
//!
//! The tree conflates the opponent's information sets: it models them as if they saw what
//! the root player sees. That is exactly what "single observer" means, and it is why the
//! opponent in the tree will not bluff and cannot be deceived. MO-ISMCTS fixes it with one
//! tree per player. Not doing that here is deliberate — `DESIGN.md` §6 specifies SO-ISMCTS,
//! and Phase 3 replaces the rollouts with a network before this limitation is the binding
//! constraint.

use crate::action::Action;
use crate::agents::{random_playout, Agent};
use crate::player::Player;
use crate::rng::Rng;
use crate::state::GameState;

/// One action out of one information set.
#[derive(Clone, Debug)]
struct Edge {
    action: Action,
    child: usize,
    /// Iterations that traversed this edge.
    visits: u32,
    /// Iterations in which this action was *legal at this node*, whether or not it was
    /// taken. The denominator of the exploration term — see the module docs.
    availability: u32,
    /// Total backed-up outcome, indexed by player. Both are kept because the player to move
    /// at a node is a property of the sampled world, not of the tree.
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

/// Information-set MCTS with uniform-random rollouts.
#[derive(Clone, Debug)]
pub struct IsmctsAgent {
    rng: Rng,
    iterations: usize,
    exploration: f32,
}

impl IsmctsAgent {
    pub const DEFAULT_ITERATIONS: usize = 800;
    /// The usual UCT constant for rewards already scaled to `0.0..=1.0`.
    pub const DEFAULT_EXPLORATION: f32 = 0.7;

    pub fn new(seed: u64, iterations: usize) -> IsmctsAgent {
        IsmctsAgent {
            rng: Rng::new(seed),
            iterations,
            exploration: IsmctsAgent::DEFAULT_EXPLORATION,
        }
    }

    pub fn derived(seed: u64, stream: u64, iterations: usize) -> IsmctsAgent {
        IsmctsAgent {
            rng: Rng::derive(seed, stream),
            iterations,
            exploration: IsmctsAgent::DEFAULT_EXPLORATION,
        }
    }

    pub fn with_exploration(mut self, c: f32) -> IsmctsAgent {
        self.exploration = c;
        self
    }

    /// Run the search and return the root's visit counts, aligned with `legal`.
    ///
    /// Exposed because visit counts *are* the policy: Phase 3 trains on them
    /// (`DESIGN.md` §6), and Phase 5's policy characterisation reads them directly. `choose`
    /// is a thin argmax over this.
    pub fn root_visits(&mut self, state: &GameState, legal: &[Action]) -> Vec<u32> {
        let me = state.acting_player();
        let mut tree: Vec<Node> = vec![Node::default()];

        for _ in 0..self.iterations {
            let mut world = state.determinize(me, &mut self.rng);
            let mut node = 0usize;
            // (node, edge) pairs traversed, for the backup.
            let mut path: Vec<(usize, usize)> = Vec::with_capacity(16);

            loop {
                if world.outcome.is_over() {
                    break;
                }
                let available = world.legal_actions();

                // Availability is credited to every legal action, taken or not. Do it
                // before selection so the edge we are about to create is counted too.
                for &action in &available {
                    if let Some(i) = tree[node].edge_for(action) {
                        tree[node].edges[i].availability += 1;
                    }
                }

                let untried: Vec<Action> = available
                    .iter()
                    .copied()
                    .filter(|&a| tree[node].edge_for(a).is_none())
                    .collect();

                if !untried.is_empty() {
                    // Expand exactly one new edge, then hand over to the rollout.
                    let action = *self
                        .rng
                        .choose(&untried)
                        .expect("untried is non-empty in this branch");
                    let child = tree.len();
                    tree.push(Node::default());
                    tree[node].edges.push(Edge {
                        action,
                        child,
                        visits: 0,
                        availability: 1,
                        reward: [0.0, 0.0],
                    });
                    let edge = tree[node].edges.len() - 1;
                    path.push((node, edge));
                    world.apply_trusted(action);
                    break;
                }

                // Every legal action already has an edge: select among them.
                let mover = world.acting_player();
                let edge = self.select(&tree[node], &available, mover);
                path.push((node, edge));
                let action = tree[node].edges[edge].action;
                node = tree[node].edges[edge].child;
                world.apply_trusted(action);
            }

            random_playout(&mut world, &mut self.rng);
            let reward = [
                world.outcome.value_for(Player::P0) as f64,
                world.outcome.value_for(Player::P1) as f64,
            ];
            for (n, e) in path {
                let edge = &mut tree[n].edges[e];
                edge.visits += 1;
                edge.reward[0] += reward[0];
                edge.reward[1] += reward[1];
            }
        }

        legal
            .iter()
            .map(|&action| {
                tree[0]
                    .edge_for(action)
                    .map(|i| tree[0].edges[i].visits)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// UCB over the edges legal in this determinization.
    fn select(&self, node: &Node, available: &[Action], mover: Player) -> usize {
        let mut best = usize::MAX;
        let mut best_score = f32::NEG_INFINITY;
        for (i, edge) in node.edges.iter().enumerate() {
            if !available.contains(&edge.action) {
                continue;
            }
            let score = if edge.visits == 0 {
                // Defensive. An edge is created only by an expansion, and every expansion
                // is followed immediately by a rollout and a backup, so a visitless edge
                // should not survive to be selected. Treat it as maximally urgent, the way
                // UCT treats an unvisited child, rather than dividing by zero.
                f32::INFINITY
            } else {
                let exploit = (edge.reward[mover.idx()] / edge.visits as f64) as f32;
                // `availability >= visits >= 1` here, so the logarithm is non-negative.
                let explore = (edge.availability.max(1) as f32).ln() / edge.visits as f32;
                exploit + self.exploration * explore.max(0.0).sqrt()
            };
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

impl Agent for IsmctsAgent {
    fn choose(&mut self, state: &GameState, legal: &[Action]) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let visits = self.root_visits(state, legal);

        // Most-visited rather than highest-mean: the standard robust-child rule. A rarely
        // visited edge can hold a high average on two lucky rollouts, and with random
        // rollouts over sampled worlds that happens constantly.
        let mut best = 0usize;
        for i in 1..visits.len() {
            if visits[i] > visits[best] {
                best = i;
            }
        }
        legal[best]
    }

    fn name(&self) -> String {
        format!("ismcts:{}", self.iterations)
    }
}
