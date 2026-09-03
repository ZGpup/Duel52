//! PyO3 bindings for the Duel 52 engine.
//!
//! `CLAUDE.md`: "The Rust engine is the sole authority on legality. Never reimplement rules
//! logic in Python — call the engine. Python does training and analysis only." These
//! bindings are the seam that makes that possible, and they deliberately expose **no** way
//! to construct or mutate a position by hand: a `Game` can only be dealt from a config and
//! a seed, and only advanced by applying an action the engine itself declared legal.
//!
//! Imported from Python as `duel52._engine`, and re-exported by the `duel52` package.
//!
//! ```python
//! from duel52 import Game
//!
//! g = Game(variant="split", seed=42)
//! while not g.is_over:
//!     actions = g.legal_actions()
//!     g.apply_index(0)          # or g.apply(actions[i])
//! print(g.outcome, g.value_for("p0"))
//! ```

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use duel52_engine::action::Side;
use duel52_engine::config::{GameConfig, TwoPower, Variant};
use duel52_engine::display;
use duel52_engine::encode;
use duel52_engine::outcome::{DrawReason, Outcome};
use duel52_engine::player::Player;
use duel52_engine::rank::Rank;
use duel52_engine::state::GameState;
// `Agent` is not imported: `AgentSpec::build` hands back a `Box<dyn Agent>`, and calling a
// trait object's own method does not need the trait in scope.
use duel52_engine::{Action, AgentSpec, Rng};

// ============================================================ small conversions ==

fn parse_player(s: &str) -> PyResult<Player> {
    match s.trim().to_ascii_lowercase().as_str() {
        "p0" | "0" | "first" => Ok(Player::P0),
        "p1" | "1" | "second" => Ok(Player::P1),
        other => Err(PyValueError::new_err(format!(
            "expected 'p0' or 'p1', got {other:?}"
        ))),
    }
}

fn player_name(p: Player) -> &'static str {
    match p {
        Player::P0 => "p0",
        Player::P1 => "p1",
    }
}

fn outcome_name(o: Outcome) -> String {
    match o {
        Outcome::Ongoing => "ongoing".to_string(),
        Outcome::Win(p) => format!("{}_wins", player_name(p)),
        Outcome::Draw(DrawReason::Stalemate) => "draw_stalemate".to_string(),
        Outcome::Draw(DrawReason::MutualLaneWin) => "draw_mutual_lane_win".to_string(),
        Outcome::Draw(DrawReason::PlyLimit) => "draw_ply_limit".to_string(),
    }
}

/// Turn an [`Action`] into a plain Python dict.
///
/// The `kind` key names the action; the rest of the keys are its arguments. Slot indices
/// are positions in the lane's card list *as of right now* and are invalidated by anything
/// that removes a card, so always re-read `legal_actions()` after applying.
fn action_to_dict<'py>(py: Python<'py>, action: Action) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match action {
        Action::Play { rank, lane } => {
            d.set_item("kind", "play")?;
            d.set_item("rank", rank.index())?;
            d.set_item("lane", lane)?;
        }
        Action::Flip { lane, slot } => {
            d.set_item("kind", "flip")?;
            d.set_item("lane", lane)?;
            d.set_item("slot", slot)?;
        }
        Action::Attack {
            lane,
            attacker,
            target,
        } => {
            d.set_item("kind", "attack")?;
            d.set_item("lane", lane)?;
            d.set_item("attacker", attacker)?;
            d.set_item("target", target)?;
        }
        Action::DeclarePair {
            lane,
            slot_a,
            slot_b,
        } => {
            d.set_item("kind", "pair")?;
            d.set_item("lane", lane)?;
            d.set_item("slot_a", slot_a)?;
            d.set_item("slot_b", slot_b)?;
        }
        Action::Pass => {
            d.set_item("kind", "pass")?;
        }
        Action::Peek { side, lane, slot } => {
            d.set_item("kind", "peek")?;
            d.set_item(
                "side",
                match side {
                    Side::Mine => "mine",
                    Side::Theirs => "theirs",
                },
            )?;
            d.set_item("lane", lane)?;
            d.set_item("slot", slot)?;
        }
        Action::ResolveNext { lane, slot } => {
            d.set_item("kind", "resolve_next")?;
            d.set_item("lane", lane)?;
            d.set_item("slot", slot)?;
        }
        Action::MoveHere { lane, slot } => {
            d.set_item("kind", "move_here")?;
            d.set_item("lane", lane)?;
            d.set_item("slot", slot)?;
        }
        Action::GiveBack { rank } => {
            d.set_item("kind", "give_back")?;
            d.set_item("rank", rank.index())?;
        }
        Action::SplitTarget { slot } => {
            d.set_item("kind", "split_target")?;
            d.set_item("slot", slot)?;
        }
    }
    Ok(d)
}

/// Read a required key out of an action dict, with a message that names the missing key
/// rather than raising a bare `KeyError`.
fn get<'py, T>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    // `for<'a>`: the extracted value must not borrow from the temporary `Bound` below, so
    // it has to be extractable from a borrow of any lifetime. Every type used here is
    // owned (integers, String), which satisfies that.
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    let item = d
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("action dict is missing key {key:?}")))?;
    item.extract()
}

fn rank_from(index: usize) -> PyResult<Rank> {
    Rank::try_from_index(index)
        .ok_or_else(|| PyValueError::new_err(format!("rank index {index} is out of range 0..13")))
}

/// Inverse of [`action_to_dict`].
fn action_from_dict(d: &Bound<'_, PyDict>) -> PyResult<Action> {
    let kind: String = get(d, "kind")?;
    Ok(match kind.as_str() {
        "play" => Action::Play {
            rank: rank_from(get(d, "rank")?)?,
            lane: get(d, "lane")?,
        },
        "flip" => Action::Flip {
            lane: get(d, "lane")?,
            slot: get(d, "slot")?,
        },
        "attack" => Action::Attack {
            lane: get(d, "lane")?,
            attacker: get(d, "attacker")?,
            target: get(d, "target")?,
        },
        "pair" => Action::DeclarePair {
            lane: get(d, "lane")?,
            slot_a: get(d, "slot_a")?,
            slot_b: get(d, "slot_b")?,
        },
        "pass" => Action::Pass,
        "peek" => {
            let side: String = get(d, "side")?;
            Action::Peek {
                side: match side.as_str() {
                    "mine" => Side::Mine,
                    "theirs" => Side::Theirs,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "peek side must be 'mine' or 'theirs', got {other:?}"
                        )))
                    }
                },
                lane: get(d, "lane")?,
                slot: get(d, "slot")?,
            }
        }
        "resolve_next" => Action::ResolveNext {
            lane: get(d, "lane")?,
            slot: get(d, "slot")?,
        },
        "move_here" => Action::MoveHere {
            lane: get(d, "lane")?,
            slot: get(d, "slot")?,
        },
        "give_back" => Action::GiveBack {
            rank: rank_from(get(d, "rank")?)?,
        },
        "split_target" => Action::SplitTarget {
            slot: get(d, "slot")?,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown action kind {other:?}"
            )))
        }
    })
}

// ==================================================================== the Game ==

/// A Duel 52 position.
///
/// Deal one with `Game(variant=..., seed=...)`, read `legal_actions()`, and advance it with
/// `apply()` or `apply_index()`. `clone()` is cheap enough for search — a position is a few
/// hundred bytes (`DESIGN.md` §1).
// `skip_from_py_object`: a Game is only ever *returned* to Python, never accepted as an
// argument, so it needs no FromPyObject impl. Opting out silences PyO3's deprecation
// warning about the automatic impl for Clone types.
#[pyclass(name = "Game", module = "duel52._engine", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGame {
    state: GameState,
}

#[pymethods]
impl PyGame {
    /// Deal a new game.
    ///
    /// `variant` is `"base"`, `"split"` (the project default), or `"mirrored"`.
    /// `two_power` is `"bottom"` (the house rule, default) or `"discard"`
    /// (rules-as-written). The same `seed` and settings always produce the same game.
    #[new]
    #[pyo3(signature = (variant="split", seed=0, two_power=None, stalemate_quiet_plies=None))]
    fn new(
        variant: &str,
        seed: u64,
        two_power: Option<&str>,
        stalemate_quiet_plies: Option<u32>,
    ) -> PyResult<PyGame> {
        let v = Variant::parse(variant).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown variant {variant:?} (expected base, split or mirrored)"
            ))
        })?;
        let mut config = GameConfig::preset(v);
        if let Some(tp) = two_power {
            config.two_power = TwoPower::parse(tp).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown two_power {tp:?} (expected bottom or discard)"
                ))
            })?;
        }
        if let Some(n) = stalemate_quiet_plies {
            config.stalemate_quiet_plies = n;
        }
        config
            .validate()
            .map_err(|e| PyValueError::new_err(format!("invalid config: {e}")))?;
        Ok(PyGame {
            state: GameState::new(config, seed),
        })
    }

    /// An independent copy. Search should clone rather than undo — there is no undo.
    fn clone_state(&self) -> PyGame {
        self.clone()
    }

    fn __copy__(&self) -> PyGame {
        self.clone()
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, _memo: Option<Bound<'_, PyDict>>) -> PyGame {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Game {} ply={} to_move={} phase={} outcome={}>",
            self.state.config.summary(),
            self.state.ply,
            player_name(self.state.to_move),
            self.state.phase(),
            outcome_name(self.state.outcome),
        )
    }

    // ---------------------------------------------------------------- properties --

    #[getter]
    fn seed(&self) -> u64 {
        self.state.seed
    }
    #[getter]
    fn variant(&self) -> String {
        self.state.config.variant.label().to_string()
    }
    #[getter]
    fn two_power(&self) -> String {
        self.state.config.two_power.label().to_string()
    }
    #[getter]
    fn ply(&self) -> u32 {
        self.state.ply
    }
    #[getter]
    fn to_move(&self) -> String {
        player_name(self.state.to_move).to_string()
    }
    #[getter]
    fn actions_remaining(&self) -> u32 {
        self.state.actions_remaining
    }
    /// `"main"` for a normal action, otherwise the sub-decision a power has opened.
    #[getter]
    fn phase(&self) -> String {
        format!("{}", self.state.phase())
    }
    /// True once every draw pile is empty — the gate on base cards, on the 5/7/Q reaching
    /// base cards, and on lane wins (`game_rules.md` §3, §7).
    #[getter]
    fn base_unlocked(&self) -> bool {
        self.state.base_unlocked
    }
    #[getter]
    fn quiet_plies(&self) -> u32 {
        self.state.quiet_plies
    }
    #[getter]
    fn is_over(&self) -> bool {
        self.state.outcome.is_over()
    }
    /// `"ongoing"`, `"p0_wins"`, `"p1_wins"`, `"draw_stalemate"`,
    /// `"draw_mutual_lane_win"`, or `"draw_ply_limit"`.
    #[getter]
    fn outcome(&self) -> String {
        outcome_name(self.state.outcome)
    }

    /// Zero-sum value from a player's point of view: 1.0 win, 0.5 draw, 0.0 loss.
    fn value_for(&self, player: &str) -> PyResult<f32> {
        Ok(self.state.outcome.value_for(parse_player(player)?))
    }

    fn lanes_won(&self, player: &str) -> PyResult<usize> {
        Ok(self.state.lanes_won_by(parse_player(player)?))
    }

    fn hand_size(&self, player: &str) -> PyResult<usize> {
        Ok(self.state.hand(parse_player(player)?).len())
    }

    /// The player's own hand, as 13 per-rank counts. This is private information — do not
    /// hand it to an agent playing the other side.
    fn hand_counts(&self, player: &str) -> PyResult<Vec<u8>> {
        Ok(self.state.hand_counts(parse_player(player)?).to_vec())
    }

    fn pile_size(&self, player: &str) -> PyResult<usize> {
        Ok(self.state.pile(parse_player(player)?).len())
    }

    /// The discard pile — public to both players (`game_rules.md` §5).
    fn discard_counts(&self, player: &str) -> PyResult<Vec<u8>> {
        let p = parse_player(player)?;
        Ok(duel52_engine::rank_counts(&self.state.discards[p.idx()]).to_vec())
    }

    // ------------------------------------------------------------------- actions --

    /// Every legal action right now, as a list of dicts. Empty exactly when the game is
    /// over.
    fn legal_actions<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.state
            .legal_actions()
            .into_iter()
            .map(|a| action_to_dict(py, a))
            .collect()
    }

    /// The same list, rendered in plain language from `observer`'s point of view. Useful
    /// for debugging and for logging a game a human has to read.
    #[pyo3(signature = (observer=None))]
    fn legal_action_descriptions(&self, observer: Option<&str>) -> PyResult<Vec<String>> {
        let obs = match observer {
            Some(s) => Some(parse_player(s)?),
            None => None,
        };
        Ok(self
            .state
            .legal_actions()
            .into_iter()
            .map(|a| display::describe_action(&self.state, a, obs))
            .collect())
    }

    fn legal_action_count(&self) -> usize {
        self.state.legal_actions().len()
    }

    /// Apply the `index`-th action from `legal_actions()`. The fast path for search.
    fn apply_index(&mut self, index: usize) -> PyResult<()> {
        let legal = self.state.legal_actions();
        let action = *legal
            .get(index)
            .ok_or_else(|| PyIndexError::new_err(format!(
                "action index {index} out of range (0..{})",
                legal.len()
            )))?;
        self.state.apply_trusted(action);
        Ok(())
    }

    /// Apply an action dict. Rejected with a `RuntimeError` if it is not legal.
    fn apply(&mut self, action: &Bound<'_, PyDict>) -> PyResult<()> {
        let a = action_from_dict(action)?;
        self.state
            .apply(a)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    // -------------------------------------------------------------- observations --

    /// The board, hands and scalars as `observer` is entitled to see them
    /// (`game_rules.md` §5). This is the raw material for `DESIGN.md` §5's encoder; it
    /// deliberately stops short of producing a float tensor, which is a Phase 3 decision.
    ///
    /// Cards whose rank the observer may not know report `rank: None` and `rank_known:
    /// False`. Base cards are unknown to **both** players, their owner included.
    fn observation<'py>(&self, py: Python<'py>, observer: &str) -> PyResult<Bound<'py, PyDict>> {
        let me = parse_player(observer)?;
        let them = me.other();
        let s = &self.state;
        let out = PyDict::new(py);

        out.set_item("observer", player_name(me))?;
        out.set_item("to_move", player_name(s.to_move))?;
        out.set_item("is_mine_to_move", s.to_move == me)?;
        out.set_item("ply", s.ply)?;
        out.set_item("phase", format!("{}", s.phase()))?;
        out.set_item("actions_remaining", s.actions_remaining)?;
        out.set_item("base_unlocked", s.base_unlocked)?;
        out.set_item("quiet_plies", s.quiet_plies)?;
        out.set_item("stalemate_quiet_plies", s.config.stalemate_quiet_plies)?;
        out.set_item("is_over", s.outcome.is_over())?;
        out.set_item("outcome", outcome_name(s.outcome))?;
        // Which *seat* the observer holds. P0 moves first and takes only two actions on the
        // opening turn (`game_rules.md` §2), so the seat is a real asymmetry and the encoder
        // wants it. This used to be `first_player = (to_move == P0)`, which was a mislabelled
        // duplicate of `to_move` and told an observer nothing about itself; nothing consumed
        // it, so it was corrected rather than kept alongside.
        out.set_item("observer_is_first_player", me == Player::P0)?;

        out.set_item("hand", s.hand_counts(me).to_vec())?;
        out.set_item("hand_size", s.hand(me).len())?;
        out.set_item("opponent_hand_size", s.hand(them).len())?;
        out.set_item("my_pile_size", s.pile(me).len())?;
        out.set_item("opponent_pile_size", s.pile(them).len())?;
        out.set_item("shared_pile", s.shared_pile())?;
        out.set_item(
            "my_discard",
            duel52_engine::rank_counts(&s.discards[me.idx()]).to_vec(),
        )?;
        out.set_item(
            "opponent_discard",
            duel52_engine::rank_counts(&s.discards[them.idx()]).to_vec(),
        )?;

        // The removed pool. `game_rules.md` §2: normally hidden from everyone and never
        // fully resolvable. §9b is the exception — there it is public.
        out.set_item("removed_size", s.all_removed().count())?;
        out.set_item("removed_revealed", s.removed_revealed)?;
        if s.removed_revealed {
            out.set_item(
                "removed_counts",
                duel52_engine::rank_counts(&s.removed[me.idx()]).to_vec(),
            )?;
        } else {
            out.set_item("removed_counts", py.None())?;
        }

        // Belief: how many of each rank this observer cannot place. `DESIGN.md` §5, and the
        // same bag `determinize` deals from, so the encoder and the sampler cannot disagree
        // about what is unknown. §2: outside the mirrored variant these never reach zero.
        out.set_item("unseen_counts", s.unseen_counts(me).to_vec())?;

        // Private knowledge from the 2's scry (§10a): what this observer put on the bottom
        // of a pile, bottom-most first.
        let bottom = PyDict::new(py);
        for owner in Player::BOTH {
            let idx = s.pile_index(owner);
            let known: Vec<Option<usize>> = s.piles[idx]
                .known_from_bottom(me)
                .into_iter()
                .map(|k| k.map(|r| r.index()))
                .collect();
            bottom.set_item(player_name(owner), known)?;
        }
        out.set_item("pile_bottom_known", bottom)?;

        // The board.
        let board = PyList::empty(py);
        for lane in 0..s.lane_count() {
            for owner in [me, them] {
                for (slot, card) in s.lanes[lane].side(owner).iter().enumerate() {
                    let c = PyDict::new(py);
                    c.set_item("lane", lane)?;
                    c.set_item("slot", slot)?;
                    c.set_item("is_mine", owner == me)?;
                    let known = card.rank_known_to(me);
                    c.set_item("rank_known", known)?;
                    c.set_item("rank", if known { Some(card.rank.index()) } else { None })?;
                    c.set_item("face_up", card.face_up)?;
                    c.set_item("is_base", card.is_base)?;
                    c.set_item("entered_as_base", card.entered_as_base)?;
                    // Both are public (§5). Damage is visible on face-down cards because
                    // the card is turned sideways, and max HP leaks nothing because every
                    // face-down card is a blank 2-HP card whatever its rank — so a Jack
                    // cannot be identified by watching it survive.
                    c.set_item("damage", card.damage)?;
                    c.set_item("max_hp", card.max_hp())?;
                    c.set_item("frozen", card.is_frozen(s.ply))?;
                    c.set_item("attacks_used", card.attacks_used)?;
                    c.set_item("attack_allowance", card.attack_allowance)?;
                    c.set_item("paired", card.pair_id.is_some())?;
                    c.set_item("pair_id", card.pair_id.map(|p| p.0))?;
                    board.append(c)?;
                }
            }
        }
        out.set_item("board", board)?;
        Ok(out)
    }

    // ------------------------------------------------------- Phase 3 encoding --

    /// The observation tensor for `observer`, as a flat list of `obs_dim` floats.
    ///
    /// **This is the only encoder.** `CLAUDE.md` puts rules logic in Rust for one reason and
    /// the same reason applies here with more force: a second copy of the feature layout on
    /// the Python side could drift from this one silently, and the trained function would
    /// stop matching the evaluated function with nothing crashing to say so. The training
    /// code calls this; it never builds a tensor itself.
    ///
    /// The layout is documented in `engine/src/encode.rs` and pinned by
    /// `obs_layout_hash` in [`encoding_spec`].
    fn encode_observation(&self, observer: &str) -> PyResult<Vec<f32>> {
        let me = parse_player(observer)?;
        let mut out = vec![0.0f32; encode::obs_dim(&self.state.config)];
        encode::encode_observation(&self.state, me, &mut out);
        Ok(out)
    }

    /// The legality mask for the current decision: `action_dim` booleans, true exactly at
    /// the indices of the legal actions.
    ///
    /// Built from the engine's own `legal_actions()`, never from a second implementation of
    /// legality.
    fn legal_mask(&self) -> Vec<bool> {
        let mut out = vec![false; encode::action_dim(&self.state.config)];
        encode::legal_mask(&self.state, &mut out);
        out
    }

    /// The policy-head index for an action dict.
    ///
    /// Takes the position as well as the action because a `split_target` carries no lane —
    /// the lane comes from the twinstrike already in flight.
    fn encode_action(&self, action: &Bound<'_, PyDict>) -> PyResult<usize> {
        let a = action_from_dict(action)?;
        Ok(encode::encode_action(&a, &self.state))
    }

    /// The action a policy index names in this position, or `None` if the index means
    /// nothing here.
    ///
    /// `None` is not an error: most of the 1325 indices are meaningless in any given
    /// position, which is what the mask is for. The `CHOOSE_SLOT` block is shared between
    /// four sub-decisions whose phases are mutually exclusive, so the answer depends on the
    /// phase.
    fn decode_action<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match encode::decode_action(index, &self.state) {
            Some(a) => Ok(Some(action_to_dict(py, a)?)),
            None => Ok(None),
        }
    }

    /// The board as text, from `observer`'s point of view. Pass `None` to reveal
    /// everything — debugging only.
    #[pyo3(signature = (observer=None))]
    fn render(&self, observer: Option<&str>) -> PyResult<String> {
        let obs = match observer {
            Some(s) => Some(parse_player(s)?),
            None => None,
        };
        Ok(display::render(&self.state, obs))
    }

    // ------------------------------------------------------------ search support --

    /// Sample a world consistent with what `observer` knows (`DESIGN.md` §6, step 1).
    ///
    /// Returns a new `Game` in which every rank `observer` is entitled to know is unchanged
    /// and every rank they are not — the opponent's hand, face-down cards including
    /// `observer`'s **own base cards**, unknown pile positions, and the removed-unseen pool —
    /// has been redealt at random subject to the deck composition.
    ///
    /// This is the honest way for a Python-side search to reach hidden state, and the only
    /// one these bindings offer: there is deliberately no accessor for the opponent's hand or
    /// the pile order. The legal action list is identical in the sampled world, so actions
    /// enumerated on the real game can be applied to it directly.
    ///
    /// ```python
    /// world = g.determinize("p0", seed=1)
    /// assert world.legal_actions() == g.legal_actions()
    /// ```
    #[pyo3(signature = (observer, seed=0))]
    fn determinize(&self, observer: &str, seed: u64) -> PyResult<PyGame> {
        let p = parse_player(observer)?;
        let mut rng = Rng::new(seed);
        Ok(PyGame {
            state: self.state.determinize(p, &mut rng),
        })
    }

    /// Ask a **freshly built** ladder agent what it would do here, as an index into
    /// `legal_actions()`.
    ///
    /// A one-shot probe: "what does greedy think of this position?". Because the agent is
    /// rebuilt from `seed` every call, the answer is a pure function of the position and the
    /// seed — which is what you want for a probe and *not* what you want for playing a game.
    /// To play, use [`Agent`], which carries its random stream forward across decisions.
    #[pyo3(signature = (agent, seed=0))]
    fn agent_action_index(&self, agent: &str, seed: u64) -> PyResult<usize> {
        let spec = AgentSpec::parse(agent).map_err(PyValueError::new_err)?;
        let inner = spec.build(seed, 0);
        PyAgent { spec, inner }.choose_index(self)
    }
}

// ========================================================================= agents ==

/// One of the frozen Phase 2 ladder agents, with its random stream.
///
/// `name` is a ladder name — `"random"`, `"greedy"`, `"flatmc:600"`, `"pimc:32x1"`,
/// `"ismcts:800"` — see `duel52.ladder_agents()` for the frozen roster and `duel52 help`
/// for the budget syntax.
///
/// **Keep one agent for a whole game.** An agent carries its random stream, and rebuilding
/// it before every decision restarts that stream: tie-breaks stop being independent and a
/// search agent samples the same first world at every node. It measurably weakens play.
///
/// ```python
/// from duel52 import Game
/// from duel52._engine import Agent
///
/// g, bot = Game(seed=1), Agent("ismcts:800", seed=7)
/// while not g.is_over:
///     g.apply_index(bot.choose_index(g))
/// ```
#[pyclass(name = "Agent")]
pub struct PyAgent {
    spec: AgentSpec,
    inner: Box<dyn duel52_engine::Agent + Send + Sync>,
}

#[pymethods]
impl PyAgent {
    #[new]
    #[pyo3(signature = (name, seed=0))]
    fn new(name: &str, seed: u64) -> PyResult<PyAgent> {
        let spec = AgentSpec::parse(name).map_err(PyValueError::new_err)?;
        let inner = spec.build(seed, 0);
        Ok(PyAgent { spec, inner })
    }

    #[getter]
    fn name(&self) -> String {
        self.spec.name()
    }

    fn __repr__(&self) -> String {
        format!("<Agent {}>", self.spec.name())
    }

    /// Choose a move in `game`, as an index into its `legal_actions()`.
    ///
    /// The agent sees only its own information set: the search agents reach hidden state
    /// through `Game.determinize`, never by reading the position's private fields.
    fn choose_index(&mut self, game: &PyGame) -> PyResult<usize> {
        let legal = game.state.legal_actions();
        if legal.is_empty() {
            return Err(PyRuntimeError::new_err(
                "the game is over, so there is no move to make",
            ));
        }
        let action = self.inner.choose(&game.state, &legal);
        legal
            .iter()
            .position(|&a| a == action)
            .ok_or_else(|| PyRuntimeError::new_err("agent returned an action it was not offered"))
    }
}

// =================================================================== module init ==

/// Run `games` random-vs-random games and return the summary statistics as a dict.
///
/// The Phase 1 deliverable, callable from Python so the numbers can go straight into a
/// notebook without shelling out to the CLI.
#[pyfunction]
#[pyo3(signature = (variant="split", first_seed=0, games=1000, two_power=None))]
fn random_play_stats<'py>(
    py: Python<'py>,
    variant: &str,
    first_seed: u64,
    games: usize,
    two_power: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let v = Variant::parse(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown variant {variant:?}")))?;
    let mut config = GameConfig::preset(v);
    if let Some(tp) = two_power {
        config.two_power = TwoPower::parse(tp)
            .ok_or_else(|| PyValueError::new_err(format!("unknown two_power {tp:?}")))?;
    }
    let stats = duel52_engine::run_random_games(config, first_seed, games);

    let d = PyDict::new(py);
    d.set_item("variant", config.variant.label())?;
    d.set_item("two_power", config.two_power.label())?;
    d.set_item("first_seed", first_seed)?;
    d.set_item("games", stats.games)?;
    d.set_item("p0_wins", stats.p0_wins)?;
    d.set_item("p1_wins", stats.p1_wins)?;
    d.set_item("draws", stats.draws)?;
    d.set_item("draws_stalemate", stats.draws_stalemate)?;
    d.set_item("draws_mutual_lane_win", stats.draws_mutual_lane_win)?;
    d.set_item("draws_ply_limit", stats.draws_ply_limit)?;
    d.set_item("p0_score", stats.p0_score())?;
    d.set_item("p0_score_ci95", stats.p0_score_ci95())?;
    d.set_item("lengths", stats.lengths.clone())?;
    d.set_item("unlock_plies", stats.unlock_plies.clone())?;
    d.set_item("hand_at_unlock", stats.hand_at_unlock.clone())?;
    d.set_item("max_side_occupancy", stats.max_side_occupancy)?;
    d.set_item("elapsed_secs", stats.elapsed_secs)?;
    d.set_item("report", stats.report())?;
    Ok(d)
}

/// The card-power reference as text.
#[pyfunction]
fn power_reference() -> String {
    display::power_reference()
}

/// The frozen Phase 2 benchmark ladder, weakest first.
///
/// `PLAN.md` Phase 2 froze these exact configurations as the permanent benchmark, so Phase 3
/// should read the list from here rather than hard-coding names that could drift.
#[pyfunction]
fn ladder_agents() -> Vec<String> {
    AgentSpec::LADDER.iter().map(|s| s.name()).collect()
}

/// The tensor shapes and layout hashes the training side must build against.
///
/// **This is what makes "Python never writes its own encoder" enforceable.** The two hashes
/// are computed once, in `engine/src/encode.rs`, from the feature table and the action
/// blocks; Python stamps whatever it reads here into the checkpoint header, and
/// `Weights::load` refuses a checkpoint whose hashes do not match the build that is about to
/// evaluate it. Nothing on the Python side ever computes a hash, so the two cannot disagree
/// about what the layout is — only about whether they are the same.
///
/// The layout is identical across the three variants (same lanes, same rank count, same
/// `encoding_slots`), so one checkpoint plays all three; `variant` is accepted anyway
/// because Duel52-mini (`DESIGN.md` §7) will not share it.
#[pyfunction]
#[pyo3(signature = (variant="split", encoding_slots=None))]
fn encoding_spec<'py>(
    py: Python<'py>,
    variant: &str,
    encoding_slots: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let v = Variant::parse(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown variant {variant:?}")))?;
    let mut config = GameConfig::preset(v);
    if let Some(n) = encoding_slots {
        config.encoding_slots = n;
    }
    config
        .validate()
        .map_err(|e| PyValueError::new_err(format!("invalid config: {e}")))?;

    let d = PyDict::new(py);
    d.set_item("variant", config.variant.label())?;
    d.set_item("obs_dim", encode::obs_dim(&config))?;
    d.set_item("action_dim", encode::action_dim(&config))?;
    d.set_item("encoding_slots", config.encoding_slots)?;
    d.set_item("lanes", config.lanes)?;
    d.set_item("ranks", config.rank_count())?;
    d.set_item("slot_features", encode::slot_features(&config))?;
    // Rendered as 16 hex characters, the same way they appear in a checkpoint header, so a
    // mismatch can be read straight off the two strings.
    d.set_item("obs_layout_hash", format!("{:016x}", encode::obs_layout_hash(&config)))?;
    d.set_item(
        "action_layout_hash",
        format!("{:016x}", encode::action_layout_hash(&config)),
    )?;
    // The full descriptions, for a diff when the hashes disagree and you need to know
    // *which* feature moved.
    d.set_item("obs_layout", encode::obs_layout_string(&config))?;
    d.set_item("action_layout", encode::action_layout_string(&config))?;

    let blocks = PyList::empty(py);
    for b in encode::action_blocks(&config) {
        let e = PyDict::new(py);
        e.set_item("name", b.name)?;
        e.set_item("offset", b.offset)?;
        e.set_item("len", b.len)?;
        blocks.append(e)?;
    }
    d.set_item("action_blocks", blocks)?;
    Ok(d)
}

// ================================================= Phase 3 step 3: the training corpus ==

/// Read a `.d52sp` self-play shard and replay it into training tensors.
///
/// This is the *only* way the trainer gets observations, and it is deliberately a call into
/// Rust rather than a decoder in Python: `CLAUDE.md` says there is exactly one encoder and
/// it lives in `engine/src/encode.rs`. The shard stores trajectories — seeds and
/// legal-action indices — so the tensors come out in whatever layout this build encodes,
/// and a shard survives an encoder change.
///
/// Returns a dict of **flat little-endian buffers** to be wrapped with
/// ``numpy.frombuffer``; see ``py/duel52/train/buffer.py``, which is the only intended
/// caller. Observations are sparse (4.8% dense — `FINDINGS.md` F3.3), as CSR-style
/// `offset`/`index`/`value` triples, because a dense generation is gigabytes and a sparse
/// one is hundreds of megabytes.
///
/// `stride` keeps one decision in `stride` (default 1 — all of them).
#[pyfunction]
#[pyo3(signature = (path, threads=0, stride=1))]
fn replay_shard<'py>(
    py: Python<'py>,
    path: &str,
    threads: usize,
    stride: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use pyo3::types::PyBytes;

    let shard = duel52_engine::selfplay::Shard::read(std::path::Path::new(path))
        .map_err(PyRuntimeError::new_err)?;
    let threads = if threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        threads
    };
    // Replaying is pure Rust over data it owns, so the GIL is not needed and holding it
    // would serialise the trainer against anything else in the process. (`detach` is
    // pyo3 0.29's name for what used to be `allow_threads`.)
    let (set, recorded) = py.detach(|| {
        let recorded = shard.sample_count();
        (duel52_engine::selfplay::replay(&shard, threads, stride), recorded)
    });

    fn bytes_of<'py, T: Copy>(py: Python<'py>, v: &[T]) -> Bound<'py, PyBytes> {
        // Every element is a plain `u32` or `f32`, so the buffer is exactly the
        // little-endian array numpy will read back. Little-endian is assumed; every
        // platform this project targets is.
        let raw = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        PyBytes::new(py, raw)
    }

    let d = PyDict::new(py);
    d.set_item("samples", set.samples)?;
    d.set_item("recorded_samples", recorded)?;
    d.set_item("obs_dim", set.obs_dim)?;
    d.set_item("action_dim", set.action_dim)?;
    d.set_item("obs_offset", bytes_of(py, &set.obs_offset))?;
    d.set_item("obs_index", bytes_of(py, &set.obs_index))?;
    d.set_item("obs_value", bytes_of(py, &set.obs_value))?;
    d.set_item("policy_offset", bytes_of(py, &set.policy_offset))?;
    d.set_item("policy_index", bytes_of(py, &set.policy_index))?;
    d.set_item("policy_prob", bytes_of(py, &set.policy_prob))?;
    d.set_item("value", bytes_of(py, &set.value))?;
    d.set_item("root_value", bytes_of(py, &set.root_value))?;

    let header = PyDict::new(py);
    for (k, v) in &shard.header {
        header.set_item(k, v)?;
    }
    d.set_item("header", header)?;
    d.set_item("games", shard.games.len())?;
    d.set_item("config", shard.config.to_config_string())?;
    d.set_item(
        "obs_layout_hash",
        format!("{:016x}", encode::obs_layout_hash(&shard.config)),
    )?;
    d.set_item(
        "action_layout_hash",
        format!("{:016x}", encode::action_layout_hash(&shard.config)),
    )?;
    Ok(d)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust engine for Duel 52. Imported via the `duel52` package.")?;
    m.add("VERSION", duel52_engine::VERSION)?;
    m.add_class::<PyGame>()?;
    m.add_class::<PyAgent>()?;
    m.add_function(wrap_pyfunction!(random_play_stats, m)?)?;
    m.add_function(wrap_pyfunction!(power_reference, m)?)?;
    m.add_function(wrap_pyfunction!(ladder_agents, m)?)?;
    m.add_function(wrap_pyfunction!(encoding_spec, m)?)?;
    m.add_function(wrap_pyfunction!(replay_shard, m)?)?;
    Ok(())
}
