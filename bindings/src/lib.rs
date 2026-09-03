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
use duel52_engine::outcome::{DrawReason, Outcome};
use duel52_engine::player::Player;
use duel52_engine::rank::Rank;
use duel52_engine::state::GameState;
use duel52_engine::Action;

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
        out.set_item("first_player", s.to_move == Player::P0)?;

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

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust engine for Duel 52. Imported via the `duel52` package.")?;
    m.add("VERSION", duel52_engine::VERSION)?;
    m.add_class::<PyGame>()?;
    m.add_function(wrap_pyfunction!(random_play_stats, m)?)?;
    m.add_function(wrap_pyfunction!(power_reference, m)?)?;
    Ok(())
}
