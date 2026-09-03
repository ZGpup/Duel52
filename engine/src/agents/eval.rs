//! A hand-written position evaluation.
//!
//! # These numbers are a prior, not a finding
//!
//! Every constant below was written by reading `game_rules.md`, not measured. They exist so
//! `PLAN.md` Phase 2 has a "greedy heuristic agent (hand-written evaluation)" and a leaf
//! evaluator for PIMC — nothing here is a claim about the game. **Phase 4's whole job is to
//! replace this table with learned values**, and the interesting result will be wherever the
//! two disagree. Do not quote a weight from this file in `FINDINGS.md`.
//!
//! # What it is allowed to look at
//!
//! Evaluation runs in two modes, and the distinction is load-bearing rather than cosmetic:
//!
//! - [`View::Informed`] reads only what one player is entitled to know — face-up ranks,
//!   their own played face-down cards, anything a 4 showed them. An agent using this mode
//!   needs no determinization and provably cannot cheat.
//! - [`View::Omniscient`] reads every rank on the board. It is correct **only** on a world
//!   that came out of [`GameState::determinize`], where the hidden ranks were sampled rather
//!   than observed. Calling it on the engine's real state is exactly the cheat that
//!   `DESIGN.md` §6 warns about.
//!
//! # Shape of the evaluation
//!
//! Four terms, in rough order of how much they move the number:
//!
//! 1. **Bodies.** Hit points still standing, plus what each card's power is worth — split
//!    into a *constant* power that is live while face-up (8, 9, 10, J) and a *one-shot* that
//!    is still in the bank while face-down (A, 2, 4, 5, 6, 7, Q, K).
//! 2. **The lane path.** You need `lanes_to_win` lanes emptied, not all of them
//!    (`game_rules.md` §7), so the objective is the total hit points standing in the
//!    opponent's *cheapest* two lanes — and symmetrically, in your own. This is the term
//!    that makes concentration emerge rather than being hard-coded, which matters because
//!    `FINDINGS.md` H3 is a hypothesis the project intends to *test*.
//! 3. **Hand.** Cards in hand are bodies you have not spent, and while you hold one the
//!    opponent cannot close a lane at all (§7). `FINDINGS.md` H2 says this is undervalued by
//!    intuition; the weight here is deliberately modest so the ladder does not smuggle the
//!    hypothesis in as an assumption.
//! 4. **Small corrections.** Pairs, freeze.

use crate::config::GameConfig;
use crate::player::Player;
use crate::rank::Rank;
use crate::state::GameState;

/// How much of the board an evaluation is entitled to read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    /// Read only the ranks this player knows: face-up cards, their own played face-down
    /// cards, and anything a 4 revealed to them.
    Informed(Player),
    /// Read every rank. **Only valid on a determinized world** — see the module docs.
    Omniscient,
}

impl View {
    /// Can this view read `card`'s rank?
    #[inline]
    fn reads(self, card: &crate::card::Card) -> bool {
        match self {
            View::Omniscient => true,
            View::Informed(p) => card.rank_known_to(p),
        }
    }
}

/// The tunable weights. See the module docs: a prior, not a measurement.
#[derive(Clone, Copy, Debug)]
pub struct EvalWeights {
    /// Per hit point still standing on a card in play.
    pub hit_point: f32,
    /// Per card in hand.
    pub hand_card: f32,
    /// Per point of hit points standing in the `lanes_to_win` cheapest lanes.
    pub lane_path: f32,
    /// Per declared pair. A pair turns two cards' attacks into one action for 2 damage
    /// (`game_rules.md` §5), which is tempo, but it also locks both members together.
    pub pair: f32,
    /// Per frozen card — a lost turn for that card (§8).
    pub frozen: f32,
    /// Logistic scale. Raw scores are squashed through this into `0.0..1.0` so the number
    /// can be mixed with terminal outcomes and backed up by a search.
    pub scale: f32,
}

impl Default for EvalWeights {
    fn default() -> EvalWeights {
        EvalWeights {
            hit_point: 1.0,
            hand_card: 0.9,
            lane_path: 0.8,
            pair: 0.4,
            frozen: -0.5,
            scale: 8.0,
        }
    }
}

/// What a card's power is worth **while it is face-up**.
///
/// Only the four constant powers (`game_rules.md` §6: 8, 9, 10, J) have ongoing value here.
/// A face-up A, 2, 4, 5, 6, 7, Q or K has already fired; what is left is a body that can be
/// paired and that a King could refire, which is the residual below.
const FACE_UP_POWER: [f32; Rank::COUNT] = [
    // A    2     3     4     5     6     7     8     9     10    J     Q     K
    0.15, 0.10, 0.10, 0.10, 0.15, 0.10, 0.10, 0.60, 0.45, 0.70, 0.90, 0.15, 0.20,
];

/// What a card's power is worth **while it is still face-down** — the one-shot in the bank.
///
/// The 3 is the odd one out and the reason this is a separate table rather than a copy of
/// the one above: Trap only does anything *while face-down* (§6), so flipping a 3
/// voluntarily throws its power away.
const ARMED_POWER: [f32; Rank::COUNT] = [
    // A    2     3     4     5     6     7     8     9     10    J     Q     K
    0.85, 0.35, 0.60, 0.25, 0.70, 0.60, 0.65, 0.60, 0.45, 0.70, 0.90, 0.50, 0.80,
];

/// What a face-down card of *unknown* rank is worth: the mean of [`ARMED_POWER`].
///
/// Using the mean rather than zero is what keeps the evaluation from leaking. An observer
/// who cannot read a rank values every hidden card identically, so no amount of staring at
/// the evaluation output can tell two hidden ranks apart.
fn unknown_armed_value(config: &GameConfig) -> f32 {
    let n = config.rank_count();
    ARMED_POWER[..n].iter().sum::<f32>() / n as f32
}

/// Evaluate `state` from `me`'s point of view, as a win probability in `0.0..=1.0`.
///
/// Terminal positions return the exact outcome, so a search can mix heuristic leaves and
/// real results without rescaling.
pub fn evaluate(state: &GameState, me: Player, view: View, w: &EvalWeights) -> f32 {
    if state.outcome.is_over() {
        return state.outcome.value_for(me);
    }
    let raw = raw_score(state, me, view, w);
    1.0 / (1.0 + (-raw / w.scale).exp())
}

/// The unsquashed score. Positive favours `me`. Exposed for tests and for tuning, where
/// looking at the logistic output compresses everything into the middle.
pub fn raw_score(state: &GameState, me: Player, view: View, w: &EvalWeights) -> f32 {
    side_score(state, me, view, w) - side_score(state, me.other(), view, w)
}

/// Everything one player's position is worth.
fn side_score(state: &GameState, p: Player, view: View, w: &EvalWeights) -> f32 {
    let unknown = unknown_armed_value(&state.config);
    let mut total = 0.0;
    let mut pairs = 0.0;

    for (_, _, card) in state.cards_of(p) {
        total += w.hit_point * card.hp_remaining() as f32;
        total += if card.face_up {
            FACE_UP_POWER[card.rank.index()]
        } else if view.reads(card) {
            ARMED_POWER[card.rank.index()]
        } else {
            unknown
        };
        if card.is_frozen(state.ply) {
            total += w.frozen;
        }
        if card.pair_id.is_some() {
            pairs += 0.5; // each pair is counted once from each of its two members
        }
    }
    total += w.pair * pairs;

    // `game_rules.md` §7: a win needs `lanes_to_win` lanes in which the opponent has
    // nothing left. So what matters about *this* player's position is how much still stands
    // in the lanes an opponent would go for — the cheapest ones.
    let mut lane_hp: Vec<f32> = state
        .lanes
        .iter()
        .map(|lane| {
            lane.side(p)
                .iter()
                .map(|c| c.hp_remaining() as f32)
                .sum::<f32>()
        })
        .collect();
    lane_hp.sort_by(|a, b| a.partial_cmp(b).expect("lane hit points are never NaN"));
    let need = state.config.lanes_to_win.min(lane_hp.len());
    total += w.lane_path * lane_hp[..need].iter().sum::<f32>();

    // Hand size is public for both players (`game_rules.md` §4 — you announce nothing, but
    // the count is visible), so this term is legal in either view.
    total += w.hand_card * state.hands[p.idx()].len() as f32;

    total
}
