//! What is a card worth? — `PLAN.md` §4's card value table.
//!
//! # The question, and why the flip-timing curve does not answer it
//!
//! `FINDINGS.md` has the order in which a strong agent turns each rank face-up, spanning
//! twenty-two turns from the 8 to the Queen. `PLAN.md` is careful that this is *when a power
//! starts paying*, not *what it is worth*, and it is right to be: a card flipped early may be
//! flipped early because it is cheap to commit, not because it is good.
//!
//! # Vary the card **in hand**, not the card on the board
//!
//! This is the whole design decision, and the first version of this module got it wrong in a
//! way worth recording, because the wrong answer looked like a finding.
//!
//! That version put rank `R` **face-up in a lane** and read the value head. Its top four
//! cards came out 8, J, 10, 9 — which is exactly `PowerId::is_constant()`, the four constant
//! powers, in a block, above everything else. That is not a fact about Duel 52. It is the
//! method reading its own selection criterion back out:
//!
//! | power kind | ranks | what "face-up in a lane" measures |
//! |---|---|---|
//! | constant | 8, 9, 10, J | the power, live and working — **100% of its value** |
//! | one-shot | A, 2, 4, 5, 6, 7, Q, K | a **spent** card. It fires on the flip; face-up means it already fired and the effect is in the past |
//! | conditional | 3 | **nothing**. The Trap works only while face-down, so a face-up 3 has no power at all |
//!
//! So the old table ranked *how much of a card's value survives being face-up*, which is a
//! property of the power's **type**, and the ordering it produced was a restatement of that
//! type. `game_rules.md` §6's own three-way split predicts it exactly.
//!
//! Varying a card **in hand** fixes this, because a card in hand has its whole future ahead
//! of it whatever kind of power it carries: it can be played, held face-down as a Trap, or
//! flipped at the moment its one-shot pays. The value head's estimate then integrates over
//! how the agent would actually use it. Every rank is measured at the same point in its life
//! cycle, which is the property the board measurement lacked.
//!
//! The board number is still reported, because the **gap** between the two says where a
//! card's value lives — a card worth much more in hand than on the board is one whose value
//! is all in the flip.
//!
//! # Keeping the deck honest
//!
//! A substitution must not create a third copy of a rank the observer already holds two of.
//! The belief features are `unseen_counts` (`determinize.rs`), computed by running the deck
//! composition down by every card the observer can place — so an over-subscribed rank drives
//! a count negative, trips a `debug_assert`, and in release clamps at zero, which perturbs
//! the tensor **per rank** for a reason that has nothing to do with the card's value.
//!
//! So a position is used only when **every** substitution it will be asked to make leaves the
//! deck consistent, checked by building each hypothetical world and asking
//! [`GameState::deck_is_consistent`] rather than by reasoning about it. Positions that fail
//! are skipped and counted, and the count is reported.
//!
//! ## What that filter costs, and the selection bias it introduces
//!
//! Asking "what if I held an `R`" only makes sense where some copy of `R` could actually be
//! somewhere the observer cannot see. That is a fact about the position, not a limitation of
//! the implementation, and it bites very differently per variant:
//!
//! | variant | positions kept | why |
//! |---|---:|---|
//! | `base` | ~91% | four copies of every rank in one shared deck — lots of slack |
//! | `split` | ~8% | two copies per player; by mid-game many ranks are fully accounted for |
//! | `mirrored` | ~0.06% | §9b publishes the removed multiset, so the observer accounts for nearly their whole deck |
//!
//! **`mirrored` is out of reach** and the CLI says so rather than printing a table built on a
//! handful of positions. `engine/tests/cardvalue.rs` asserts it, so a change that appears to
//! fix it gets checked rather than believed.
//!
//! ⚠️ **The surviving positions are biased, and the bias is worth stating.** A position where
//! all thirteen ranks are still plausibly holdable is one where relatively little has been
//! discarded or revealed — so the table is measured on **earlier, higher-uncertainty
//! positions** than a uniform sample of play. That is defensible for this question, since a
//! card in hand with two turns left is worth little whatever its rank, but it is a bias and
//! not a neutral sample.
//!
//! # The control
//!
//! Every measurement is run a third time, substituting into the **opponent's** hand. The
//! observation carries `my_hand_counts` for the observer and nothing but
//! `opponent_hand_size` for the other side, so all thirteen tensors are bit-identical and
//! the value head must return one number thirteen times.
//!
//! It exercises the same code path as the measurement, which the old face-down control did
//! not. Report it. A table without it is an assertion.
//!
//! # What it is blocked on
//!
//! A trustworthy value head. `PLAN.md` §4 names this as the one prerequisite, and it is what
//! the long runs are for. The numbers here are real measurements of *the checkpoint you give
//! it*; whether that checkpoint's value head is good enough to be evidence about **Duel 52**
//! rather than about itself is a separate question this module cannot answer.

use crate::config::GameConfig;
use crate::nn::Evaluator;
use crate::player::Player;
use crate::rank::Rank;
use crate::rng::Rng;
use crate::state::GameState;

/// One rank's measured value, in win-probability points.
#[derive(Clone, Debug)]
pub struct CardValue {
    pub rank: Rank,
    /// **The measurement.** Mean value of the position with this rank held in hand.
    pub in_hand: f64,
    /// How much better than the average card this rank is *in hand*, on the same board.
    ///
    /// A **paired** statistic: every rank is measured on the identical position, so the
    /// position's own difficulty subtracts out exactly. This is the column to read.
    pub advantage: f64,
    /// Standard error of [`CardValue::advantage`], typically an order of magnitude tighter
    /// than the error on the absolute value for the reason above.
    pub advantage_stderr: f64,
    /// Mean value with this rank **face-up in a lane** instead.
    ///
    /// ⚠️ Not comparable across power kinds — see the module docs. A one-shot has already
    /// fired by the time it is face-up and a 3 has no power at all, so this systematically
    /// favours the four constant powers. Read it only as a contrast with `advantage`.
    pub on_board: f64,
    /// `on_board`'s paired advantage, same caveat.
    pub board_advantage: f64,
    /// The null control: this rank substituted into the **opponent's** hand, which the
    /// observer cannot see. Must be identical for every rank.
    pub control: f64,
}

/// The finished table.
#[derive(Clone, Debug)]
pub struct ValueTable {
    pub rows: Vec<CardValue>,
    /// Positions that survived the deck-consistency filter and were actually measured.
    pub positions: usize,
    /// Positions rejected because some rank could not be substituted without creating a
    /// card the deck does not contain.
    pub rejected: usize,
    /// Spread of the control across ranks, in win-probability points. **Must be ~0.**
    pub control_spread: f64,
}

impl ValueTable {
    /// Ranks ordered by in-hand advantage, best first.
    pub fn ranked(&self) -> Vec<&CardValue> {
        let mut rows: Vec<&CardValue> = self.rows.iter().collect();
        rows.sort_by(|a, b| b.advantage.partial_cmp(&a.advantage).expect("no NaN"));
        rows
    }

    /// Best card minus worst, in win-probability points. The number a balance decision is
    /// actually about: a spread of 1 point is a balanced game and a spread of 20 is not.
    pub fn spread(&self) -> f64 {
        let hi = self
            .rows
            .iter()
            .map(|r| r.advantage)
            .fold(f64::NEG_INFINITY, f64::max);
        let lo = self
            .rows
            .iter()
            .map(|r| r.advantage)
            .fold(f64::INFINITY, f64::min);
        hi - lo
    }

    /// Is the control tight enough for the table to mean anything?
    ///
    /// Strict on purpose: the tensors being compared are *identical*, so the only spread a
    /// correct implementation can produce is float non-determinism inside one forward pass,
    /// which is zero for this engine's reference kernels.
    pub fn control_is_clean(&self) -> bool {
        self.control_spread < 1e-3
    }
}

/// A sampled position, plus the observer and where to make the substitutions.
struct Sample {
    state: GameState,
    observer: Player,
    /// Index into the observer's hand whose rank gets replaced.
    hand_slot: usize,
    /// Index into the opponent's hand, for the control.
    their_hand_slot: usize,
    /// Lane holding a card of the observer's, for the secondary board measurement.
    lane: usize,
}

/// The observer's hand with `slot` replaced by `rank`, re-sorted.
///
/// Hands are kept sorted so equal positions compare equal. The observation reads rank
/// *counts*, so sorting does not move the tensor — it keeps the state honest.
fn with_hand_rank(state: &GameState, who: Player, slot: usize, rank: Rank) -> GameState {
    let mut world = state.clone();
    world.hands[who.idx()][slot] = rank;
    world.hands[who.idx()].sort_unstable();
    world
}

/// The observer's card in `lane` slot 0, replaced by a healthy face-up `rank`.
fn with_board_rank(state: &GameState, who: Player, lane: usize, rank: Rank) -> GameState {
    let mut world = state.clone();
    let card = &mut world.lanes[lane].side_mut(who)[0];
    card.rank = rank;
    card.face_up = true;
    card.known_to = crate::card::KNOWN_TO_BOTH;
    card.is_base = false;
    // Damage survivable for the old rank may not be for the new one, so every substitution
    // is a healthy card and the comparison is like-for-like.
    card.damage = 0;
    world
}

/// Walk random games and keep positions where every rank can be substituted into the
/// observer's hand without inventing a card the deck does not hold.
///
/// Random rather than hand-built, for the reason `engine/tests/common` gives: a hand-built
/// position was chosen by someone who knew what they wanted to show.
fn sample_positions(
    config: GameConfig,
    wanted: usize,
    seed: u64,
    ranks: &[Rank],
) -> (Vec<Sample>, usize) {
    let mut out = Vec::new();
    let mut rejected = 0usize;
    let mut seed_at = seed;
    let limit = seed + wanted as u64 * 60;

    while out.len() < wanted && seed_at < limit {
        let mut state = GameState::new(config, seed_at);
        let mut rng = Rng::derive(seed_at, 0xCA2D_0A1E_0000_0001);
        // Deep enough that the board is not the opening, shallow enough that plenty of game
        // is left for a card in hand to be worth something.
        let depth = 12 + (seed_at % 30) as usize;
        let mut alive = true;
        for _ in 0..depth {
            if state.outcome.is_over() {
                alive = false;
                break;
            }
            let legal = state.legal_actions();
            let action = *rng.choose(&legal).expect("a running game has actions");
            state.apply_trusted(action);
        }
        seed_at += 1;
        if !alive || state.outcome.is_over() || !state.pending.is_empty() {
            continue;
        }

        let observer = state.acting_player();
        let opponent = observer.other();
        if state.hand(observer).is_empty() || state.hand(opponent).is_empty() {
            continue;
        }
        let lane = match (0..config.lanes).find(|&l| !state.lanes[l].side(observer).is_empty()) {
            Some(l) => l,
            None => continue,
        };

        // ⚠️ **Deck consistency, checked on every world that will actually be measured** —
        // not inferred from a precondition on the original position.
        //
        // The first version of this filter reasoned about it by hand: "the displaced card
        // returns to the unseen pool, so only the rank substituted *in* can over-subscribe,
        // so require `unseen_counts[R] >= 1`". That was wrong twice over. It aggregated
        // across pools, and the split variants give each player their own — a rank can have
        // room in P1's and none in P0's. And it covered only the hand substitution, ignoring
        // the board one entirely, where turning a hidden card into a known face-up `R`
        // consumes a copy that was never counted before. `engine/tests/cardvalue.rs` caught
        // it as a chain of `debug_assert` failures out of `determinize.rs`.
        //
        // Asking the state is both correct and shorter than being right about the rules.
        let all_legal = ranks.iter().all(|&r| {
            with_hand_rank(&state, observer, 0, r).deck_is_consistent(observer)
                && with_board_rank(&state, observer, lane, r).deck_is_consistent(observer)
                && with_hand_rank(&state, opponent, 0, r).deck_is_consistent(observer)
        });
        if !all_legal {
            rejected += 1;
            continue;
        }

        out.push(Sample {
            state,
            observer,
            hand_slot: 0,
            their_hand_slot: 0,
            lane,
        });
    }
    (out, rejected)
}

/// Measure every rank's contribution to the value head's estimate.
pub fn measure(
    evaluator: &dyn Evaluator,
    config: &GameConfig,
    positions: usize,
    seed: u64,
) -> ValueTable {
    let ranks: Vec<Rank> = Rank::ALL
        .into_iter()
        .filter(|r| r.index() <= config.max_rank_index)
        .collect();
    let (samples, rejected) = sample_positions(*config, positions, seed, &ranks);

    let obs_dim = crate::encode::obs_dim(config);
    let mut obs = vec![0.0f32; obs_dim];
    let mut logits = vec![0.0f32; crate::encode::action_dim(config)];
    let mut value = [0.0f32; 1];

    let mut in_hand: Vec<Vec<f64>> = vec![Vec::new(); ranks.len()];
    let mut on_board: Vec<Vec<f64>> = vec![Vec::new(); ranks.len()];
    let mut control: Vec<Vec<f64>> = vec![Vec::new(); ranks.len()];

    for sample in &samples {
        let observer = sample.observer;
        let opponent = observer.other();

        let mut read = |world: &GameState| -> f64 {
            crate::encode::encode_observation(world, observer, &mut obs);
            evaluator.eval_batch(&obs, 1, &mut logits, &mut value);
            to_win_probability(value[0])
        };

        for (i, &rank) in ranks.iter().enumerate() {
            // --- the measurement: this card in our hand, its whole life ahead of it ---
            in_hand[i].push(read(&with_hand_rank(
                &sample.state,
                observer,
                sample.hand_slot,
                rank,
            )));

            // --- the contrast: this card already face-up in a lane ---
            //
            // Not comparable across power kinds, and reported only against the column above.
            // See the module docs for why this was the wrong primary measurement.
            on_board[i].push(read(&with_board_rank(
                &sample.state,
                observer,
                sample.lane,
                rank,
            )));

            // --- the control: their hand, which we cannot see ---
            //
            // The observation carries `my_hand_counts` for us and only `opponent_hand_size`
            // for them, so all thirteen of these tensors are identical.
            control[i].push(read(&with_hand_rank(
                &sample.state,
                opponent,
                sample.their_hand_slot,
                rank,
            )));
        }
    }

    let advantage = paired(&in_hand, ranks.len());
    let board_advantage = paired(&on_board, ranks.len());

    let rows: Vec<CardValue> = ranks
        .iter()
        .enumerate()
        .map(|(i, &rank)| CardValue {
            rank,
            in_hand: mean(&in_hand[i]),
            advantage: mean(&advantage[i]),
            advantage_stderr: stderr(&advantage[i]),
            on_board: mean(&on_board[i]),
            board_advantage: mean(&board_advantage[i]),
            control: mean(&control[i]),
        })
        .collect();

    let controls: Vec<f64> = rows.iter().map(|r| r.control).collect();
    let control_spread = controls.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - controls.iter().cloned().fold(f64::INFINITY, f64::min);

    ValueTable {
        rows,
        positions: samples.len(),
        rejected,
        control_spread,
    }
}

/// Subtract each position's own mean across ranks, which removes the position's difficulty —
/// the term that dominates the raw spread and is identical for every rank being compared.
fn paired(columns: &[Vec<f64>], ranks: usize) -> Vec<Vec<f64>> {
    let n = columns.first().map(|v| v.len()).unwrap_or(0);
    let mut out: Vec<Vec<f64>> = vec![Vec::with_capacity(n); ranks];
    for p in 0..n {
        let here: f64 = columns.iter().map(|v| v[p]).sum::<f64>() / ranks as f64;
        for (i, col) in columns.iter().enumerate() {
            out[i].push(col[p] - here);
        }
    }
    out
}

/// The value head speaks in `(-1, 1)`; a balance decision is made in win probability.
fn to_win_probability(v: f32) -> f64 {
    100.0 * (v as f64 + 1.0) / 2.0
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn stderr(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return f64::NAN;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    (var / xs.len() as f64).sqrt()
}
