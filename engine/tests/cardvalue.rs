//! The card value table's method, checked without a checkpoint.
//!
//! `engine/src/cardvalue.rs` measures what a card is worth by substituting a rank into the
//! observer's hand and reading the value head. Two things have to be true for that to mean
//! anything, and neither depends on the network:
//!
//! 1. The substitution must leave the deck consistent. Over-subscribing a rank drives
//!    `unseen_counts` negative, which trips a `debug_assert` in `determinize.rs` and, in
//!    release, clamps at zero and perturbs the belief features **per rank** — a difference in
//!    the tensor that has nothing to do with the card's value. These tests run in debug, so
//!    the assertion is live.
//!
//! 2. Substituting into the **opponent's** hand must not move the observation at all, because
//!    the tensor carries `my_hand_counts` for the observer and only `opponent_hand_size` for
//!    the other side. That is the null control the table prints, and it is checked here
//!    against a hash rather than a network — which is strictly stronger, because a real value
//!    head could return the same number for two different tensors by luck.

mod common;

use duel52_engine::cardvalue;
use duel52_engine::nn::Evaluator;
use duel52_engine::{GameConfig, Rank};

/// A stand-in for a network that maps an observation to a value **injectively enough to
/// notice a changed float**.
///
/// The point is the contrapositive: if this returns the same value for two observations, they
/// were almost certainly the same observation. A trained value head could easily return the
/// same number for two different positions, so testing the control against one would prove
/// much less than testing it against this.
struct HashEvaluator {
    obs_dim: usize,
    action_dim: usize,
}

impl Evaluator for HashEvaluator {
    fn eval_batch(&self, obs: &[f32], n: usize, logits_out: &mut [f32], values_out: &mut [f32]) {
        for row in 0..n {
            let start = row * self.obs_dim;
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &v in &obs[start..start + self.obs_dim] {
                for byte in v.to_bits().to_le_bytes() {
                    h ^= byte as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            // Into (-1, 1), the range a real value head speaks in.
            values_out[row] = ((h >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
            for i in 0..self.action_dim {
                logits_out[row * self.action_dim + i] = 0.0;
            }
        }
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }
}

fn evaluator(config: &GameConfig) -> HashEvaluator {
    HashEvaluator {
        obs_dim: duel52_engine::encode::obs_dim(config),
        action_dim: duel52_engine::encode::action_dim(config),
    }
}

/// **The control, checked exactly.**
///
/// Substituting each of the thirteen ranks into the *opponent's* hand must leave the
/// observation bit-identical, because the observer is not entitled to know what is in it.
/// Against a hash of the tensor, "the value did not move" means "the tensor did not move".
#[test]
fn card_value_control_is_exactly_flat_across_ranks() {
    let config = GameConfig::default();
    let table = cardvalue::measure(&evaluator(&config), &config, 60, 1);

    assert!(table.positions > 20, "too few positions to prove anything");
    assert_eq!(
        table.control_spread, 0.0,
        "substituting into the opponent's hand moved the observation — the observer is \
         reading a hand they cannot see"
    );
    assert!(table.control_is_clean());
}

/// The measurement must actually *do* something: substituting into our **own** hand has to
/// move the observation, or the table would be thirteen readings of one position.
///
/// The companion to the control above, and the reason the control is not vacuous.
#[test]
fn card_value_measurement_moves_the_observation() {
    let config = GameConfig::default();
    let table = cardvalue::measure(&evaluator(&config), &config, 60, 1);

    let distinct = {
        let mut vs: Vec<u64> = table.rows.iter().map(|r| r.in_hand.to_bits()).collect();
        vs.sort_unstable();
        vs.dedup();
        vs.len()
    };
    assert_eq!(
        distinct,
        table.rows.len(),
        "two ranks produced the identical in-hand reading, so the substitution is not \
         reaching the observation"
    );
}

/// **Deck consistency.** Running the measurement exercises `unseen_counts`, whose
/// `debug_assert` fires if the observer can account for more copies of a rank than the deck
/// holds. This runs it over thousands of substitutions in a debug build.
///
/// If the filter in `sample_positions` were removed, this is the test that would fail — it is
/// how the first version's hand-reasoned precondition was caught.
#[test]
fn card_value_substitutions_never_over_subscribe_the_deck() {
    for variant in [duel52_engine::Variant::Base, duel52_engine::Variant::SplitDeck] {
        let config = GameConfig::preset(variant);
        let table = cardvalue::measure(&evaluator(&config), &config, 40, 7);
        println!(
            "{variant}: {} positions measured, {} rejected",
            table.positions, table.rejected
        );
        assert!(
            table.positions > 10,
            "{variant}: only {} positions survived the deck-consistency filter ({} rejected), \
             which is too thin a sample to measure a card on",
            table.positions,
            table.rejected
        );
    }
}

/// **The method does not reach `mirrored`, and that is a fact about the variant.**
///
/// To ask "what if I held an `R`", some copy of `R` has to be somewhere the observer cannot
/// see. §9b reveals the removed multiset at setup, so a mirrored-removal observer accounts
/// for nearly every card in their own deck and almost no rank has an unseen copy left —
/// which means almost no position admits all thirteen substitutions at once.
///
/// Measured: **0.06% of sampled positions survive**, against 8% for `split` and 91% for
/// `base`, whose four copies per rank leave much more slack.
///
/// Asserted rather than merely noted, so that if the filter is ever loosened by accident,
/// this fails and says why the numbers moved instead of quietly producing a table nobody can
/// interpret.
#[test]
fn card_value_cannot_measure_the_mirrored_variant() {
    let config = GameConfig::preset(duel52_engine::Variant::MirroredRemoval);
    let table = cardvalue::measure(&evaluator(&config), &config, 40, 7);
    println!(
        "mirrored: {} positions measured, {} rejected",
        table.positions, table.rejected
    );
    assert!(
        table.positions < 10,
        "mirrored now yields {} usable positions where it used to yield ~1. Either the \
         deck-consistency filter was loosened — check it is still exact — or the sampler \
         changed. The `card-value` docs claim this variant is out of reach; update them.",
        table.positions
    );
}

/// Every rank in play gets a row, in a table that covers the configured deck and no more.
#[test]
fn card_value_covers_every_rank_in_play() {
    let config = GameConfig::default();
    let table = cardvalue::measure(&evaluator(&config), &config, 30, 3);
    assert_eq!(table.rows.len(), config.rank_count());
    for (i, row) in table.rows.iter().enumerate() {
        assert_eq!(row.rank, Rank::from_index(i));
    }
}

/// The paired advantages are deviations from each position's own mean, so they sum to zero
/// across ranks by construction. A drift means the pairing is wrong.
#[test]
fn card_value_advantages_are_deviations_and_sum_to_zero() {
    let config = GameConfig::default();
    let table = cardvalue::measure(&evaluator(&config), &config, 60, 11);
    let total: f64 = table.rows.iter().map(|r| r.advantage).sum();
    assert!(
        total.abs() < 1e-9,
        "paired advantages sum to {total}, not 0 — they are not deviations from a common mean"
    );
}
