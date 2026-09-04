//! Random-vs-random measurement — the Phase 1 deliverable.
//!
//! `PLAN.md` Phase 1: "**Deliverable:** random-vs-random statistics — game length
//! distribution, first-player win rate, how often games reach the stalemate cutoff, across
//! all three variants."
//!
//! Everything here is seeded off a contiguous range of game seeds, and the report prints
//! the config and the seed range, because `CLAUDE.md` is blunt about it: "An unreproducible
//! finding is not a finding."
//!
//! Two extra measurements ride along because they are nearly free and later phases need
//! them:
//!
//! - **Maximum side occupancy.** `DESIGN.md` §3 guesses 8 slots per side per lane as the
//!   encoding bound. Phase 1 can replace the guess with a measurement.
//! - **Hand size when the last pile empties.** `FINDINGS.md` hypothesises that hand size at
//!   pile-empty is a defensive resource — "every card in hand is a turn the opponent cannot
//!   close a lane". This records the raw quantity now so Phase 5 has a random-play baseline
//!   to compare learned play against.

use std::time::Instant;

use crate::agents::{Agent, RandomAgent};
use crate::config::{GameConfig, Variant};
use crate::outcome::{DrawReason, Outcome};
use crate::player::Player;
use crate::state::GameState;

/// RNG stream tags for the two agents. Distinct from every `setup.rs` stream, so how many
/// random numbers an agent consumes cannot perturb the deal.
///
/// Public so the CLI's `demo` command can replay exactly the game `stats` counted for a
/// given seed.
pub const AGENT_STREAM_P0: u64 = 0x4147_454E_5430_0001;
pub const AGENT_STREAM_P1: u64 = 0x4147_454E_5431_0001;

/// What one game produced.
#[derive(Clone, Copy, Debug)]
pub struct GameSummary {
    pub seed: u64,
    pub outcome: Outcome,
    /// Total player turns played, counting from 1.
    pub plies: u32,
    /// Total decisions made, including the free sub-choices powers open.
    pub decisions: u32,
    /// The ply on which every draw pile first became empty, unlocking base cards.
    pub ply_at_unlock: Option<u32>,
    /// Hand sizes `[P0, P1]` at that moment.
    pub hand_sizes_at_unlock: [usize; 2],
    /// Largest number of cards seen on one side of one lane.
    pub max_side_occupancy: usize,
    /// Total cards each player drew over the whole game. See `GameState::draws_taken`.
    pub draws_taken: [u32; 2],
}

/// Play one random-vs-random game.
///
/// The deal, P0's choices, and P1's choices are three independent streams derived from the
/// one `seed`, so the same seed always produces the same game.
pub fn play_random_game(config: GameConfig, seed: u64) -> GameSummary {
    let mut state = GameState::new(config, seed);
    let mut p0 = RandomAgent::derived(seed, AGENT_STREAM_P0);
    let mut p1 = RandomAgent::derived(seed, AGENT_STREAM_P1);

    let mut summary = GameSummary {
        seed,
        outcome: state.outcome,
        plies: 1,
        decisions: 0,
        ply_at_unlock: None,
        hand_sizes_at_unlock: [0, 0],
        max_side_occupancy: 0,
        draws_taken: [0, 0],
    };

    let record = |state: &GameState, summary: &mut GameSummary| {
        for lane in &state.lanes {
            for side in &lane.sides {
                summary.max_side_occupancy = summary.max_side_occupancy.max(side.len());
            }
        }
        if summary.ply_at_unlock.is_none() && state.base_unlocked {
            summary.ply_at_unlock = Some(state.ply);
            summary.hand_sizes_at_unlock = [state.hands[0].len(), state.hands[1].len()];
        }
    };

    record(&state, &mut summary);
    while !state.outcome.is_over() {
        let legal = state.legal_actions();
        let action = match state.to_move {
            Player::P0 => p0.choose(&state, &legal),
            Player::P1 => p1.choose(&state, &legal),
        };
        state.apply_trusted(action);
        summary.decisions += 1;
        record(&state, &mut summary);
    }

    summary.outcome = state.outcome;
    summary.plies = state.ply + 1;
    summary.draws_taken = state.draws_taken;
    summary
}

/// Aggregated results over a seed range.
#[derive(Clone, Debug)]
pub struct RandomPlayStats {
    pub config: GameConfig,
    pub first_seed: u64,
    pub games: usize,

    pub p0_wins: usize,
    pub p1_wins: usize,
    pub draws: usize,
    pub draws_stalemate: usize,
    pub draws_mutual_lane_win: usize,
    pub draws_ply_limit: usize,

    /// Game lengths in plies, sorted ascending.
    pub lengths: Vec<u32>,
    /// Plies at which base cards unlocked, sorted ascending. Games that ended before the
    /// unlock contribute nothing.
    pub unlock_plies: Vec<u32>,
    /// Combined hand size of both players at the unlock, sorted ascending.
    pub hand_at_unlock: Vec<u32>,
    pub max_side_occupancy: usize,
    pub total_decisions: u64,
    pub elapsed_secs: f64,

    /// Total draws taken by each player, summed over every game.
    ///
    /// The direct measurement of the `game_rules.md` §10a parity claim. Under the house
    /// rule the 2 is pile-neutral, so both players draw the pile dry and the totals differ
    /// only by the odd card left when a game ends mid-pile. Under `two_power = discard`
    /// each firing shrinks the pile — and in the **base** variant that pile is shared, so
    /// the shrinkage lands on whoever would have drawn last.
    pub draws_taken: [u64; 2],
    /// Games in which one player drew strictly more cards than the other.
    pub games_with_unequal_draws: usize,
}

impl RandomPlayStats {
    /// P0's score: 1 per win, 0.5 per draw. `0.5` means no first-player advantage.
    pub fn p0_score(&self) -> f64 {
        if self.games == 0 {
            return 0.5;
        }
        (self.p0_wins as f64 + 0.5 * self.draws as f64) / self.games as f64
    }

    /// Half-width of a 95% confidence interval on [`RandomPlayStats::p0_score`].
    ///
    /// Normal approximation on the per-game score, which is what a Bernoulli-with-ties
    /// estimate needs; draws contribute 0.25 to the variance of a single game rather than
    /// 0, so counting them as half-wins without adjusting the variance would understate the
    /// interval.
    pub fn p0_score_ci95(&self) -> f64 {
        if self.games < 2 {
            return f64::NAN;
        }
        let n = self.games as f64;
        let mean = self.p0_score();
        // Sum of squared scores: wins contribute 1, draws 0.25, losses 0.
        let sum_sq = self.p0_wins as f64 + 0.25 * self.draws as f64;
        let variance = (sum_sq / n - mean * mean).max(0.0);
        1.96 * (variance / n).sqrt()
    }

    pub fn draw_rate(&self) -> f64 {
        self.draws as f64 / self.games.max(1) as f64
    }

    /// The headline number `PLAN.md` asks for: how often games reach the stalemate cutoff.
    pub fn stalemate_rate(&self) -> f64 {
        self.draws_stalemate as f64 / self.games.max(1) as f64
    }

    pub fn games_per_sec(&self) -> f64 {
        if self.elapsed_secs <= 0.0 {
            f64::INFINITY
        } else {
            self.games as f64 / self.elapsed_secs
        }
    }

    fn mean(values: &[u32]) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64
    }

    /// Nearest-rank percentile of a pre-sorted slice. `q` in `0.0..=1.0`.
    fn percentile(sorted: &[u32], q: f64) -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        let idx = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
        sorted[idx] as f64
    }

    /// A human-readable report.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Random vs random — {} games, seeds {}..{}\n",
            self.games,
            self.first_seed,
            self.first_seed + self.games as u64 - 1
        ));
        out.push_str(&format!("  config: {}\n", self.config.summary()));
        out.push_str(&format!(
            "  throughput: {:.0} games/sec ({:.2}s total, {} decisions)\n",
            self.games_per_sec(),
            self.elapsed_secs,
            self.total_decisions
        ));
        out.push_str(&format!(
            "  results: P0 {} ({:.1}%) · P1 {} ({:.1}%) · draw {} ({:.1}%)\n",
            self.p0_wins,
            100.0 * self.p0_wins as f64 / self.games.max(1) as f64,
            self.p1_wins,
            100.0 * self.p1_wins as f64 / self.games.max(1) as f64,
            self.draws,
            100.0 * self.draw_rate(),
        ));
        out.push_str(&format!(
            "  P0 score: {:.4} +/- {:.4} (95% CI)\n",
            self.p0_score(),
            self.p0_score_ci95()
        ));
        out.push_str(&format!(
            "  draws by reason: stalemate {} ({:.1}%) · mutual lane win {} · ply cap {}\n",
            self.draws_stalemate,
            100.0 * self.stalemate_rate(),
            self.draws_mutual_lane_win,
            self.draws_ply_limit,
        ));
        out.push_str(&format!(
            "  game length (plies): min {} · p10 {} · median {} · mean {:.1} · p90 {} · max {}\n",
            Self::percentile(&self.lengths, 0.0),
            Self::percentile(&self.lengths, 0.10),
            Self::percentile(&self.lengths, 0.50),
            Self::mean(&self.lengths),
            Self::percentile(&self.lengths, 0.90),
            Self::percentile(&self.lengths, 1.0),
        ));
        if self.unlock_plies.is_empty() {
            out.push_str("  base unlock: never reached in any game\n");
        } else {
            out.push_str(&format!(
                "  base unlock: reached in {}/{} games, median ply {} (mean {:.1})\n",
                self.unlock_plies.len(),
                self.games,
                Self::percentile(&self.unlock_plies, 0.50),
                Self::mean(&self.unlock_plies),
            ));
            out.push_str(&format!(
                "  combined hand size at unlock: median {} (mean {:.1}, max {})\n",
                Self::percentile(&self.hand_at_unlock, 0.50),
                Self::mean(&self.hand_at_unlock),
                Self::percentile(&self.hand_at_unlock, 1.0),
            ));
        }
        out.push_str(&format!(
            "  max cards on one side of one lane: {} (encoding bound in use: {})\n",
            self.max_side_occupancy, self.config.max_slots_per_side
        ));
        out.push_str(&format!(
            "  draws taken: P0 {} · P1 {} (P0 advantage {:+.4}/game, unequal in {:.1}% of games)\n",
            self.draws_taken[0],
            self.draws_taken[1],
            self.draw_advantage_p0(),
            100.0 * self.games_with_unequal_draws as f64 / self.games.max(1) as f64,
        ));
        out
    }

    /// Mean extra cards P0 drew per game, relative to P1.
    ///
    /// This is the `game_rules.md` §10a parity lever, measured. Zero means the 2 is
    /// pile-neutral as the house rule intends; a positive number means the rule is handing
    /// the first player extra draws.
    pub fn draw_advantage_p0(&self) -> f64 {
        (self.draws_taken[0] as f64 - self.draws_taken[1] as f64) / self.games.max(1) as f64
    }

    /// One row of a Markdown table, for pasting into `FINDINGS.md`.
    pub fn markdown_row(&self) -> String {
        format!(
            "| {} | {} | {} | {:.4} ± {:.4} | {:.1}% | {:.1}% | {:.0} | {:+.3} | {} |",
            self.config.variant,
            self.config.two_power,
            self.games,
            self.p0_score(),
            self.p0_score_ci95(),
            100.0 * self.draw_rate(),
            100.0 * self.stalemate_rate(),
            Self::mean(&self.lengths),
            self.draw_advantage_p0(),
            self.max_side_occupancy,
        )
    }

    /// Header for [`RandomPlayStats::markdown_row`].
    pub fn markdown_header() -> String {
        "| variant | 2's power | games | P0 score (95% CI) | draw | stalemate | mean plies | \
         P0 draw edge | max lane |\n\
         |---|---|---:|---:|---:|---:|---:|---:|---:|"
            .to_string()
    }
}

/// Play `games` random-vs-random games on seeds `first_seed .. first_seed + games`.
pub fn run_random_games(config: GameConfig, first_seed: u64, games: usize) -> RandomPlayStats {
    let started = Instant::now();
    let mut stats = RandomPlayStats {
        config,
        first_seed,
        games,
        p0_wins: 0,
        p1_wins: 0,
        draws: 0,
        draws_stalemate: 0,
        draws_mutual_lane_win: 0,
        draws_ply_limit: 0,
        lengths: Vec::with_capacity(games),
        unlock_plies: Vec::new(),
        hand_at_unlock: Vec::new(),
        max_side_occupancy: 0,
        total_decisions: 0,
        elapsed_secs: 0.0,
        draws_taken: [0, 0],
        games_with_unequal_draws: 0,
    };

    for i in 0..games {
        let summary = play_random_game(config, first_seed + i as u64);
        match summary.outcome {
            Outcome::Win(Player::P0) => stats.p0_wins += 1,
            Outcome::Win(Player::P1) => stats.p1_wins += 1,
            Outcome::Draw(reason) => {
                stats.draws += 1;
                match reason {
                    DrawReason::Stalemate => stats.draws_stalemate += 1,
                    DrawReason::MutualLaneWin => stats.draws_mutual_lane_win += 1,
                    DrawReason::PlyLimit => stats.draws_ply_limit += 1,
                }
            }
            Outcome::Ongoing => unreachable!("play_random_game returned an unfinished game"),
        }
        stats.lengths.push(summary.plies);
        if let Some(ply) = summary.ply_at_unlock {
            stats.unlock_plies.push(ply);
            stats.hand_at_unlock
                .push((summary.hand_sizes_at_unlock[0] + summary.hand_sizes_at_unlock[1]) as u32);
        }
        stats.max_side_occupancy = stats.max_side_occupancy.max(summary.max_side_occupancy);
        stats.total_decisions += summary.decisions as u64;
        stats.draws_taken[0] += summary.draws_taken[0] as u64;
        stats.draws_taken[1] += summary.draws_taken[1] as u64;
        if summary.draws_taken[0] != summary.draws_taken[1] {
            stats.games_with_unequal_draws += 1;
        }
    }

    stats.lengths.sort_unstable();
    stats.unlock_plies.sort_unstable();
    stats.hand_at_unlock.sort_unstable();
    stats.elapsed_secs = started.elapsed().as_secs_f64();
    stats
}

/// Run the full Phase 1 sweep: all three variants, plus the `two_power` comparison the
/// house rule in `game_rules.md` §10a asks to be measured rather than assumed.
pub fn phase1_sweep(first_seed: u64, games: usize) -> Vec<RandomPlayStats> {
    let mut out = Vec::new();
    for variant in Variant::ALL {
        out.push(run_random_games(GameConfig::preset(variant), first_seed, games));
    }
    // §10a: "Whether the parity problem is real is an empirical question, and
    // `two_power: discard` exists precisely so Phase 1 can answer it."
    for variant in Variant::ALL {
        let mut cfg = GameConfig::preset(variant);
        cfg.two_power = crate::config::TwoPower::Discard;
        out.push(run_random_games(cfg, first_seed, games));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CLAUDE.md`: "Everything is seeded and deterministic. Same seed + same config →
    /// identical game. Non-reproducible results are bugs."
    #[test]
    fn random_games_are_reproducible() {
        for variant in Variant::ALL {
            let cfg = GameConfig::preset(variant);
            for seed in [0u64, 7, 1234] {
                let a = play_random_game(cfg, seed);
                let b = play_random_game(cfg, seed);
                assert_eq!(a.outcome, b.outcome, "{variant} seed {seed}");
                assert_eq!(a.plies, b.plies, "{variant} seed {seed}");
                assert_eq!(a.decisions, b.decisions, "{variant} seed {seed}");
            }
        }
    }

    #[test]
    fn every_random_game_terminates() {
        for variant in Variant::ALL {
            let stats = run_random_games(GameConfig::preset(variant), 0, 60);
            assert_eq!(stats.p0_wins + stats.p1_wins + stats.draws, 60);
            assert_eq!(
                stats.draws_ply_limit, 0,
                "{variant}: the ply-limit safety cap fired, which indicates a rules bug"
            );
        }
    }

    /// The house 2 is pile-neutral, so `game_rules.md` §10a predicts that turns-to-unlock
    /// is fixed at deal time: "One draw per turn and no pile shrinkage means the pile
    /// empties after exactly *pile size* turns, regardless of how many 2s are played."
    ///
    /// P0 draws on plies 0, 2, 4, ... so P0's 13-card pile empties on ply 24 and P1's on
    /// ply 25; the global unlock is therefore ply 25 in every split-deck game that gets
    /// that far.
    #[test]
    fn rule_10a_house_two_makes_turns_to_unlock_invariant() {
        let cfg = GameConfig::split_deck();
        let expected = 2 * (cfg.expected_pile_size() as u32 - 1) + 1;
        let mut seen = 0;
        for seed in 0..400u64 {
            let summary = play_random_game(cfg, seed);
            if let Some(ply) = summary.ply_at_unlock {
                assert_eq!(
                    ply, expected,
                    "seed {seed}: unlock ply must be fixed under the house 2"
                );
                seen += 1;
            }
        }
        assert!(seen > 0, "no game in the sample reached the unlock");
    }

    /// `game_rules.md` §10a, the reason the house rule exists: "With a shared 26-card pile
    /// the players alternate draws 13/13; remove one card and someone now gets an extra
    /// draw, and the player who fires the 2 chooses who."
    ///
    /// Measured directly rather than inferred from win rates. Because P0 draws first from
    /// the shared pile, every card the RAW 2 removes shifts the pile's parity, and the
    /// shrinkage lands on P1.
    ///
    /// The house baseline is **not** zero: P0 also draws first, so a game that ends before
    /// the pile is exhausted leaves P0 with the odd draw. That residual (~0.05 cards/game)
    /// is structural and has nothing to do with the 2. The RAW effect is an order of
    /// magnitude larger, which is what makes it a lever rather than noise.
    #[test]
    fn rule_10a_the_raw_two_hands_the_first_player_extra_draws_in_the_base_game() {
        let house = run_random_games(GameConfig::base(), 1, 4000);
        let mut raw_config = GameConfig::base();
        raw_config.two_power = crate::config::TwoPower::Discard;
        let raw = run_random_games(raw_config, 1, 4000);

        assert!(
            house.draw_advantage_p0() < 0.15,
            "the house 2 is pile-neutral, so P0's edge should stay near the structural \
             residual; got {:+.4} cards/game",
            house.draw_advantage_p0()
        );
        assert!(
            raw.draw_advantage_p0() > 0.35,
            "RAW should hand P0 a much larger draw advantage, got {:+.4} cards/game",
            raw.draw_advantage_p0()
        );
        assert!(
            raw.draw_advantage_p0() > 5.0 * house.draw_advantage_p0().max(0.01),
            "the RAW effect should dwarf the structural residual: RAW {:+.4} vs house {:+.4}",
            raw.draw_advantage_p0(),
            house.draw_advantage_p0()
        );
    }

    /// The control: the lever is absent from the split variants, because there you bottom
    /// or discard into **your own** pile — there is no shared pile whose parity to flip
    /// (`game_rules.md` §9a). This is what shows the effect above is really about pile
    /// *sharing* rather than about the 2 as such.
    #[test]
    fn rule_9a_the_split_deck_has_no_shared_pile_parity_to_flip() {
        for two_power in [
            crate::config::TwoPower::Bottom,
            crate::config::TwoPower::Discard,
        ] {
            let mut config = GameConfig::split_deck();
            config.two_power = two_power;
            let stats = run_random_games(config, 1, 4000);
            assert!(
                stats.draw_advantage_p0().abs() < 0.05,
                "{two_power:?}: split decks give each player their own pile, so there \
                 should be no systematic draw edge; got {:+.4}",
                stats.draw_advantage_p0()
            );
        }
    }

    /// A second consequence of the house rule, and a genuine difference between the two
    /// settings: bottoming **recycles**, so a player can take more draws than their pile
    /// holds cards. `game_rules.md` §10a: "Cards are recycled, not destroyed. A bottomed
    /// card *will* be drawn again if the pile outlasts it."
    ///
    /// Under RAW the pile only ever shrinks, so 13 draws is a hard ceiling in the split
    /// variant and every game hits it exactly.
    #[test]
    fn rule_10a_bottoming_recycles_cards_so_draw_counts_exceed_the_pile_size() {
        let pile = GameConfig::split_deck().expected_pile_size() as f64;

        let mut raw_config = GameConfig::split_deck();
        raw_config.two_power = crate::config::TwoPower::Discard;
        let raw = run_random_games(raw_config, 1, 2000);
        let raw_mean = raw.draws_taken[0] as f64 / raw.games as f64;
        assert!(
            raw_mean <= pile,
            "RAW destroys cards, so draws cannot exceed the pile size; got {raw_mean:.2}"
        );

        let house = run_random_games(GameConfig::split_deck(), 1, 2000);
        let house_mean = house.draws_taken[0] as f64 / house.games as f64;
        assert!(
            house_mean > pile,
            "bottoming recycles, so draws should exceed the {pile} card pile; got \
             {house_mean:.2}"
        );
    }
}
