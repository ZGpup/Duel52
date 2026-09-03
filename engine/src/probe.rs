//! Instrumented play — the Phase 2 deliverable.
//!
//! `PLAN.md` Phase 2: "**Deliverable:** first real strategic observations logged to
//! `FINDINGS.md`", and the note under it: "A competent ISMCTS bot with zero training will
//! expose lane-allocation patterns, flip timing, and the first-player edge. Do not rush past
//! it to get to the neural net."
//!
//! So this module watches games rather than just counting them. It records what
//! `FINDINGS.md`'s hypotheses actually ask about:
//!
//! | Recorded | Answers |
//! |---|---|
//! | hand size when the last pile empties | **H2** — is hand size the resource, and where is the crossover |
//! | share of plays and attacks in a player's busiest two lanes | **H3** — does strong play concentrate |
//! | how many played cards are ever flipped, and how late | **H4** — is concealment worth its tempo |
//! | flips per rank, and the mean ply of each | **H5, H6, H7** — which cards get held, which get fired on sight |
//! | quiet-ply draws | **F1.2** — the stalemate rule, which random play could not reach |
//! | maximum cards on one side of one lane | **F1.7** — the Phase 3 encoding bound |
//!
//! # Instrumentation reads ground truth, and that is fine
//!
//! [`GameStats`] records the rank of every flipped card by looking at the engine's state.
//! That is measurement, not play: the number never re-enters a decision. The rule that
//! matters — agents see only their own information set — is enforced where decisions are
//! made, in [`crate::agents`] and [`GameState::determinize`].

use crate::action::Action;
use crate::agents::{Agent, AgentSpec};
use crate::config::GameConfig;
use crate::outcome::{DrawReason, Outcome};
use crate::player::Player;
use crate::rank::Rank;
use crate::state::GameState;

/// RNG stream tags for a match. Distinct from every `setup.rs` stream, so how many random
/// numbers an agent consumes cannot perturb the deal.
pub const AGENT_STREAM: [u64; 2] = [0x4147_454E_5430_0002, 0x4147_454E_5431_0002];

/// Everything one game produced.
///
/// Per-player arrays are indexed by [`Player::idx`], **not** by seat: `by_player[0]` is P0's,
/// whichever agent that was. [`MatchStats`] does the seat-to-agent mapping.
#[derive(Clone, Debug)]
pub struct GameStats {
    pub seed: u64,
    pub outcome: Outcome,
    /// Total player turns played, counting from 1.
    pub plies: u32,
    /// Total decisions made, including the free sub-choices powers open.
    pub decisions: u32,

    /// The ply on which every draw pile first became empty, unlocking base cards.
    pub ply_at_unlock: Option<u32>,
    /// Hand sizes at that moment — the H2 measurement.
    pub hand_at_unlock: [u32; 2],
    /// Hand sizes when the game ended.
    pub hand_at_end: [u32; 2],
    /// Largest number of cards seen on one side of one lane.
    pub max_side_occupancy: usize,
    pub draws_taken: [u32; 2],

    /// Cards played from hand, by player and lane.
    pub plays_by_lane: [Vec<u32>; 2],
    /// Attack actions, by player and lane.
    pub attacks_by_lane: [Vec<u32>; 2],
    /// Cards played from hand, by player and rank.
    pub plays_by_rank: [[u32; Rank::COUNT]; 2],
    /// Flips, by player and rank. Includes base cards, which were never "played".
    pub flips_by_rank: [[u32; Rank::COUNT]; 2],
    /// Of those, the ones that were base cards, by player and rank. Counted separately so
    /// the flip *rate* has a clean denominator: a base card was never played from hand, so
    /// it belongs to neither side of "how much of what I played did I turn face-up".
    pub base_flips_by_rank: [[u32; Rank::COUNT]; 2],
    /// Summed ply of every flip, by player and rank. Divide by `flips_by_rank` for the mean
    /// ply at which that rank goes face-up.
    pub flip_ply_sum: [[u64; Rank::COUNT]; 2],
    /// Cards still face-down on the board when the game ended.
    pub unflipped_at_end: [u32; 2],
    /// `Pass` actions — a turn forfeited with actions still in hand.
    pub passes: [u32; 2],
    pub pairs_declared: [u32; 2],
}

impl GameStats {
    fn new(config: &GameConfig, seed: u64) -> GameStats {
        GameStats {
            seed,
            outcome: Outcome::Ongoing,
            plies: 0,
            decisions: 0,
            ply_at_unlock: None,
            hand_at_unlock: [0, 0],
            hand_at_end: [0, 0],
            max_side_occupancy: 0,
            draws_taken: [0, 0],
            plays_by_lane: [vec![0; config.lanes], vec![0; config.lanes]],
            attacks_by_lane: [vec![0; config.lanes], vec![0; config.lanes]],
            plays_by_rank: [[0; Rank::COUNT]; 2],
            flips_by_rank: [[0; Rank::COUNT]; 2],
            base_flips_by_rank: [[0; Rank::COUNT]; 2],
            flip_ply_sum: [[0; Rank::COUNT]; 2],
            unflipped_at_end: [0, 0],
            passes: [0, 0],
            pairs_declared: [0, 0],
        }
    }

    /// Record what an action was, reading the board *before* it is applied — a `Flip` names
    /// a slot, and the card's rank has to be looked up while it is still there.
    fn note_action(&mut self, state: &GameState, action: Action) {
        let me = state.to_move.idx();
        match action {
            Action::Play { rank, lane } => {
                self.plays_by_lane[me][lane as usize] += 1;
                self.plays_by_rank[me][rank.index()] += 1;
            }
            Action::Attack { lane, .. } => self.attacks_by_lane[me][lane as usize] += 1,
            Action::DeclarePair { .. } => self.pairs_declared[me] += 1,
            Action::Pass => self.passes[me] += 1,
            Action::Flip { lane, slot } => {
                if let Some(card) = state.at(lane as usize, state.to_move, slot as usize) {
                    self.flips_by_rank[me][card.rank.index()] += 1;
                    self.flip_ply_sum[me][card.rank.index()] += state.ply as u64;
                    if card.entered_as_base {
                        self.base_flips_by_rank[me][card.rank.index()] += 1;
                    }
                }
            }
            _ => {}
        }
    }

    /// Record board facts that have to be sampled continuously rather than at the end.
    fn note_state(&mut self, state: &GameState) {
        for lane in &state.lanes {
            for side in &lane.sides {
                self.max_side_occupancy = self.max_side_occupancy.max(side.len());
            }
        }
        if self.ply_at_unlock.is_none() && state.base_unlocked {
            self.ply_at_unlock = Some(state.ply);
            self.hand_at_unlock = [state.hands[0].len() as u32, state.hands[1].len() as u32];
        }
    }

    fn finish(&mut self, state: &GameState) {
        self.outcome = state.outcome;
        self.plies = state.ply + 1;
        self.draws_taken = state.draws_taken;
        self.hand_at_end = [state.hands[0].len() as u32, state.hands[1].len() as u32];
        for p in Player::BOTH {
            self.unflipped_at_end[p.idx()] = state
                .cards_of(p)
                .filter(|(_, _, card)| !card.face_up)
                .count() as u32;
        }
    }

    /// Share of this player's plays that landed in their busiest `lanes_to_win` lanes.
    ///
    /// The H3 measurement. It has no meaningful absolute scale — picking the busiest lanes
    /// after the fact inflates it even for a uniform player — so read it only against the
    /// random rung's value in the same table. `None` when the player played no cards.
    pub fn lane_concentration(&self, p: Player, lanes_to_win: usize) -> Option<f64> {
        top_share(&self.plays_by_lane[p.idx()], lanes_to_win)
    }

    /// The same measure over attacks rather than plays.
    pub fn attack_concentration(&self, p: Player, lanes_to_win: usize) -> Option<f64> {
        top_share(&self.attacks_by_lane[p.idx()], lanes_to_win)
    }
}

/// Fraction of the total held by the `k` largest entries.
fn top_share(counts: &[u32], k: usize) -> Option<f64> {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return None;
    }
    let mut sorted: Vec<u32> = counts.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let top: u32 = sorted.iter().take(k).sum();
    Some(top as f64 / total as f64)
}

/// Play one instrumented game between two agents.
pub fn play_instrumented(
    config: GameConfig,
    seed: u64,
    p0: &mut dyn Agent,
    p1: &mut dyn Agent,
) -> GameStats {
    let mut state = GameState::new(config, seed);
    let mut stats = GameStats::new(&config, seed);
    stats.note_state(&state);

    while !state.outcome.is_over() {
        let legal = state.legal_actions();
        let action = match state.to_move {
            Player::P0 => p0.choose(&state, &legal),
            Player::P1 => p1.choose(&state, &legal),
        };
        stats.note_action(&state, action);
        state.apply_trusted(action);
        stats.decisions += 1;
        stats.note_state(&state);
    }

    stats.finish(&state);
    stats
}

/// Play one instrumented game between two [`AgentSpec`]s, building both from `seed`.
pub fn play_spec_game(
    config: GameConfig,
    seed: u64,
    p0: AgentSpec,
    p1: AgentSpec,
) -> GameStats {
    let mut a = p0.build(seed, AGENT_STREAM[0]);
    let mut b = p1.build(seed, AGENT_STREAM[1]);
    play_instrumented(config, seed, a.as_mut(), b.as_mut())
}

// ============================================================== aggregation ==

/// Aggregated statistics for one agent playing one role in a set of games.
///
/// Kept per *agent*, not per seat, so the numbers are about how that agent plays rather than
/// about which side of the table it sat on.
#[derive(Clone, Debug, Default)]
pub struct AgentBehaviour {
    pub games: usize,
    pub hand_at_unlock: Vec<u32>,
    /// Hand size at the unlock, split by how the game then ended.
    ///
    /// The sharp test of `FINDINGS.md` H2. Comparing hand size *across* agents confounds the
    /// hypothesis with agent strength; comparing winners against losers **within one agent's
    /// self-play** holds strength fixed and asks the question directly: among games this
    /// agent played against itself, did the side holding more cards at pile-empty win?
    pub hand_at_unlock_won: Vec<u32>,
    pub hand_at_unlock_lost: Vec<u32>,
    pub hand_at_end: Vec<u32>,
    pub plays: u64,
    pub flips: u64,
    pub base_flips: u64,
    pub base_flips_by_rank: [u64; Rank::COUNT],
    pub passes: u64,
    pub pairs: u64,
    pub unflipped_at_end: u64,
    pub plays_by_rank: [u64; Rank::COUNT],
    pub flips_by_rank: [u64; Rank::COUNT],
    pub flip_ply_sum: [u64; Rank::COUNT],
    pub lane_concentration: Vec<f64>,
    pub attack_concentration: Vec<f64>,
}

impl AgentBehaviour {
    fn absorb(&mut self, stats: &GameStats, p: Player, lanes_to_win: usize) {
        let i = p.idx();
        self.games += 1;
        if stats.ply_at_unlock.is_some() {
            self.hand_at_unlock.push(stats.hand_at_unlock[i]);
            match stats.outcome {
                Outcome::Win(w) if w == p => self.hand_at_unlock_won.push(stats.hand_at_unlock[i]),
                Outcome::Win(_) => self.hand_at_unlock_lost.push(stats.hand_at_unlock[i]),
                // A draw is evidence for neither side, so it is left out rather than
                // counted half in each bucket, which would blur the very difference the
                // split exists to show.
                _ => {}
            }
        }
        self.hand_at_end.push(stats.hand_at_end[i]);
        self.plays += stats.plays_by_lane[i].iter().map(|&v| v as u64).sum::<u64>();
        self.flips += stats.flips_by_rank[i].iter().map(|&v| v as u64).sum::<u64>();
        self.base_flips += stats.base_flips_by_rank[i].iter().map(|&v| v as u64).sum::<u64>();
        self.passes += stats.passes[i] as u64;
        self.pairs += stats.pairs_declared[i] as u64;
        self.unflipped_at_end += stats.unflipped_at_end[i] as u64;
        for r in 0..Rank::COUNT {
            self.plays_by_rank[r] += stats.plays_by_rank[i][r] as u64;
            self.flips_by_rank[r] += stats.flips_by_rank[i][r] as u64;
            self.base_flips_by_rank[r] += stats.base_flips_by_rank[i][r] as u64;
            self.flip_ply_sum[r] += stats.flip_ply_sum[i][r];
        }
        if let Some(v) = stats.lane_concentration(p, lanes_to_win) {
            self.lane_concentration.push(v);
        }
        if let Some(v) = stats.attack_concentration(p, lanes_to_win) {
            self.attack_concentration.push(v);
        }
    }

    /// Merge another shard's counts. Used to combine per-thread results.
    pub fn merge(&mut self, other: &AgentBehaviour) {
        self.games += other.games;
        self.hand_at_unlock.extend_from_slice(&other.hand_at_unlock);
        self.hand_at_unlock_won
            .extend_from_slice(&other.hand_at_unlock_won);
        self.hand_at_unlock_lost
            .extend_from_slice(&other.hand_at_unlock_lost);
        self.hand_at_end.extend_from_slice(&other.hand_at_end);
        self.plays += other.plays;
        self.flips += other.flips;
        self.base_flips += other.base_flips;
        self.passes += other.passes;
        self.pairs += other.pairs;
        self.unflipped_at_end += other.unflipped_at_end;
        for r in 0..Rank::COUNT {
            self.plays_by_rank[r] += other.plays_by_rank[r];
            self.flips_by_rank[r] += other.flips_by_rank[r];
            self.base_flips_by_rank[r] += other.base_flips_by_rank[r];
            self.flip_ply_sum[r] += other.flip_ply_sum[r];
        }
        self.lane_concentration
            .extend_from_slice(&other.lane_concentration);
        self.attack_concentration
            .extend_from_slice(&other.attack_concentration);
    }

    pub fn mean_hand_at_unlock(&self) -> f64 {
        mean_u32(&self.hand_at_unlock)
    }

    /// Mean hand size at the unlock in games this agent went on to win, and in games it went
    /// on to lose. The H2 test — see [`AgentBehaviour::hand_at_unlock_won`].
    pub fn hand_at_unlock_by_result(&self) -> (f64, f64) {
        (
            mean_u32(&self.hand_at_unlock_won),
            mean_u32(&self.hand_at_unlock_lost),
        )
    }

    /// Half-width of a 95% interval on the winner-minus-loser difference in hand size at the
    /// unlock. Two independent means, so the variances add.
    pub fn hand_at_unlock_gap_ci95(&self) -> f64 {
        let se = |v: &Vec<u32>| {
            if v.len() < 2 {
                return f64::INFINITY;
            }
            let m = mean_u32(v);
            let var = v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
            var / v.len() as f64
        };
        1.96 * (se(&self.hand_at_unlock_won) + se(&self.hand_at_unlock_lost)).sqrt()
    }
    pub fn mean_lane_concentration(&self) -> f64 {
        mean_f64(&self.lane_concentration)
    }
    pub fn mean_attack_concentration(&self) -> f64 {
        mean_f64(&self.attack_concentration)
    }
    pub fn plays_per_game(&self) -> f64 {
        self.plays as f64 / self.games.max(1) as f64
    }
    pub fn passes_per_game(&self) -> f64 {
        self.passes as f64 / self.games.max(1) as f64
    }

    /// Fraction of cards this agent played from hand that it ever turned face-up.
    ///
    /// The H4 measurement. Base-card flips are subtracted out: a base card was never played
    /// from hand, and counting it would push the ratio above 1 in an endgame where the base
    /// cards come up.
    pub fn flip_rate(&self) -> f64 {
        if self.plays == 0 {
            return f64::NAN;
        }
        self.flips.saturating_sub(self.base_flips) as f64 / self.plays as f64
    }

    /// Fraction of the cards of one rank this agent played from hand that it then turned
    /// face-up. `None` if it never played that rank.
    ///
    /// Base-card flips are subtracted from the numerator, so the ratio is exactly "of the
    /// `rank`s I chose to play, how many did I choose to flip". The sharper form of the H4
    /// question than mean flip ply: *whether* a rank is ever turned up, not just when.
    ///
    /// One rank needs a caveat, and it is the interesting one. A face-down 3 that is killed
    /// returns **face-up** through its Trap (`game_rules.md` §6), and that is not a `Flip`
    /// action — so it is counted nowhere here. This ratio therefore measures *voluntary*
    /// flips of a 3, which is the right quantity for the question but not the same as "how
    /// often a 3 ends up face-up".
    pub fn flip_rate_for(&self, rank: Rank) -> Option<f64> {
        let r = rank.index();
        if self.plays_by_rank[r] == 0 {
            return None;
        }
        let voluntary = self.flips_by_rank[r].saturating_sub(self.base_flips_by_rank[r]);
        Some(voluntary as f64 / self.plays_by_rank[r] as f64)
    }

    /// Mean ply at which this agent turns a given rank face-up. `None` if it never did.
    pub fn mean_flip_ply(&self, rank: Rank) -> Option<f64> {
        let n = self.flips_by_rank[rank.index()];
        if n == 0 {
            None
        } else {
            Some(self.flip_ply_sum[rank.index()] as f64 / n as f64)
        }
    }
}

pub(crate) fn mean_u32(values: &[u32]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64
}

pub(crate) fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Everything a set of games between two agents produced.
#[derive(Clone, Debug)]
pub struct MatchStats {
    pub config: GameConfig,
    pub agents: [AgentSpec; 2],
    pub games: usize,

    /// Wins by agent index, then draws.
    pub wins: [usize; 2],
    pub draws: usize,
    pub draws_stalemate: usize,
    pub draws_mutual_lane_win: usize,
    pub draws_ply_limit: usize,

    /// Score for the agent that sat as P0, over the games in which it did.
    pub p0_seat_score: f64,
    /// Sum of squared per-game P0 scores, for the interval on [`MatchStats::first_player_score`].
    pub p0_seat_score_sq: f64,
    pub p0_seat_games: usize,

    pub lengths: Vec<u32>,
    pub unlock_plies: Vec<u32>,
    pub max_side_occupancy: usize,
    pub behaviour: [AgentBehaviour; 2],
    pub elapsed_secs: f64,
}

impl MatchStats {
    pub(crate) fn empty(config: GameConfig, agents: [AgentSpec; 2]) -> MatchStats {
        MatchStats {
            config,
            agents,
            games: 0,
            wins: [0, 0],
            draws: 0,
            draws_stalemate: 0,
            draws_mutual_lane_win: 0,
            draws_ply_limit: 0,
            p0_seat_score: 0.0,
            p0_seat_score_sq: 0.0,
            p0_seat_games: 0,
            lengths: Vec::new(),
            unlock_plies: Vec::new(),
            max_side_occupancy: 0,
            behaviour: [AgentBehaviour::default(), AgentBehaviour::default()],
            elapsed_secs: 0.0,
        }
    }

    /// Fold in one finished game. `seats[i]` is the agent index that played as `Player::i`.
    pub(crate) fn absorb(&mut self, stats: &GameStats, seats: [usize; 2]) {
        self.games += 1;
        match stats.outcome {
            Outcome::Win(p) => self.wins[seats[p.idx()]] += 1,
            Outcome::Draw(reason) => {
                self.draws += 1;
                match reason {
                    DrawReason::Stalemate => self.draws_stalemate += 1,
                    DrawReason::MutualLaneWin => self.draws_mutual_lane_win += 1,
                    DrawReason::PlyLimit => self.draws_ply_limit += 1,
                }
            }
            Outcome::Ongoing => unreachable!("an unfinished game reached the aggregator"),
        }

        self.p0_seat_games += 1;
        let p0_score = stats.outcome.value_for(Player::P0) as f64;
        self.p0_seat_score += p0_score;
        self.p0_seat_score_sq += p0_score * p0_score;

        self.lengths.push(stats.plies);
        if let Some(ply) = stats.ply_at_unlock {
            self.unlock_plies.push(ply);
        }
        self.max_side_occupancy = self.max_side_occupancy.max(stats.max_side_occupancy);
        for p in Player::BOTH {
            self.behaviour[seats[p.idx()]].absorb(stats, p, self.config.lanes_to_win);
        }
    }

    /// Merge another shard. Used to combine per-thread results.
    pub(crate) fn merge(&mut self, other: &MatchStats) {
        self.games += other.games;
        self.wins[0] += other.wins[0];
        self.wins[1] += other.wins[1];
        self.draws += other.draws;
        self.draws_stalemate += other.draws_stalemate;
        self.draws_mutual_lane_win += other.draws_mutual_lane_win;
        self.draws_ply_limit += other.draws_ply_limit;
        self.p0_seat_score += other.p0_seat_score;
        self.p0_seat_score_sq += other.p0_seat_score_sq;
        self.p0_seat_games += other.p0_seat_games;
        self.lengths.extend_from_slice(&other.lengths);
        self.unlock_plies.extend_from_slice(&other.unlock_plies);
        self.max_side_occupancy = self.max_side_occupancy.max(other.max_side_occupancy);
        self.behaviour[0].merge(&other.behaviour[0]);
        self.behaviour[1].merge(&other.behaviour[1]);
    }

    /// Score for agent 0: 1 per win, 0.5 per draw.
    pub fn score(&self) -> f64 {
        if self.games == 0 {
            return 0.5;
        }
        (self.wins[0] as f64 + 0.5 * self.draws as f64) / self.games as f64
    }

    /// Half-width of a 95% confidence interval on [`MatchStats::score`].
    ///
    /// Draws contribute 0.25 to a single game's variance rather than 0, so counting them as
    /// half-wins without adjusting the variance would understate the interval.
    pub fn score_ci95(&self) -> f64 {
        if self.games < 2 {
            return f64::NAN;
        }
        let n = self.games as f64;
        let mean = self.score();
        let sum_sq = self.wins[0] as f64 + 0.25 * self.draws as f64;
        1.96 * ((sum_sq / n - mean * mean).max(0.0) / n).sqrt()
    }

    /// Score of whoever sat first, pooled over both colour assignments — the first-player
    /// advantage at this level of play (`FINDINGS.md` H8).
    pub fn first_player_score(&self) -> f64 {
        if self.p0_seat_games == 0 {
            return 0.5;
        }
        self.p0_seat_score / self.p0_seat_games as f64
    }

    /// Half-width of a 95% interval on [`MatchStats::first_player_score`].
    ///
    /// **Deliberately conservative.** Colour-paired deals mean each deal is played twice
    /// with the seats swapped, which cancels deal luck out of the *agent* score but not out
    /// of the first-player score: both games of a pair deal P0 the identical cards. So the
    /// effective sample is the number of distinct deals, half the game count, and the
    /// interval is computed on that. Treating 400 paired games as 400 independent
    /// observations would understate the interval by about √2 — enough to turn a
    /// null result into a spurious one at these effect sizes.
    pub fn first_player_score_ci95(&self) -> f64 {
        let deals = self.p0_seat_games / 2;
        if deals < 2 {
            return f64::NAN;
        }
        let n = self.p0_seat_games as f64;
        let mean = self.first_player_score();
        let variance = (self.p0_seat_score_sq / n - mean * mean).max(0.0);
        1.96 * (variance / deals as f64).sqrt()
    }

    pub fn draw_rate(&self) -> f64 {
        self.draws as f64 / self.games.max(1) as f64
    }

    /// How often the quiet-ply rule ended the game. `FINDINGS.md` F1.2 asks for exactly this
    /// number against non-random agents, because random play could never produce it.
    pub fn stalemate_rate(&self) -> f64 {
        self.draws_stalemate as f64 / self.games.max(1) as f64
    }

    pub fn mean_plies(&self) -> f64 {
        mean_u32(&self.lengths)
    }

    pub fn games_per_sec(&self) -> f64 {
        if self.elapsed_secs <= 0.0 {
            f64::INFINITY
        } else {
            self.games as f64 / self.elapsed_secs
        }
    }

    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{} vs {} — {} games\n",
            self.agents[0], self.agents[1], self.games
        ));
        out.push_str(&format!("  config: {}\n", self.config.summary()));
        out.push_str(&format!(
            "  score for {}: {:.4} +/- {:.4} (95% CI) — W{} L{} D{}\n",
            self.agents[0],
            self.score(),
            self.score_ci95(),
            self.wins[0],
            self.wins[1],
            self.draws,
        ));
        out.push_str(&format!(
            "  first-player score: {:.4} +/- {:.4}  ·  draws {:.1}% (stalemate {:.1}%, mutual lane win {}, ply cap {})\n",
            self.first_player_score(),
            self.first_player_score_ci95(),
            100.0 * self.draw_rate(),
            100.0 * self.stalemate_rate(),
            self.draws_mutual_lane_win,
            self.draws_ply_limit,
        ));
        out.push_str(&format!(
            "  mean plies {:.1} · max cards on one side of one lane {} · {:.1} games/sec\n",
            self.mean_plies(),
            self.max_side_occupancy,
            self.games_per_sec(),
        ));
        for i in 0..2 {
            let b = &self.behaviour[i];
            let (won, lost) = b.hand_at_unlock_by_result();
            out.push_str(&format!(
                "  {:<14} hand@unlock {:.2} (won {:.2} vs lost {:.2}, gap {:+.2} +/- {:.2}) · \
                 plays/game {:.1} · flip rate {:.2} · lane conc {:.3} · attack conc {:.3} · \
                 passes/game {:.2}\n",
                self.agents[i].name(),
                b.mean_hand_at_unlock(),
                won,
                lost,
                won - lost,
                b.hand_at_unlock_gap_ci95(),
                b.plays_per_game(),
                b.flip_rate(),
                b.mean_lane_concentration(),
                b.mean_attack_concentration(),
                b.passes_per_game(),
            ));
        }
        out
    }
}
