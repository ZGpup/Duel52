//! The round-robin ladder.
//!
//! `PLAN.md` Phase 2: "Round-robin Elo ladder, frozen as the permanent benchmark."
//!
//! # Two variance controls, because the budget is small
//!
//! A search agent plays a few games a second, not sixteen thousand, so a Phase 2 ladder runs
//! on hundreds of games per pairing where Phase 1 ran on two hundred thousand. Two devices
//! recover most of the lost precision:
//!
//! - **Colour-paired deals.** Every deal is played **twice**, with the seats swapped, and
//!   both agents get the same random stream in both games. A deal that hands one side four
//!   Jacks then helps each agent exactly once, so deal luck cancels within the pair instead
//!   of averaging out slowly across the run. This is why `games` is always rounded up to an
//!   even number.
//! - **Shared seeds across pairings.** Every pairing starts from the same `first_seed`, so
//!   the whole table is computed on one set of deals. Differences between rungs are then
//!   differences in play rather than in what they were dealt.
//!
//! # Threading
//!
//! Games are independent, so the shards are seed ranges and nothing is shared but the
//! merge at the end. Each shard re-derives its agents from the game seed, so the result does
//! not depend on the thread count — `rule_2_the_ladder_is_thread_count_independent` pins
//! that, because a benchmark whose numbers move when you change `--threads` is not a
//! benchmark.

use std::time::Instant;

use crate::agents::AgentSpec;
use crate::config::GameConfig;
use crate::elo::{fit, EloTable, Pairing};
use crate::probe::{play_instrumented, MatchStats, AGENT_STREAM};

/// Play `games` games between two agents, alternating who moves first.
///
/// `games` is rounded up to an even number so every deal is played from both sides.
/// `threads` of 0 or 1 runs single-threaded.
pub fn run_match(
    config: GameConfig,
    a: AgentSpec,
    b: AgentSpec,
    first_seed: u64,
    games: usize,
    threads: usize,
) -> MatchStats {
    let started = Instant::now();
    let games = games + (games % 2);
    let threads = threads.max(1).min(games.max(1));

    let mut total = MatchStats::empty(config, [a, b]);
    if games == 0 {
        return total;
    }

    let shards: Vec<(usize, usize)> = (0..threads)
        .map(|t| {
            let lo = games * t / threads;
            let hi = games * (t + 1) / threads;
            (lo, hi)
        })
        .filter(|(lo, hi)| lo < hi)
        .collect();

    let results: Vec<MatchStats> = std::thread::scope(|scope| {
        let handles: Vec<_> = shards
            .iter()
            .map(|&(lo, hi)| {
                scope.spawn(move || {
                    let mut shard = MatchStats::empty(config, [a, b]);
                    for g in lo..hi {
                        shard.absorb(&play_indexed(config, [a, b], first_seed, g), seats(g));
                    }
                    shard
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a ladder worker panicked"))
            .collect()
    });

    for shard in &results {
        total.merge(shard);
    }
    total.elapsed_secs = started.elapsed().as_secs_f64();
    total
}

/// Which agent index sits in which seat for game `g`: even games put agent 0 first.
#[inline]
fn seats(g: usize) -> [usize; 2] {
    if g % 2 == 0 {
        [0, 1]
    } else {
        [1, 0]
    }
}

/// Play game number `g` of a match. The deal depends only on `g / 2`, so games `2k` and
/// `2k + 1` are the same deal with the seats swapped.
fn play_indexed(
    config: GameConfig,
    agents: [AgentSpec; 2],
    first_seed: u64,
    g: usize,
) -> crate::probe::GameStats {
    let seed = first_seed + (g / 2) as u64;
    let seats = seats(g);

    // Stream tags follow the *agent*, not the seat, so an agent consumes the same random
    // numbers in both halves of a colour-paired deal.
    let mut first = agents[seats[0]].build(seed, AGENT_STREAM[seats[0]]);
    let mut second = agents[seats[1]].build(seed, AGENT_STREAM[seats[1]]);
    play_instrumented(config, seed, first.as_mut(), second.as_mut())
}

/// A complete round-robin, plus the ratings fitted to it.
#[derive(Clone, Debug)]
pub struct LadderResult {
    pub config: GameConfig,
    pub roster: Vec<AgentSpec>,
    /// One entry per unordered pair, in `(0,1), (0,2), … (1,2), …` order.
    pub matches: Vec<MatchStats>,
    pub elo: EloTable,
    pub first_seed: u64,
    pub games_per_pairing: usize,
    pub elapsed_secs: f64,
}

impl LadderResult {
    /// The head-to-head record between two roster entries, if they met.
    pub fn head_to_head(&self, i: usize, j: usize) -> Option<&MatchStats> {
        self.matches
            .iter()
            .find(|m| m.agents[0] == self.roster[i] && m.agents[1] == self.roster[j])
    }

    /// The full cross-table, plus the fitted ratings.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Round-robin ladder — {} agents, {} games per pairing, seeds {}..{}\n",
            self.roster.len(),
            self.games_per_pairing,
            self.first_seed,
            self.first_seed + (self.games_per_pairing as u64 + 1) / 2 - 1,
        ));
        out.push_str(&format!("  config: {}\n", self.config.summary()));
        out.push_str(&format!(
            "  total: {} games in {:.1}s\n\n",
            self.matches.iter().map(|m| m.games).sum::<usize>(),
            self.elapsed_secs,
        ));

        out.push_str("Head to head (row's score against column)\n");
        out.push_str(&format!("  {:<16}", ""));
        for spec in &self.roster {
            out.push_str(&format!("{:>16}", spec.name()));
        }
        out.push('\n');
        for (i, row) in self.roster.iter().enumerate() {
            out.push_str(&format!("  {:<16}", row.name()));
            for j in 0..self.roster.len() {
                if i == j {
                    out.push_str(&format!("{:>16}", "—"));
                } else if let Some(m) = self.head_to_head(i, j) {
                    out.push_str(&format!("{:>16.3}", m.score()));
                } else if let Some(m) = self.head_to_head(j, i) {
                    out.push_str(&format!("{:>16.3}", 1.0 - m.score()));
                } else {
                    out.push_str(&format!("{:>16}", "·"));
                }
            }
            out.push('\n');
        }

        out.push_str("\nRatings\n");
        out.push_str(&format!("{}\n", self.elo));
        out
    }

    /// The Elo table as Markdown, for pasting into `FINDINGS.md`.
    pub fn markdown(&self) -> String {
        let mut out = self.elo.markdown();
        out.push_str(&format!(
            "\nConfig: `{}` · {} games per pairing · seeds from {} · engine {}\n",
            self.config.summary(),
            self.games_per_pairing,
            self.first_seed,
            crate::VERSION,
        ));
        out
    }
}

/// Run every pairing in `roster` against every other, and fit ratings to the result.
///
/// `anchor_name` is pinned to 0 Elo if it is in the roster — normally `random`, so the whole
/// table reads as "how far above uniform play". `progress` writes one line per pairing to
/// stderr, because a full ladder is minutes of silence otherwise.
pub fn run_ladder(
    config: GameConfig,
    roster: &[AgentSpec],
    first_seed: u64,
    games_per_pairing: usize,
    threads: usize,
    anchor_name: &str,
    progress: bool,
) -> LadderResult {
    let started = Instant::now();
    let mut matches = Vec::new();
    let mut pairings = Vec::new();

    for i in 0..roster.len() {
        for j in (i + 1)..roster.len() {
            if progress {
                eprintln!(
                    "  [{}/{}] {} vs {} …",
                    matches.len() + 1,
                    roster.len() * (roster.len() - 1) / 2,
                    roster[i].name(),
                    roster[j].name(),
                );
            }
            let m = run_match(
                config,
                roster[i],
                roster[j],
                first_seed,
                games_per_pairing,
                threads,
            );
            if progress {
                eprintln!(
                    "        {:.3} +/- {:.3} for {} ({:.1} games/sec)",
                    m.score(),
                    m.score_ci95(),
                    roster[i].name(),
                    m.games_per_sec(),
                );
            }
            pairings.push(Pairing::new(i, j, m.wins[0], m.wins[1], m.draws));
            matches.push(m);
        }
    }

    let names: Vec<String> = roster.iter().map(|s| s.name()).collect();
    let anchor = names
        .iter()
        .position(|n| n == anchor_name)
        .unwrap_or(0);
    let elo = fit(names, &pairings, anchor);

    LadderResult {
        config,
        roster: roster.to_vec(),
        matches,
        elo,
        first_seed,
        games_per_pairing,
        elapsed_secs: started.elapsed().as_secs_f64(),
    }
}
