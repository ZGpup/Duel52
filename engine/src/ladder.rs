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
//!
//! # Batched evaluation
//!
//! `--eval-batch N` keeps `N` games in flight per worker so their network evaluations go
//! through the trunk together (`PLAN.md` §4.2d). It is a speed knob and nothing else: the
//! batch is taken across games and never inside a search, so
//! `rule_2_the_ladder_is_eval_batch_independent` holds the result fixed for the same reason
//! the thread-count test does. This matters more here than in self-play, because the gate is
//! the measurement the training loop *promotes* on — a batched gate that scored differently
//! would quietly change which candidates ship.
//!
//! Only `netmcts` batches. Every other agent decides inline, so a panel row against `random`
//! or `greedy` has one network in it rather than two.

use std::time::Instant;

use std::sync::Arc;

use crate::agents::{AgentSpec, SearchStep};
use crate::config::GameConfig;
use crate::elo::{fit, EloTable, Pairing};
use crate::encode::{action_dim, obs_dim};
use crate::nn::BatchScratch;
use crate::probe::{play_instrumented, MatchGame, MatchStats, AGENT_STREAM};

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
    eval_batch: usize,
) -> MatchStats {
    let started = Instant::now();
    let games = games + (games % 2);
    let threads = threads.max(1).min(games.max(1));
    let eval_batch = eval_batch.max(1);

    let mut total = MatchStats::empty(config, [a.clone(), b.clone()]);
    if games == 0 {
        return total;
    }
    // `AgentSpec` stopped being `Copy` when `NetPolicy` gained a checkpoint path, so the
    // pair is borrowed into the workers rather than copied into them. Cloning per shard
    // would be harmless too — this is once per thread, not once per game.
    let specs = [a, b];

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
                let specs = &specs;
                scope.spawn(move || {
                    if eval_batch > 1 {
                        return play_shard_batched(
                            config, specs, first_seed, lo, hi, eval_batch,
                        );
                    }
                    let mut shard = MatchStats::empty(config, specs.clone());
                    for g in lo..hi {
                        shard.absorb(&play_indexed(config, specs, first_seed, g), seats(g));
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

/// Build both agents for one game, seated.
///
/// Stream tags follow the *agent*, not the seat, so an agent consumes the same random
/// numbers in both halves of a colour-paired deal.
fn build_seated(
    agents: &[AgentSpec; 2],
    seed: u64,
    seats: [usize; 2],
) -> (Box<dyn crate::agents::Agent>, Box<dyn crate::agents::Agent>) {
    (
        agents[seats[0]].build(seed, AGENT_STREAM[seats[0]]),
        agents[seats[1]].build(seed, AGENT_STREAM[seats[1]]),
    )
}

/// Play games `lo..hi` of a match with up to `batch` of them in flight at once.
///
/// The gate and the reference panel are 47% of a `train-3h-new` generation and were the half
/// `PLAN.md` §4.2d did not reach at first. Same idea as `selfplay::play_shard_batched` and
/// the same guarantee — the batch is taken across games, no search is altered, and
/// `rule_2_the_ladder_is_eval_batch_independent` asserts the result does not move.
///
/// **One thing is different here: a match has two agents with two different checkpoints.**
/// The candidate and the incumbent are separate networks, so a round's suspended games are
/// grouped by the evaluator they are waiting on and each group is evaluated on its own. That
/// halves the batch a gate can reach relative to self-play — the games in flight split
/// roughly evenly between the two sides — which is why the gate's speed-up is nearer 2x than
/// self-play's 3.26x. A panel row against `random` or `greedy` has only one network and does
/// not pay that, because the non-network agent never suspends at all.
fn play_shard_batched(
    config: GameConfig,
    specs: &[AgentSpec; 2],
    first_seed: u64,
    lo: usize,
    hi: usize,
    batch: usize,
) -> MatchStats {
    let slots = crate::nn::batch_slots(hi - lo, batch);
    let (od, ad) = (obs_dim(&config), action_dim(&config));

    // Per slot, because a slot's observation is written when it suspends and its mask must
    // still be the buffer `supply` clears.
    let mut obs = vec![0.0f32; slots * od];
    let mut masks = vec![false; slots * ad];
    // Contiguous per evaluator, because `eval_masked_batch` takes rows in one block and a
    // group's slots are scattered. Fully overwritten each time, so nothing goes stale.
    let mut stage_obs = vec![0.0f32; slots * od];
    let mut stage_masks = vec![false; slots * ad];
    let mut logits = vec![0.0f32; slots * ad];
    let mut values = vec![0.0f32; slots];

    let mut games: Vec<Option<(usize, MatchGame)>> = (0..slots).map(|_| None).collect();
    let mut pending: Vec<(usize, usize)> = Vec::with_capacity(slots);
    let mut keys: Vec<usize> = Vec::with_capacity(2);
    let mut rows: Vec<usize> = Vec::with_capacity(slots);
    let mut scratches: Vec<(usize, BatchScratch)> = Vec::new();
    let mut next = lo;
    // Collected rather than absorbed as they finish, then absorbed in game order.
    //
    // ⚠️ Not fussiness. `AgentBehaviour::absorb` *pushes* each game's lane and attack
    // concentration into a `Vec<f64>` whose mean `probe` later reports, and a mean over f64
    // depends on summation order. Absorbing games as they complete would leave that mean
    // differing in its last bits between `--eval-batch 1` and anything else — a difference
    // small enough to look like nothing and large enough to make the probe tables
    // irreproducible. The gate itself reads only integers and would not have noticed.
    let mut finished: Vec<(usize, crate::probe::GameStats)> = Vec::with_capacity(hi - lo);

    loop {
        pending.clear();
        for slot in 0..slots {
            if games[slot].is_none() && next < hi {
                let g = next;
                next += 1;
                let seed = first_seed + (g / 2) as u64;
                let (first, second) = build_seated(specs, seed, seats(g));
                games[slot] = Some((g, MatchGame::new(config, seed, first, second)));
            }
            let Some((g, game)) = games[slot].as_mut() else {
                continue;
            };
            match game.advance(
                &mut obs[slot * od..(slot + 1) * od],
                &mut masks[slot * ad..(slot + 1) * ad],
            ) {
                SearchStep::NeedsEval => {
                    let key = Arc::as_ptr(
                        game.pending_evaluator()
                            .expect("a suspended search names its network"),
                    ) as usize;
                    pending.push((slot, key));
                }
                SearchStep::Done => {
                    let g = *g;
                    let (_, game) = games[slot].take().expect("live a line ago");
                    finished.push((g, game.finish()));
                }
            }
        }

        if pending.is_empty() {
            // Every live game contributes a row, so no rows means no live games.
            debug_assert!(games.iter().all(|g| g.is_none()));
            if next >= hi {
                break;
            }
            continue;
        }

        keys.clear();
        for &(_, key) in &pending {
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        for &key in &keys {
            rows.clear();
            rows.extend(pending.iter().filter(|(_, k)| *k == key).map(|(s, _)| *s));
            for (row, &slot) in rows.iter().enumerate() {
                stage_obs[row * od..(row + 1) * od]
                    .copy_from_slice(&obs[slot * od..(slot + 1) * od]);
                stage_masks[row * ad..(row + 1) * ad]
                    .copy_from_slice(&masks[slot * ad..(slot + 1) * ad]);
            }
            let evaluator = games[rows[0]]
                .as_ref()
                .expect("a pending slot is live")
                .1
                .pending_evaluator()
                .expect("a pending slot has a suspended search")
                .clone();
            let at = match scratches.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    scratches.push((key, evaluator.batch_scratch(slots)));
                    scratches.len() - 1
                }
            };
            let n = rows.len();
            evaluator.eval_masked_batch(
                &stage_obs[..n * od],
                n,
                &stage_masks[..n * ad],
                &mut logits[..n * ad],
                &mut values[..n],
                &mut scratches[at].1,
            );
            for (row, &slot) in rows.iter().enumerate() {
                games[slot]
                    .as_mut()
                    .expect("a pending slot is live")
                    .1
                    .supply(
                        &logits[row * ad..(row + 1) * ad],
                        values[row],
                        &mut masks[slot * ad..(slot + 1) * ad],
                    );
            }
        }
    }

    finished.sort_by_key(|(g, _)| *g);
    let mut shard = MatchStats::empty(config, specs.clone());
    for (g, stats) in &finished {
        shard.absorb(stats, seats(*g));
    }
    shard
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
    agents: &[AgentSpec; 2],
    first_seed: u64,
    g: usize,
) -> crate::probe::GameStats {
    let seed = first_seed + (g / 2) as u64;
    let seats = seats(g);

    // Stream tags follow the *agent*, not the seat, so an agent consumes the same random
    // numbers in both halves of a colour-paired deal.
    let (first, second) = build_seated(agents, seed, seats);
    play_instrumented(config, seed, first, second)
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
    eval_batch: usize,
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
                roster[i].clone(),
                roster[j].clone(),
                first_seed,
                games_per_pairing,
                threads,
                eval_batch,
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
