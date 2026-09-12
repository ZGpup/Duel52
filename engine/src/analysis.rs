//! The analysis corpus — one row per player-game, one row per card.
//!
//! # Why this is a corpus and not a report
//!
//! Every other measurement command in this engine decides what it is measuring before the
//! games are played: `stats` counts outcomes, `probe` fills a fixed set of per-rank tables,
//! `ladder` fits a rating. Ask a question none of them anticipated and the answer is a code
//! change and a re-run — and at a thousand simulations a re-run is hours.
//!
//! So this command does not compute statistics. It plays an agent against **itself** and
//! writes down what happened, flat:
//!
//! | file | one row per | answers |
//! |---|---|---|
//! | `games.csv` | player-game, so two per game | first-player advantage, hand at unlock, win rate given a card in hand, pairs per game |
//! | `cards.csv` | card that entered play | when cards are played and flipped, how long they stay hidden, how they die |
//! | `meta.json` | run | which ruleset, which agent, which seeds — everything needed to say whether two corpora may be compared |
//!
//! Every statistic `py/duel52/analysis` reports is a fold over those two tables, so a new
//! question costs a function rather than a run. That is the whole design.
//!
//! # Self-play, deliberately
//!
//! The agent plays itself, so "the mean turn a 7 is flipped on" is a property of *that
//! agent* rather than of a pairing. Two agents' corpora are then comparable column by column
//! because each one describes how that agent plays, not how it plays against some third
//! thing. [`extract`] takes one [`AgentSpec`] and seats it on both sides; there is no way to
//! ask for anything else.
//!
//! Deals are still colour-paired the way [`crate::ladder`] pairs them — every deal is played
//! twice with the seats swapped — so the first-player number is measured against the same
//! deal from both sides, and `games` is rounded up to an even number.
//!
//! # Missing is empty
//!
//! A field that does not apply is written as the empty string rather than as `0` or `-1`: a
//! game that ended before the unlock has no hand size at the unlock, and a card that was
//! never flipped has no flip ply. Writing a sentinel there is how a mean ends up including
//! twelve thousand zeros that never happened.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::agents::AgentSpec;
use crate::config::GameConfig;
use crate::ladder::{run_games, GameSink};
use crate::outcome::{DrawReason, Outcome};
use crate::player::Player;
use crate::probe::GameStats;
use crate::rank::Rank;

/// Bumped when a column changes meaning or disappears. The reader refuses a corpus it does
/// not know, because a silently-shifted column is a wrong table rather than an error.
pub const SCHEMA: u32 = 1;

/// Outcome tallies over a run, for `meta.json`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub games: usize,
    pub wins: [usize; 2],
    pub draws: usize,
    pub draws_stalemate: usize,
    pub draws_mutual_lane_win: usize,
    pub draws_ply_limit: usize,
    pub card_rows: usize,
}

impl Counts {
    fn absorb(&mut self, stats: &GameStats) {
        self.games += 1;
        self.card_rows += stats.cards.len();
        match stats.outcome {
            Outcome::Win(p) => self.wins[p.idx()] += 1,
            Outcome::Draw(reason) => {
                self.draws += 1;
                match reason {
                    DrawReason::Stalemate => self.draws_stalemate += 1,
                    DrawReason::MutualLaneWin => self.draws_mutual_lane_win += 1,
                    DrawReason::PlyLimit => self.draws_ply_limit += 1,
                }
            }
            Outcome::Ongoing => {}
        }
    }

    fn merge(&mut self, other: &Counts) {
        self.games += other.games;
        self.wins[0] += other.wins[0];
        self.wins[1] += other.wins[1];
        self.draws += other.draws;
        self.draws_stalemate += other.draws_stalemate;
        self.draws_mutual_lane_win += other.draws_mutual_lane_win;
        self.draws_ply_limit += other.draws_ply_limit;
        self.card_rows += other.card_rows;
    }
}

/// One worker's slice of the corpus, streamed to its own pair of part files.
///
/// Streamed rather than collected because a 20,000-game corpus is 800,000 card rows: holding
/// them as `String`s to concatenate at the end would cost more memory than the searches that
/// produced them.
pub struct Corpus {
    games: BufWriter<File>,
    cards: BufWriter<File>,
    counts: Counts,
    ranks: usize,
    lanes_to_win: usize,
    row: String,
}

impl Corpus {
    fn new(dir: &Path, worker: usize, config: &GameConfig) -> Result<Corpus, String> {
        Ok(Corpus {
            games: part(dir, worker, "games")?,
            cards: part(dir, worker, "cards")?,
            counts: Counts::default(),
            ranks: config.rank_count(),
            lanes_to_win: config.lanes_to_win,
            row: String::with_capacity(512),
        })
    }
}

fn part(dir: &Path, worker: usize, name: &str) -> Result<BufWriter<File>, String> {
    let path = dir.join(format!(".part-{worker:03}.{name}"));
    File::create(&path)
        .map(|f| BufWriter::with_capacity(1 << 16, f))
        .map_err(|e| format!("creating {}: {e}", path.display()))
}

impl GameSink for Corpus {
    fn take_game(&mut self, index: usize, stats: &GameStats, _seats: [usize; 2]) {
        self.counts.absorb(stats);
        // The buffer is a field so it is allocated once per worker rather than once per
        // game, and taken out of `self` so the writes below can borrow `self` too.
        let mut me = std::mem::take(&mut self.row);
        me.clear();
        for p in Player::BOTH {
            let i = p.idx();
            let me = &mut me;
            push_num(me, index);
            push_num(me, stats.seed);
            push_num(me, i);
            me.push_str(match stats.outcome {
                Outcome::Win(w) if w == p => "win",
                Outcome::Win(_) => "loss",
                _ => "draw",
            });
            me.push(',');
            me.push_str(match stats.outcome {
                Outcome::Draw(DrawReason::Stalemate) => "stalemate",
                Outcome::Draw(DrawReason::MutualLaneWin) => "mutual_lane_win",
                Outcome::Draw(DrawReason::PlyLimit) => "ply_limit",
                _ => "",
            });
            me.push(',');
            push_num(me, stats.plies);
            push_num(me, stats.decisions);
            push_opt(me, stats.ply_at_unlock);
            // Hand sizes at the unlock exist only if the unlock happened. `hand_at_unlock`
            // is zeroed otherwise, and a zero there is a real hand size, so it has to be
            // gated on the ply rather than tested for zero.
            let unlocked = stats.ply_at_unlock.is_some();
            push_opt(me, unlocked.then_some(stats.hand_at_unlock[i]));
            push_opt(me, unlocked.then_some(stats.hand_at_unlock[1 - i]));
            push_num(me, stats.hand_at_end[i]);
            push_num(me, stats.draws_taken[i]);
            push_num(me, stats.stuck_turns[i]);
            push_num(me, stats.plays_by_lane[i].iter().sum::<u32>());
            push_num(me, stats.attacks_by_lane[i].iter().sum::<u32>());
            push_num(me, stats.pairs_declared[i]);
            push_float(me, stats.lane_concentration(p, self.lanes_to_win));
            push_float(me, stats.attack_concentration(p, self.lanes_to_win));
            for r in 0..self.ranks {
                push_num(me, stats.start_hand[i][r]);
            }
            for r in 0..self.ranks {
                push_opt(me, unlocked.then_some(stats.unlock_hand[i][r]));
            }
            for r in 0..self.ranks {
                push_num(me, stats.pairs_by_rank[i][r]);
            }
            me.pop(); // the trailing comma `push_*` leaves
            me.push('\n');
        }
        let _ = self.games.write_all(me.as_bytes());
        me.clear();

        for card in &stats.cards {
            let me = &mut me;
            push_num(me, index);
            push_num(me, card.owner.idx());
            push_num(me, card.rank.index());
            push_num(me, u8::from(card.entered_as_base));
            push_num(me, card.entered_ply);
            push_opt(me, card.face_up_ply);
            me.push_str(card.face_up_kind.label());
            me.push(',');
            push_opt(me, card.death_ply);
            // Only meaningful with a death ply, so it is blank without one rather than a
            // `0` that reads as "died face-down".
            push_opt(me, card.death_ply.map(|_| u8::from(card.died_face_up)));
            push_num(me, u8::from(card.ever_paired));
            me.pop();
            me.push('\n');
        }
        let _ = self.cards.write_all(me.as_bytes());
        me.clear();
        self.row = me;
    }
}

fn push_num<T: std::fmt::Display>(out: &mut String, v: T) {
    use std::fmt::Write;
    let _ = write!(out, "{v},");
}

fn push_opt<T: std::fmt::Display>(out: &mut String, v: Option<T>) {
    match v {
        Some(v) => push_num(out, v),
        None => out.push(','),
    }
}

fn push_float(out: &mut String, v: Option<f64>) {
    use std::fmt::Write;
    match v {
        // Enough digits to round-trip an f64 exactly, so the analysis script's mean matches
        // the one `probe` prints rather than nearly matching it.
        Some(v) => {
            let _ = write!(out, "{v:.17},");
        }
        None => out.push(','),
    }
}

/// What one extraction produced.
pub struct Run {
    pub out_dir: PathBuf,
    pub counts: Counts,
    pub elapsed_secs: f64,
}

impl Run {
    pub fn games_per_sec(&self) -> f64 {
        if self.elapsed_secs <= 0.0 {
            f64::INFINITY
        } else {
            self.counts.games as f64 / self.elapsed_secs
        }
    }
}

/// Play `games` self-play games with `spec` on both sides and write the corpus to `out_dir`.
pub fn extract(
    config: GameConfig,
    spec: AgentSpec,
    first_seed: u64,
    games: usize,
    threads: usize,
    eval_batch: usize,
    out_dir: &Path,
) -> Result<Run, String> {
    fs::create_dir_all(out_dir).map_err(|e| format!("creating {}: {e}", out_dir.display()))?;
    let started = Instant::now();
    let specs = [spec.clone(), spec.clone()];

    // Every part file is opened **before** a game is played. `GameSink::take_game` has
    // nowhere to put an error, and discovering a full disk four hours in — after the
    // expensive part — is the failure worth designing against. `worker_count` is the same
    // clamp `run_games` applies, called rather than re-derived.
    let workers = crate::ladder::worker_count(games, threads);
    let mut prepared = Vec::with_capacity(workers);
    for w in 0..workers {
        prepared.push(Some(Corpus::new(out_dir, w, &config)?));
    }
    let prepared = std::sync::Mutex::new(prepared);

    let sinks = run_games(config, &specs, first_seed, games, threads, eval_batch, |w| {
        prepared
            .lock()
            .expect("the sink pool")[w]
            .take()
            .expect("run_games asks for each worker's sink exactly once")
    });
    let mut counts = Counts::default();
    for sink in &sinks {
        counts.merge(&sink.counts);
    }
    // Flush before the parts are read back.
    for mut sink in sinks {
        sink.games
            .flush()
            .map_err(|e| format!("writing the games corpus: {e}"))?;
        sink.cards
            .flush()
            .map_err(|e| format!("writing the cards corpus: {e}"))?;
    }

    let elapsed = started.elapsed().as_secs_f64();
    join_parts(out_dir, workers, "games", &games_header(&config))?;
    join_parts(out_dir, workers, "cards", CARDS_HEADER)?;

    let run = Run {
        out_dir: out_dir.to_path_buf(),
        counts,
        elapsed_secs: elapsed,
    };
    write_meta(&run, &config, &spec, first_seed, threads, eval_batch)?;
    Ok(run)
}

/// Concatenate the workers' part files, in worker order, under one header line.
///
/// Worker order is game order: [`run_games`] hands worker `w` a contiguous seed range and
/// each worker's sink sees its own games in order, so the whole file comes out sorted by
/// game index without a sort.
fn join_parts(dir: &Path, workers: usize, name: &str, header: &str) -> Result<(), String> {
    let path = dir.join(format!("{name}.csv"));
    let file =
        File::create(&path).map_err(|e| format!("creating {}: {e}", path.display()))?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    out.write_all(header.as_bytes())
        .and_then(|_| out.write_all(b"\n"))
        .map_err(|e| format!("writing {}: {e}", path.display()))?;
    let mut buf = vec![0u8; 1 << 20];
    for w in 0..workers {
        let part = dir.join(format!(".part-{w:03}.{name}"));
        let mut src =
            File::open(&part).map_err(|e| format!("reading {}: {e}", part.display()))?;
        loop {
            let n = src
                .read(&mut buf)
                .map_err(|e| format!("reading {}: {e}", part.display()))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])
                .map_err(|e| format!("writing {}: {e}", path.display()))?;
        }
        drop(src);
        fs::remove_file(&part).map_err(|e| format!("removing {}: {e}", part.display()))?;
    }
    out.flush()
        .map_err(|e| format!("writing {}: {e}", path.display()))
}

/// The `cards.csv` header. Fixed, because none of its columns are per-rank.
pub const CARDS_HEADER: &str =
    "game,owner,rank,base,enter_ply,faceup_ply,faceup_kind,death_ply,died_face_up,paired";

/// The `games.csv` header. Three blocks of it are per-rank, so it is built from the config —
/// a ruleset with fewer ranks gets fewer columns rather than columns of zeros.
pub fn games_header(config: &GameConfig) -> String {
    let mut head = String::from(
        "game,seed,seat,result,draw_reason,plies,decisions,unlock_ply,hand_at_unlock,\
         opp_hand_at_unlock,hand_at_end,draws_taken,stuck_turns,plays,attacks,pairs,\
         lane_conc,attack_conc",
    );
    for prefix in ["start", "unlock", "pairs"] {
        for r in 0..config.rank_count() {
            head.push(',');
            head.push_str(prefix);
            head.push('_');
            head.push_str(Rank::ALL[r].label());
        }
    }
    head
}

/// Everything needed to decide whether two corpora may be compared, and to reproduce either.
fn write_meta(
    run: &Run,
    config: &GameConfig,
    spec: &AgentSpec,
    first_seed: u64,
    threads: usize,
    eval_batch: usize,
) -> Result<(), String> {
    let mut json = String::new();
    json.push_str("{\n");
    let mut field = |key: &str, value: String| {
        json.push_str("  \"");
        json.push_str(key);
        json.push_str("\": ");
        json.push_str(&value);
        json.push_str(",\n");
    };
    field("schema", SCHEMA.to_string());
    field("agent", quote(&spec.name()));
    field("games", run.counts.games.to_string());
    field("first_seed", first_seed.to_string());
    field("deals", (run.counts.games / 2).to_string());
    field("threads", threads.to_string());
    field("eval_batch", eval_batch.to_string());
    field("elapsed_secs", format!("{:.3}", run.elapsed_secs));
    field("games_per_sec", format!("{:.4}", run.games_per_sec()));
    field("variant", quote(&config.variant.to_string()));
    field("two_power", quote(&config.two_power.to_string()));
    field("rules_name", quote(&config.rules_name.to_string()));
    field("rules_hash", quote(&format!("{:016x}", config.rules_hash())));
    field("rules_label", quote(&config.rules_label()));
    field("config_summary", quote(&config.summary()));
    field("lanes", config.lanes.to_string());
    field("lanes_to_win", config.lanes_to_win.to_string());
    field("hand_size", config.hand_size.to_string());
    field("copies_per_rank", config.copies_per_rank.to_string());
    field("stalemate_quiet_plies", config.stalemate_quiet_plies.to_string());
    field("wins_p0", run.counts.wins[0].to_string());
    field("wins_p1", run.counts.wins[1].to_string());
    field("draws", run.counts.draws.to_string());
    field("draws_stalemate", run.counts.draws_stalemate.to_string());
    field(
        "draws_mutual_lane_win",
        run.counts.draws_mutual_lane_win.to_string(),
    );
    field("draws_ply_limit", run.counts.draws_ply_limit.to_string());
    field("card_rows", run.counts.card_rows.to_string());
    // The ranks in play, in encoder order, with the power each one carries under *this*
    // ruleset — so a modded ruleset's table is labelled with the powers it actually has
    // rather than with the canonical ones.
    let ranks: Vec<String> = (0..config.rank_count())
        .map(|r| quote(Rank::ALL[r].label()))
        .collect();
    field("ranks", format!("[{}]", ranks.join(", ")));
    let powers: Vec<String> = (0..config.rank_count())
        .map(|r| quote(config.power(Rank::ALL[r]).display_name()))
        .collect();
    field("powers", format!("[{}]", powers.join(", ")));
    // No trailing comma on the last field.
    json.truncate(json.trim_end_matches([',', '\n']).len());
    json.push_str("\n}\n");

    let path = run.out_dir.join("meta.json");
    fs::write(&path, json).map_err(|e| format!("writing {}: {e}", path.display()))
}

/// A JSON string literal. The engine has no dependencies, so this is four lines rather than
/// a crate — the same trade `record.rs` makes.
fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
