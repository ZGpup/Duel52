//! Self-play, and the trajectory shard it writes — `PLAN.md` Phase 3, step 3.
//!
//! One generation of the AlphaZero loop starts here: play `N` games with the current
//! checkpoint on both sides, record the search's visit distribution at every decision, and
//! write the lot to a `.d52sp` shard for the Python trainer to fit.
//!
//! # The shard stores trajectories, not tensors
//!
//! `PLAN.md` is specific about this and the reason is worth restating: **the buffer holds
//! `(config, seed, decision sequence)`, and the observations are recomputed on demand.** The
//! engine is deterministic, so replaying a game costs a few hundred microseconds of CPU,
//! while storing its encoded observations costs 17 KB *per decision* — 1.2 MB a game, against
//! 9 KB for the trajectory. More importantly it means a change to `encode.rs` costs a replay
//! rather than a discarded corpus, and `FINDINGS.md` F2.7 and F3.1 both expect the slot bound
//! to move.
//!
//! Two consequences shape the format:
//!
//! - **Actions are stored as indices into [`GameState::legal_actions`]**, not as encoded
//!   action indices. A legal-action list is a property of the engine; an encoded index is a
//!   property of the current action layout. Storing the latter would tie every shard to one
//!   `action_dim`, which is the coupling this design exists to avoid.
//! - **Forced decisions are not recorded.** A position with one legal action carries no
//!   policy signal and no search was run, so the replay skips them by the same rule the
//!   generator did: consume a sample only where `legal.len() > 1`.
//!
//! # What a sample's targets are
//!
//! The policy target is the root **visit distribution** (`DESIGN.md` §6), stored sparsely
//! over the legal actions that got a visit. The value target is the game's eventual result
//! from the point of view of *the player to move at that node*, which the replay reconstructs
//! — it is not stored per sample, because it is one bit per game.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::agents::{NetMctsAgent, RootNoise};
use crate::config::GameConfig;
use crate::encode::{action_layout_hash, obs_layout_hash};
use crate::outcome::{DrawReason, Outcome};
use crate::player::Player;
use crate::rng::Rng;
use crate::state::GameState;

/// Shard magic. Distinct from the checkpoint's and the dump's, so a file handed to the wrong
/// reader is rejected rather than misread.
pub const SHARD_MAGIC: &[u8; 6] = b"D52SP\0";
/// Version 2 splits the single "draw" outcome byte into its three reasons, because a
/// learner values them differently — see [`outcome_code`] and `FINDINGS.md` F3.6. Version 1
/// shards are refused rather than read, because the information needed to interpret them
/// correctly is not in the file: reading one would silently train every engine-declared
/// stalemate as an honest tie, which is the bug the version exists to prevent.
pub const SHARD_VERSION: u16 = 2;

/// Stream tags, so a game's search and its move-sampling are independent streams of one seed
/// and the whole game reproduces from `(checkpoint, config, seed)`.
const SEARCH_STREAM: u64 = 0x5350_0000_0000_0001;
const PICK_STREAM: u64 = 0x5350_0000_0000_0002;

/// How self-play differs from evaluation play.
#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    pub sims: usize,
    pub c_puct: f32,
    /// Root exploration noise. Self-play only — `AgentSpec::build` never adds it.
    pub noise: RootNoise,
    /// Visit counts are raised to `1 / temperature` before sampling.
    pub temperature: f32,
    /// Decisions after which play switches to the most-visited action. Early moves are
    /// sampled for diversity; late moves are played properly, so the value target is the
    /// result of a game both sides were trying to win.
    pub temperature_decisions: u32,
}

impl Default for SelfPlayConfig {
    fn default() -> SelfPlayConfig {
        SelfPlayConfig {
            sims: NetMctsAgent::DEFAULT_SIMS,
            c_puct: NetMctsAgent::DEFAULT_C_PUCT,
            noise: RootNoise::DEFAULT,
            temperature: 1.0,
            // A self-play game runs ~70 decisions, so this is roughly the first third —
            // the same proportion AlphaZero's 30-of-~80 chess moves works out to.
            temperature_decisions: 24,
        }
    }
}

/// One recorded decision.
#[derive(Clone, Debug)]
pub struct Sample {
    /// Index into `legal_actions()` of the action actually played.
    pub chosen: u16,
    /// The root's backed-up value for the player to move, in `0.0..=1.0`. Diagnostic: it is
    /// what makes "is the value head learning anything?" answerable without a second run.
    pub root_value: f32,
    /// `(index into legal_actions, share of root visits)`, for the actions that got one.
    pub policy: Vec<(u16, f32)>,
}

/// One self-play game.
#[derive(Clone, Debug)]
pub struct GameRecord {
    pub seed: u64,
    pub outcome: Outcome,
    pub samples: Vec<Sample>,
}

/// What a generation produced, for the progress line and for `FINDINGS.md`.
#[derive(Clone, Debug, Default)]
pub struct SelfPlayReport {
    pub games: usize,
    pub samples: usize,
    pub p0_wins: usize,
    pub p1_wins: usize,
    pub draws: usize,
    pub total_decisions: usize,
    pub total_plies: usize,
    pub max_slots_seen: usize,
    pub elapsed_secs: f64,
    pub bytes: usize,
}

impl SelfPlayReport {
    pub fn report(&self, path: &Path) -> String {
        let g = self.games.max(1) as f64;
        format!(
            "wrote {} — {} games, {} samples, {:.1} MB\n  \
             {:.1} games/sec · {:.1} decisions/game · P0 {:.1}% P1 {:.1}% draw {:.1}%\n  \
             widest lane side seen: {}\n",
            path.display(),
            self.games,
            self.samples,
            self.bytes as f64 / 1e6,
            self.games as f64 / self.elapsed_secs.max(1e-9),
            self.samples as f64 / g,
            100.0 * self.p0_wins as f64 / g,
            100.0 * self.p1_wins as f64 / g,
            100.0 * self.draws as f64 / g,
            self.max_slots_seen,
        )
    }
}

/// Play one self-play game. Reproducible from `(checkpoint, config, seed)` alone.
pub fn play_game(
    config: GameConfig,
    sp: &SelfPlayConfig,
    checkpoint: &Path,
    seed: u64,
) -> GameRecord {
    let mut state = GameState::new(config, seed);
    let mut agent = NetMctsAgent::derived(checkpoint, seed, SEARCH_STREAM, sp.sims)
        .with_c_puct(sp.c_puct)
        .with_root_noise(Some(sp.noise));
    let mut rng = Rng::derive(seed, PICK_STREAM);
    let mut samples: Vec<Sample> = Vec::new();

    while !state.outcome.is_over() {
        let legal = state.legal_actions();
        if legal.len() == 1 {
            // Forced. No search, no sample — and the replay skips it by the same rule.
            state.apply_trusted(legal[0]);
            continue;
        }
        let result = agent.search(&state, &legal);
        let total: u32 = result.visits.iter().sum();

        let policy: Vec<(u16, f32)> = if total == 0 {
            // Only reachable at `sims == 1`, where the single simulation expands the root
            // and backs up without traversing an edge. Uniform is the honest target there.
            let p = 1.0 / legal.len() as f32;
            (0..legal.len()).map(|i| (i as u16, p)).collect()
        } else {
            result
                .visits
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0)
                .map(|(i, &v)| (i as u16, v as f32 / total as f32))
                .collect()
        };

        let chosen = if (samples.len() as u32) < sp.temperature_decisions {
            sample_policy(&policy, sp.temperature, &mut rng)
        } else {
            // Most-visited. Ties go to the lowest index, which is deterministic and matches
            // `NetMctsAgent::choose`.
            policy
                .iter()
                .copied()
                .fold((u16::MAX, f32::NEG_INFINITY), |best, e| {
                    if e.1 > best.1 {
                        e
                    } else {
                        best
                    }
                })
                .0
        };

        samples.push(Sample {
            chosen,
            root_value: result.root_value,
            policy,
        });
        state.apply_trusted(legal[chosen as usize]);
    }

    GameRecord {
        seed,
        outcome: state.outcome,
        samples,
    }
}

/// Sample an entry with probability proportional to `share^(1/temperature)`.
fn sample_policy(policy: &[(u16, f32)], temperature: f32, rng: &mut Rng) -> u16 {
    debug_assert!(!policy.is_empty());
    if temperature <= 0.0 {
        return policy
            .iter()
            .copied()
            .fold((u16::MAX, f32::NEG_INFINITY), |b, e| if e.1 > b.1 { e } else { b })
            .0;
    }
    let inv = 1.0 / temperature;
    let weights: Vec<f64> = policy
        .iter()
        .map(|&(_, p)| (p as f64).powf(inv as f64))
        .collect();
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) {
        return policy[rng.index(policy.len())].0;
    }
    let mut r = rng.unit() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r < 0.0 {
            return policy[i].0;
        }
    }
    // Floating-point residue only.
    policy[policy.len() - 1].0
}

/// Play `games` self-play games and write a shard.
///
/// Games are sharded by contiguous seed range, so the file's game order — and therefore its
/// bytes — do not depend on `threads`. `phase3_a_shard_does_not_depend_on_the_thread_count`
/// pins that, for the same reason the ladder has such a test: a corpus whose contents move
/// when you change `--threads` is not reproducible.
#[allow(clippy::too_many_arguments)]
pub fn run(
    config: GameConfig,
    sp: &SelfPlayConfig,
    checkpoint: &Path,
    first_seed: u64,
    games: usize,
    threads: usize,
    generation: u32,
    out: &Path,
    progress: bool,
) -> Result<SelfPlayReport, String> {
    let started = Instant::now();
    // Load once, up front, so a bad checkpoint is an error here rather than a panic inside a
    // worker thread a minute later.
    let evaluator = crate::nn::evaluator_for(checkpoint, &config)?;
    let arch = evaluator.arch();
    crate::encode::reset_observed_max_slots();

    let threads = threads.max(1).min(games.max(1));
    let shards: Vec<(usize, usize)> = (0..threads)
        .map(|t| (games * t / threads, games * (t + 1) / threads))
        .filter(|(lo, hi)| lo < hi)
        .collect();

    let done = AtomicUsize::new(0);
    let report_every = (games / 20).clamp(1, 500);

    let results: Vec<Vec<GameRecord>> = std::thread::scope(|scope| {
        let handles: Vec<_> = shards
            .iter()
            .map(|&(lo, hi)| {
                let done = &done;
                scope.spawn(move || {
                    let mut out = Vec::with_capacity(hi - lo);
                    for g in lo..hi {
                        out.push(play_game(config, sp, checkpoint, first_seed + g as u64));
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if progress && (n % report_every == 0 || n == games) {
                            let secs = started.elapsed().as_secs_f64();
                            let rate = n as f64 / secs.max(1e-9);
                            eprintln!(
                                "  self-play {n}/{games} · {rate:.1} games/sec · \
                                 eta {:.0}s",
                                (games - n) as f64 / rate.max(1e-9)
                            );
                        }
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a self-play worker panicked"))
            .collect()
    });

    let mut report = SelfPlayReport::default();
    let mut header = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(header, "engine_version={}", crate::VERSION);
    let _ = writeln!(header, "generation={generation}");
    let _ = writeln!(header, "checkpoint={}", checkpoint.display());
    let _ = writeln!(header, "first_seed={first_seed}");
    let _ = writeln!(header, "games={games}");
    let _ = writeln!(header, "sims={}", sp.sims);
    let _ = writeln!(header, "c_puct={}", sp.c_puct);
    let _ = writeln!(header, "dirichlet_alpha={}", sp.noise.alpha);
    let _ = writeln!(header, "dirichlet_weight={}", sp.noise.weight);
    let _ = writeln!(header, "temperature={}", sp.temperature);
    let _ = writeln!(
        header,
        "temperature_decisions={}",
        sp.temperature_decisions
    );
    let _ = writeln!(header, "width={}", arch.width);
    let _ = writeln!(header, "blocks={}", arch.blocks);
    // Recorded, not relied on: the replay recomputes observations with whatever encoder is
    // current. They are here so a shard can say which layout produced it.
    let _ = writeln!(
        header,
        "obs_layout_hash={:016x}",
        crate::encode::obs_layout_hash(&config)
    );
    let _ = writeln!(
        header,
        "action_layout_hash={:016x}",
        crate::encode::action_layout_hash(&config)
    );

    let config_text = config.to_config_string();
    let mut bytes: Vec<u8> = Vec::with_capacity(games * 12_000);
    bytes.extend_from_slice(SHARD_MAGIC);
    bytes.extend_from_slice(&SHARD_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(config_text.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(config_text.as_bytes());

    for record in results.iter().flatten() {
        report.games += 1;
        report.samples += record.samples.len();
        report.total_decisions += record.samples.len();
        match record.outcome {
            Outcome::Win(Player::P0) => report.p0_wins += 1,
            Outcome::Win(Player::P1) => report.p1_wins += 1,
            _ => report.draws += 1,
        }
        bytes.extend_from_slice(&record.seed.to_le_bytes());
        bytes.extend_from_slice(&(record.samples.len() as u32).to_le_bytes());
        bytes.push(outcome_code(record.outcome));
        for sample in &record.samples {
            bytes.extend_from_slice(&sample.chosen.to_le_bytes());
            bytes.extend_from_slice(&(sample.policy.len() as u16).to_le_bytes());
            bytes.extend_from_slice(&sample.root_value.to_le_bytes());
            for &(index, share) in &sample.policy {
                bytes.extend_from_slice(&index.to_le_bytes());
                bytes.extend_from_slice(&share.to_le_bytes());
            }
        }
    }

    if let Some(dir) = out.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create `{}`: {e}", dir.display()))?;
        }
    }
    std::fs::write(out, &bytes).map_err(|e| format!("cannot write `{}`: {e}", out.display()))?;

    report.bytes = bytes.len();
    report.max_slots_seen = crate::encode::observed_max_slots();
    report.elapsed_secs = started.elapsed().as_secs_f64();
    Ok(report)
}

/// How a game ended, in one byte.
///
/// **The three kinds of draw are kept apart, and that is the whole reason for shard version
/// 2.** `FINDINGS.md` F3.6: an engine-declared stalemate is worth
/// `config.stalemate_value` to a learner while a mutual lane win — a real outcome in
/// `game_rules.md` §7 — is worth half a point. A shard that only recorded "draw" could not
/// tell the replay which one it was, so every stall would have trained as an honest tie.
fn outcome_code(outcome: Outcome) -> u8 {
    match outcome {
        Outcome::Win(Player::P0) => 0,
        Outcome::Win(Player::P1) => 1,
        Outcome::Draw(DrawReason::MutualLaneWin) => 2,
        Outcome::Draw(DrawReason::Stalemate) => 3,
        Outcome::Draw(DrawReason::PlyLimit) => 4,
        // Self-play only records finished games; `run` plays every one to a terminal.
        Outcome::Ongoing => unreachable!("an unfinished game reached the shard writer"),
    }
}

fn outcome_from_code(code: u8) -> Outcome {
    match code {
        0 => Outcome::Win(Player::P0),
        1 => Outcome::Win(Player::P1),
        2 => Outcome::Draw(DrawReason::MutualLaneWin),
        4 => Outcome::Draw(DrawReason::PlyLimit),
        // 3, and anything unknown, is the conservative reading: a stall.
        _ => Outcome::Draw(DrawReason::Stalemate),
    }
}

// ============================================================== reading and replaying ==

/// A shard, parsed but not yet replayed.
#[derive(Clone, Debug)]
pub struct Shard {
    pub path: PathBuf,
    /// The header's `key=value` lines, in file order.
    pub header: Vec<(String, String)>,
    /// The configuration the games were played under, reconstructed from the shard.
    pub config: GameConfig,
    pub games: Vec<ShardGame>,
}

#[derive(Clone, Debug)]
pub struct ShardGame {
    pub seed: u64,
    pub outcome_code: u8,
    pub samples: Vec<Sample>,
}

impl ShardGame {
    /// How this game ended, including *which* kind of draw — see [`outcome_code`].
    pub fn outcome(&self) -> Outcome {
        outcome_from_code(self.outcome_code)
    }
}

impl Shard {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.header
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn sample_count(&self) -> usize {
        self.games.iter().map(|g| g.samples.len()).sum()
    }

    pub fn read(path: &Path) -> Result<Shard, String> {
        let data = std::fs::read(path).map_err(|e| format!("cannot read `{}`: {e}", path.display()))?;
        let mut at = 0usize;

        macro_rules! take {
            ($n:expr, $what:expr) => {{
                let n = $n;
                if data.len() < at + n {
                    return Err(format!("{}: truncated reading {}", path.display(), $what));
                }
                let slice = &data[at..at + n];
                at += n;
                slice
            }};
        }

        if take!(6, "magic") != SHARD_MAGIC {
            return Err(format!("{}: not a .d52sp shard (bad magic)", path.display()));
        }
        let version = u16::from_le_bytes(take!(2, "version").try_into().expect("2 bytes"));
        if version != SHARD_VERSION {
            let why = if version == 1 {
                " — version 1 recorded every draw as one byte, so it cannot say which of \
                 its draws were engine-declared stalemates. Regenerate rather than reuse \
                 it; see FINDINGS.md F3.6"
            } else {
                ""
            };
            return Err(format!(
                "{}: shard format version {version}, this build reads {SHARD_VERSION}{why}",
                path.display()
            ));
        }
        let header_len =
            u32::from_le_bytes(take!(4, "header length").try_into().expect("4 bytes")) as usize;
        let config_len =
            u32::from_le_bytes(take!(4, "config length").try_into().expect("4 bytes")) as usize;
        let header_text = std::str::from_utf8(take!(header_len, "header"))
            .map_err(|_| format!("{}: header is not UTF-8", path.display()))?
            .to_string();
        let config_text = std::str::from_utf8(take!(config_len, "config"))
            .map_err(|_| format!("{}: config is not UTF-8", path.display()))?
            .to_string();

        let header: Vec<(String, String)> = header_text
            .lines()
            .filter_map(|line| line.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let config = GameConfig::from_config_str(&config_text)
            .map_err(|e| format!("{}: embedded config is invalid: {e}", path.display()))?;

        // A sample's `chosen` and its policy indices are positions in `legal_actions()`, so
        // a shard is only replayable by a build that enumerates the same actions in the same
        // order. The header has always carried the layout hashes; until 2026-09-03 nothing
        // read them back, and removing `Action::Pass` is exactly the change that would have
        // gone unnoticed — the indices still parse, they just name different moves. That
        // failure is invisible: nothing crashes, the corpus is quietly wrong, and the
        // natural suspect is the training loop.
        for (key, current) in [
            ("action_layout_hash", action_layout_hash(&config)),
            ("obs_layout_hash", obs_layout_hash(&config)),
        ] {
            let recorded = header
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
                .ok_or_else(|| format!("{}: header has no {key}", path.display()))?;
            let expected = format!("{current:016x}");
            if recorded != expected {
                return Err(format!(
                    "{}: {key} is {recorded} in the shard but {expected} in this build — the \
                     shard was generated against a different encoder and its action indices \
                     mean something else here. Regenerate it.",
                    path.display()
                ));
            }
        }

        let mut games = Vec::new();
        while at < data.len() {
            let seed = u64::from_le_bytes(take!(8, "game seed").try_into().expect("8 bytes"));
            let count =
                u32::from_le_bytes(take!(4, "decision count").try_into().expect("4 bytes")) as usize;
            let outcome_code = take!(1, "outcome")[0];
            let mut samples = Vec::with_capacity(count);
            for _ in 0..count {
                let chosen = u16::from_le_bytes(take!(2, "chosen").try_into().expect("2 bytes"));
                let entries =
                    u16::from_le_bytes(take!(2, "entry count").try_into().expect("2 bytes")) as usize;
                let root_value =
                    f32::from_le_bytes(take!(4, "root value").try_into().expect("4 bytes"));
                let mut policy = Vec::with_capacity(entries);
                for _ in 0..entries {
                    let index = u16::from_le_bytes(take!(2, "policy index").try_into().expect("2"));
                    let share = f32::from_le_bytes(take!(4, "policy share").try_into().expect("4"));
                    policy.push((index, share));
                }
                samples.push(Sample {
                    chosen,
                    root_value,
                    policy,
                });
            }
            games.push(ShardGame {
                seed,
                outcome_code,
                samples,
            });
        }

        Ok(Shard {
            path: path.to_path_buf(),
            header,
            config,
            games,
        })
    }
}

/// A replayed shard: exactly the tensors the trainer fits, in the current encoding.
///
/// Observations are **sparse** — `FINDINGS.md` F3.3 measures them at 4.8% dense — because a
/// dense generation is 2 GB of f32 and a sparse one is 300 MB. The trainer scatters a batch
/// into a dense tensor on the way to the device.
#[derive(Clone, Debug, Default)]
pub struct TrainingSet {
    pub samples: usize,
    pub obs_dim: usize,
    pub action_dim: usize,
    /// `obs_offset[i]..obs_offset[i+1]` indexes `obs_index` / `obs_value` for sample `i`.
    pub obs_offset: Vec<u32>,
    pub obs_index: Vec<u32>,
    pub obs_value: Vec<f32>,
    /// Same shape, over encoded action indices. This is where the layout-independent legal
    /// indices in the shard become indices into the policy head.
    pub policy_offset: Vec<u32>,
    pub policy_index: Vec<u32>,
    pub policy_prob: Vec<f32>,
    /// The value target, in `-1.0..=1.0` to match the network's `tanh` head.
    pub value: Vec<f32>,
    /// The search's own root value, same convention. Diagnostic only.
    pub root_value: Vec<f32>,
}

/// Replay one game and append its samples to `out`.
///
/// `stride` keeps one sample in `stride`. Consecutive decisions in one game are highly
/// correlated — the board barely moves between two of the three actions in a turn — so
/// thinning is nearly free in signal and linear in memory, and a generation's replayed
/// tensors are the largest thing the trainer holds. The phase is offset by the game seed so
/// the kept samples are not always the same points in the turn cycle.
fn replay_game(
    config: GameConfig,
    game: &ShardGame,
    stride: usize,
    obs: &mut Vec<f32>,
    out: &mut TrainingSet,
) {
    let mut state = GameState::new(config, game.seed);
    let mut next = 0usize;
    let phase = (game.seed % stride as u64) as usize;
    while !state.outcome.is_over() && next < game.samples.len() {
        let legal = state.legal_actions();
        if legal.len() == 1 {
            // The generator recorded nothing here, so neither does the replay.
            state.apply_trusted(legal[0]);
            continue;
        }
        let sample = &game.samples[next];
        let keep = (next + phase) % stride == 0;
        next += 1;
        if !keep {
            state.apply_trusted(legal[sample.chosen as usize]);
            continue;
        }
        let actor = state.acting_player();

        crate::encode::encode_observation(&state, actor, obs);
        for (i, &v) in obs.iter().enumerate() {
            if v != 0.0 {
                out.obs_index.push(i as u32);
                out.obs_value.push(v);
            }
        }
        out.obs_offset.push(out.obs_index.len() as u32);

        for &(legal_index, share) in &sample.policy {
            let action = legal[legal_index as usize];
            out.policy_index
                .push(crate::encode::encode_action(&action, &state) as u32);
            out.policy_prob.push(share);
        }
        out.policy_offset.push(out.policy_index.len() as u32);

        // `learning_value`, not `value_for` — see `GameConfig::stalemate_value`. Rescaled
        // from the engine's `0.0..=1.0` to the `tanh` head's `-1.0..=1.0`.
        out.value.push(
            2.0 * config.learning_value(outcome_from_code(game.outcome_code), actor) - 1.0,
        );
        out.root_value.push(2.0 * sample.root_value - 1.0);
        out.samples += 1;

        state.apply_trusted(legal[sample.chosen as usize]);
    }
    debug_assert_eq!(
        next,
        game.samples.len(),
        "replay of seed {} consumed {next} of {} samples — the recorded trajectory does not \
         match what the engine now does",
        game.seed,
        game.samples.len()
    );
}

/// Replay every game in `shard`, encoding observations with the **current** encoder and
/// valuing outcomes under the configuration the shard was played with.
///
/// `threads` shards by game, and the output is concatenated in shard order, so the result
/// does not depend on the thread count. `stride` thins the samples — see [`replay_game`].
pub fn replay(shard: &Shard, threads: usize, stride: usize) -> TrainingSet {
    replay_with(shard, shard.config, threads, stride)
}

/// [`replay`], but valuing outcomes under `config` rather than the shard's own.
///
/// The reason this is public rather than a test hook: `config.stalemate_value` is a
/// *learning* weight, not a rule (`FINDINGS.md` F3.6), so changing it should not cost a
/// corpus. Trajectories are stored precisely so that a decision like "how bad is a stall,
/// really?" can be re-asked of games already played, and re-valuing 50,000 games is seconds
/// of CPU against hours of self-play.
///
/// `config` must be the shard's configuration in every respect that changes how a game
/// *plays*; only the learning weights are meant to differ. The replay reconstructs positions
/// from `(config, seed, decisions)`, so a genuinely different rules config would desynchronise
/// and the sample-count check in `Shard`'s callers would catch it.
pub fn replay_with(
    shard: &Shard,
    config: GameConfig,
    threads: usize,
    stride: usize,
) -> TrainingSet {
    let stride = stride.max(1);
    let obs_dim = crate::encode::obs_dim(&config);
    let action_dim = crate::encode::action_dim(&config);
    let n = shard.games.len();
    let threads = threads.max(1).min(n.max(1));

    let ranges: Vec<(usize, usize)> = (0..threads)
        .map(|t| (n * t / threads, n * (t + 1) / threads))
        .filter(|(lo, hi)| lo < hi)
        .collect();

    let parts: Vec<TrainingSet> = std::thread::scope(|scope| {
        let handles: Vec<_> = ranges
            .iter()
            .map(|&(lo, hi)| {
                let games = &shard.games[lo..hi];
                scope.spawn(move || {
                    let mut part = TrainingSet {
                        obs_dim,
                        action_dim,
                        ..TrainingSet::default()
                    };
                    let mut obs = vec![0.0f32; obs_dim];
                    for game in games {
                        replay_game(config, game, stride, &mut obs, &mut part);
                    }
                    part
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a replay worker panicked"))
            .collect()
    });

    let mut out = TrainingSet {
        obs_dim,
        action_dim,
        obs_offset: vec![0],
        policy_offset: vec![0],
        ..TrainingSet::default()
    };
    for part in parts {
        let obs_base = out.obs_index.len() as u32;
        let policy_base = out.policy_index.len() as u32;
        out.samples += part.samples;
        out.obs_index.extend_from_slice(&part.obs_index);
        out.obs_value.extend_from_slice(&part.obs_value);
        out.obs_offset
            .extend(part.obs_offset.iter().map(|o| o + obs_base));
        out.policy_index.extend_from_slice(&part.policy_index);
        out.policy_prob.extend_from_slice(&part.policy_prob);
        out.policy_offset
            .extend(part.policy_offset.iter().map(|o| o + policy_base));
        out.value.extend_from_slice(&part.value);
        out.root_value.extend_from_slice(&part.root_value);
    }
    out
}
