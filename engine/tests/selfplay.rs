//! Phase 3 step 3: self-play, the `.d52sp` shard, and the replay that feeds the trainer.
//!
//! The `phase3_` prefix follows `CLAUDE.md`'s convention for tests that pin *machinery*
//! rather than a ruling in `game_rules.md`, so a failure reads as "the training corpus
//! broke" rather than "the rules broke".
//!
//! # What is actually at stake here
//!
//! A training corpus can be wrong in a way that nothing crashes on. `PLAN.md` chose to store
//! trajectories rather than tensors precisely so that a shard survives an encoder change —
//! but that only holds if a replayed trajectory really does reproduce the game it was
//! recorded from. If it drifts, every observation is paired with a policy target from a
//! *different* position, the network fits noise, and the natural suspect is the learning
//! rate. [`phase3_a_shard_replays_to_the_positions_it_was_recorded_from`] is the tripwire.

use std::path::{Path, PathBuf};

use duel52_engine::selfplay::{self, SelfPlayConfig};
use duel52_engine::{GameConfig, GameState};

/// A small random-init checkpoint under the default config, written once per test process.
///
/// `Weights::random` is what makes this possible without Python in the loop — a test that
/// needed `maturin develop` first would not run under plain `cargo test`, and a test that
/// does not run is not a test.
fn test_checkpoint() -> PathBuf {
    use std::sync::OnceLock;
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let config = GameConfig::default();
        let arch = duel52_engine::nn::Arch {
            obs_dim: duel52_engine::encode::obs_dim(&config),
            action_dim: duel52_engine::encode::action_dim(&config),
            width: 24,
            blocks: 2,
            value_hidden: 12,
        };
        let path = std::env::temp_dir().join(format!("duel52-selfplay-{}.d52nn", std::process::id()));
        duel52_engine::nn::Weights::random(20260904, arch)
            .save(&path, &config)
            .expect("write the self-play test checkpoint");
        path
    })
    .clone()
}

fn tiny() -> SelfPlayConfig {
    SelfPlayConfig {
        sims: 12,
        temperature_decisions: 6,
        ..SelfPlayConfig::default()
    }
}

fn write_shard(games: usize, threads: usize, tag: &str) -> PathBuf {
    let out = std::env::temp_dir().join(format!(
        "duel52-shard-{}-{tag}.d52sp",
        std::process::id()
    ));
    selfplay::run(
        GameConfig::default(),
        &tiny(),
        &test_checkpoint(),
        1,
        games,
        threads,
        0,
        &out,
        false,
    )
    .expect("self-play should write a shard");
    out
}

/// `CLAUDE.md`: "Everything is seeded and deterministic. Same seed + same config →
/// identical game." A corpus whose *contents* move when you change `--threads` is not
/// reproducible, and the ladder has the same test for the same reason.
#[test]
fn phase3_a_shard_does_not_depend_on_the_thread_count() {
    let one = std::fs::read(write_shard(6, 1, "t1")).expect("read shard");
    let many = std::fs::read(write_shard(6, 4, "t4")).expect("read shard");
    assert_eq!(
        one, many,
        "the same six games sharded across four threads produced different bytes"
    );
}

/// The claim the whole storage design rests on: a recorded trajectory replays into exactly
/// the positions it was recorded from.
///
/// The shard stores indices into `legal_actions()`, so this is really the assertion that the
/// engine's legal-action enumeration is a deterministic function of `(config, seed, action
/// history)`. If that ever stops holding, a shard silently pairs observations with policy
/// targets from elsewhere.
#[test]
fn phase3_a_shard_replays_to_the_positions_it_was_recorded_from() {
    let path = write_shard(8, 3, "replay");
    let shard = selfplay::Shard::read(&path).expect("read shard");
    let set = selfplay::replay(&shard, 2, 1);

    assert_eq!(set.samples, shard.sample_count(), "replay lost samples");
    assert!(set.samples > 50, "only {} samples — too few to mean much", set.samples);
    assert_eq!(set.obs_offset.len(), set.samples + 1);
    assert_eq!(set.policy_offset.len(), set.samples + 1);
    assert_eq!(set.value.len(), set.samples);

    // Every policy target is a distribution over encoded actions.
    for i in 0..set.samples {
        let lo = set.policy_offset[i] as usize;
        let hi = set.policy_offset[i + 1] as usize;
        assert!(hi > lo, "sample {i} has an empty policy target");
        let total: f32 = set.policy_prob[lo..hi].iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "sample {i}'s policy target sums to {total}"
        );
        for &index in &set.policy_index[lo..hi] {
            assert!((index as usize) < set.action_dim, "action index {index} is out of range");
        }
    }
    // And every value target is one of the three outcomes, from the mover's point of view.
    for (i, &v) in set.value.iter().enumerate() {
        assert!(
            v == 1.0 || v == 0.0 || v == -1.0,
            "sample {i} has value target {v}, which is not a win, draw or loss"
        );
    }
}

/// Replaying is a function of the shard, not of how it is parallelised.
#[test]
fn phase3_replay_does_not_depend_on_the_thread_count() {
    let shard = selfplay::Shard::read(&write_shard(6, 2, "rt")).expect("read shard");
    let one = selfplay::replay(&shard, 1, 1);
    let many = selfplay::replay(&shard, 4, 1);
    assert_eq!(one.samples, many.samples);
    assert_eq!(one.obs_index, many.obs_index);
    assert_eq!(one.obs_value, many.obs_value);
    assert_eq!(one.policy_index, many.policy_index);
    assert_eq!(one.value, many.value);
}

/// Striding thins the corpus without changing what survives: a strided replay must be a
/// *subsequence* of the full one, sample for sample, or the trainer would be fitting
/// observations that were never generated.
#[test]
fn phase3_striding_keeps_a_subsequence_of_the_full_replay() {
    let shard = selfplay::Shard::read(&write_shard(6, 2, "stride")).expect("read shard");
    let full = selfplay::replay(&shard, 2, 1);
    let thinned = selfplay::replay(&shard, 2, 3);

    assert!(thinned.samples > 0);
    assert!(
        thinned.samples < full.samples,
        "stride 3 kept {} of {} samples",
        thinned.samples,
        full.samples
    );
    // Every kept sample must appear, intact, somewhere in the full replay: match on the
    // observation's non-zero indices, which identify a position tightly.
    let mut cursor = 0usize;
    for i in 0..thinned.samples {
        let want = &thinned.obs_index[thinned.obs_offset[i] as usize..thinned.obs_offset[i + 1] as usize];
        let found = (cursor..full.samples).find(|&j| {
            &full.obs_index[full.obs_offset[j] as usize..full.obs_offset[j + 1] as usize] == want
                && full.value[j] == thinned.value[i]
        });
        let j = found.unwrap_or_else(|| panic!("strided sample {i} is not in the full replay"));
        cursor = j + 1;
    }
}

/// The shard is self-describing: it carries the configuration its games were played under,
/// so a replay cannot be run against a different one by accident.
#[test]
fn phase3_a_shard_carries_the_config_its_games_were_played_under() {
    let shard = selfplay::Shard::read(&write_shard(4, 2, "cfg")).expect("read shard");
    let config = GameConfig::default();
    assert_eq!(shard.config.variant, config.variant);
    assert_eq!(shard.config.two_power, config.two_power);
    assert_eq!(shard.config.encoding_slots, config.encoding_slots);
    assert_eq!(shard.get("sims"), Some("12"));
    assert!(shard.get("checkpoint").is_some());
}

/// A shard fed to the wrong reader must be rejected by magic rather than misread. The three
/// binary formats in the project deliberately do not share one.
#[test]
fn phase3_a_checkpoint_is_not_mistaken_for_a_shard() {
    let err = selfplay::Shard::read(&test_checkpoint()).expect_err("a checkpoint is not a shard");
    assert!(err.contains("bad magic"), "unexpected error: {err}");
}

/// A shard generated against a different action layout is **refused**, not replayed.
///
/// This is the sharpest version of the module's opening warning. A sample's `chosen` is a
/// position in `legal_actions()`, so a build that enumerates actions differently reads the
/// same bytes as different moves — every observation paired with a policy target from a
/// position the agent never saw. Nothing crashes. The header has carried the layout hashes
/// since the format was written; removing `Action::Pass` on 2026-09-03 is what made it
/// obvious that nobody was reading them back.
///
/// The corruption here is one hex digit in the recorded hash, which stands in for any
/// encoder change at all — the reader's job is to compare, not to understand.
#[test]
fn phase3_a_shard_from_a_different_action_layout_is_refused() {
    let path = write_shard(2, 1, "layout");
    let good = std::fs::read(&path).expect("read shard");

    let recorded = selfplay::Shard::read(&path).expect("the untouched shard reads");
    let hash = recorded
        .get("action_layout_hash")
        .expect("the header carries the hash")
        .to_string();

    // Flip the first hex digit of the recorded hash, in place, so nothing else moves.
    let at = good
        .windows(hash.len())
        .position(|w| w == hash.as_bytes())
        .expect("the hash appears verbatim in the header");
    let mut corrupted = good.clone();
    corrupted[at] = if hash.as_bytes()[0] == b'0' { b'1' } else { b'0' };

    let bad_path = path.with_extension("layout-drift.d52sp");
    std::fs::write(&bad_path, &corrupted).expect("write the corrupted shard");

    let err = selfplay::Shard::read(&bad_path).expect_err("a stale layout must be refused");
    assert!(
        err.contains("action_layout_hash") && err.contains("Regenerate"),
        "the error must name the hash and say what to do: {err}"
    );
}

/// Self-play must produce a *game*, not a fragment: every recorded trajectory has to end in
/// a terminal position when replayed.
#[test]
fn phase3_every_recorded_game_is_played_to_the_end() {
    let shard = selfplay::Shard::read(&write_shard(6, 2, "terminal")).expect("read shard");
    for game in &shard.games {
        let mut state = GameState::new(shard.config, game.seed);
        let mut next = 0usize;
        while !state.outcome.is_over() {
            let legal = state.legal_actions();
            let action = if legal.len() == 1 {
                legal[0]
            } else {
                let sample = &game.samples[next];
                next += 1;
                legal[sample.chosen as usize]
            };
            state.apply_trusted(action);
        }
        assert_eq!(next, game.samples.len(), "seed {} has leftover samples", game.seed);
        assert!(state.outcome.is_over());
    }
}

/// The path a `netmcts` agent takes to a checkpoint has to be the one the CLI parses, or a
/// gate that names a file will silently play a different one.
#[test]
fn phase3_netmcts_spec_round_trips_through_its_name() {
    use duel52_engine::AgentSpec;
    let path = test_checkpoint();
    let spec = AgentSpec::NetMcts {
        checkpoint: path.display().to_string(),
        sims: 96,
    };
    let reparsed = AgentSpec::parse(&spec.name()).expect("a spec's own name must parse");
    assert_eq!(spec, reparsed);

    // `@` splits from the right, so a checkpoint path containing one survives.
    let awkward = AgentSpec::parse("netmcts:runs/a@b/gen3.d52nn@64").expect("parse");
    assert_eq!(
        awkward,
        AgentSpec::NetMcts {
            checkpoint: "runs/a@b/gen3.d52nn".to_string(),
            sims: 64
        }
    );
    // No `@` at all means the default budget rather than an error.
    assert!(matches!(
        AgentSpec::parse("netmcts:x.d52nn"),
        Ok(AgentSpec::NetMcts { sims, .. }) if sims == duel52_engine::NetMctsAgent::DEFAULT_SIMS
    ));
    assert!(AgentSpec::parse("netmcts").is_err());
}

/// Sanity that the generator honours its own knobs: more simulations means a policy target
/// spread over more actions, because more of them get visited at all.
#[test]
fn phase3_more_simulations_visit_more_actions() {
    fn mean_support(sims: usize, tag: &str) -> f64 {
        let out = std::env::temp_dir().join(format!("duel52-sims-{}-{tag}.d52sp", std::process::id()));
        selfplay::run(
            GameConfig::default(),
            &SelfPlayConfig {
                sims,
                ..tiny()
            },
            &test_checkpoint(),
            1,
            4,
            2,
            0,
            &out,
            false,
        )
        .expect("self-play");
        let shard = selfplay::Shard::read(&out).expect("read shard");
        let total: usize = shard
            .games
            .iter()
            .flat_map(|g| g.samples.iter())
            .map(|s| s.policy.len())
            .sum();
        total as f64 / shard.sample_count().max(1) as f64
    }
    let few = mean_support(4, "few");
    let many = mean_support(64, "many");
    assert!(
        many > few,
        "64 simulations spread over {many:.1} actions, 4 over {few:.1}"
    );
}

/// A shard path that does not exist is a clear error, not a panic — the trainer calls this
/// from Python and a stack trace through PyO3 is worse than a sentence.
#[test]
fn phase3_a_missing_shard_is_an_error_not_a_panic() {
    let err = selfplay::Shard::read(Path::new("/nonexistent/nope.d52sp")).expect_err("should fail");
    assert!(err.contains("cannot read"), "unexpected error: {err}");
}

// ============================================ the stalemate is not worth half a point ==

/// `FINDINGS.md` F3.6, and the fix for it: an **engine-declared stalemate** is worth
/// `config.stalemate_value` to *both* players for learning, while a **mutual lane win** —
/// a real outcome in `game_rules.md` §7 — stays at half a point for both.
///
/// Getting this backwards is the whole failure: at 0.5 a stalemate is a safe half point,
/// mutual refusal to attack is a stable equilibrium, and a learner walks straight into it.
#[test]
fn rule_7_a_stalemate_is_not_worth_half_a_point_to_a_learner() {
    use duel52_engine::{DrawReason, Outcome, Player};

    let mut config = GameConfig::default();
    config.stalemate_value = 0.0;

    let stalemate = Outcome::Draw(DrawReason::Stalemate);
    for p in Player::BOTH {
        // Scoring is untouched — the Elo ladder still calls a draw half a point, so every
        // number in FINDINGS.md F1 and F2 means what it meant.
        assert_eq!(stalemate.value_for(p), 0.5, "scoring must not move");
        // Learning is not.
        assert_eq!(config.learning_value(stalemate, p), 0.0, "learning must move");
    }

    // A mutual lane win is a rule, not an engine artefact, and keeps its half point.
    let mutual = Outcome::Draw(DrawReason::MutualLaneWin);
    for p in Player::BOTH {
        assert_eq!(config.learning_value(mutual, p), 0.5);
    }

    // Wins and losses are untouched by any of this.
    assert_eq!(config.learning_value(Outcome::Win(Player::P0), Player::P0), 1.0);
    assert_eq!(config.learning_value(Outcome::Win(Player::P0), Player::P1), 0.0);

    // And the default configuration is exactly the old behaviour, so nothing measured
    // before this existed has quietly changed meaning.
    let old = GameConfig::default();
    assert_eq!(old.stalemate_value, 0.5);
    for p in Player::BOTH {
        assert_eq!(old.learning_value(stalemate, p), 0.5);
    }
}

/// The penalty has to reach the training targets, or it is a number in a config file that
/// nothing reads. A stalled game's samples must carry the penalty; a decisive game's must
/// not move at all.
#[test]
fn phase3_the_stalemate_penalty_reaches_the_value_targets() {
    use duel52_engine::{DrawReason, Outcome};

    let path = write_shard(8, 3, "penalty");
    let shard = selfplay::Shard::read(&path).expect("read shard");
    let stalls = shard
        .games
        .iter()
        .filter(|g| matches!(g.outcome(), Outcome::Draw(DrawReason::Stalemate)))
        .count();

    let neutral = selfplay::replay(&shard, 2, 1);
    let mut penalised_config = shard.config;
    penalised_config.stalemate_value = 0.0;
    let penalised = selfplay::replay_with(&shard, penalised_config, 2, 1);

    assert_eq!(neutral.samples, penalised.samples);
    let moved = neutral
        .value
        .iter()
        .zip(&penalised.value)
        .filter(|(a, b)| a != b)
        .count();
    if stalls == 0 {
        assert_eq!(moved, 0, "no stalls, so nothing should have moved");
    } else {
        assert!(moved > 0, "{stalls} stalled games but no value target moved");
        // Only the draws move, and they move down: 0.0 -> -1.0 on the tanh scale.
        for (a, b) in neutral.value.iter().zip(&penalised.value) {
            if a != b {
                assert_eq!(*a, 0.0, "a target that moved was not a draw");
                assert_eq!(*b, -1.0, "a penalised draw should be a loss, not {b}");
            }
        }
    }
}

/// A version 1 shard is refused with a sentence that says what to do, not just a number.
/// Version 1 recorded one byte for every draw, so its stalls cannot be told from its mutual
/// lane wins — reading one would silently train every stall as an honest tie.
#[test]
fn phase3_a_version_one_shard_is_refused_with_a_reason() {
    let path = write_shard(2, 1, "v1");
    let mut bytes = std::fs::read(&path).expect("read shard");
    bytes[6] = 1; // the version field, immediately after the six-byte magic
    let downgraded = path.with_extension("v1.d52sp");
    std::fs::write(&downgraded, &bytes).expect("write downgraded shard");

    let err = selfplay::Shard::read(&downgraded).expect_err("version 1 must be refused");
    assert!(err.contains("version 1"), "unexpected error: {err}");
    assert!(err.contains("F3.6"), "the error should point at the finding: {err}");
}
