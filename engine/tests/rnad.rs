//! `PLAN.md` item 8 — R-NaD, beside AlphaZero.
//!
//! Two kinds of test live here. The first kind pins that the R-NaD work **did not move
//! anything AlphaZero relies on** (the golden self-play shard is its sibling, in
//! `selfplay.rs`). The second pins the two pieces the engine gained for R-NaD: the `netsample`
//! agent and the optional linear value head.

use std::path::{Path, PathBuf};

use duel52_engine::agents::{play_game, Agent, AgentSpec, RandomAgent};
use duel52_engine::encode::{encode_observation, obs_dim};
use duel52_engine::nn::{Arch, Evaluator, Weights};
use duel52_engine::{GameConfig, GameState, Player};

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("duel52-rnad-{}-{name}.d52nn", std::process::id()))
}

fn header_fields(bytes: &[u8]) -> Vec<(String, String)> {
    Weights::header_of(bytes)
        .expect("a header")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// `PLAN.md` item 8, test 2: **every shipped checkpoint still loads, and rewriting it changes
/// nothing but the ruleset stamp the older ones predate.**
///
/// The payload must be byte-identical, and no key this work added (`value_head`) may appear
/// in the rewrite of an AlphaZero checkpoint. Older checkpoints legitimately gain the keys
/// that postdate them — `arch` (gen016/022/031 predate it) and `rules_name`/`rules_hash` —
/// and that was already true before R-NaD; nothing else may change.
#[test]
fn rnad_every_shipped_checkpoint_loads_and_rewrites_unchanged() {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
    let mut config = GameConfig::default();
    config.encoding_slots = 21;

    let mut files: Vec<PathBuf> = std::fs::read_dir(&models)
        .expect("models/ is tracked")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "d52nn"))
        .collect();
    files.sort();
    assert!(files.len() >= 6, "expected the six shipped checkpoints, found {}", files.len());

    for path in &files {
        let original = std::fs::read(path).unwrap();
        let weights = Weights::load(path, &config)
            .unwrap_or_else(|e| panic!("a shipped checkpoint no longer loads: {e}"));
        assert!(!weights.linear_value, "{} is an AlphaZero net", path.display());

        let rewritten = weights.to_bytes(&config);
        let (old_header, new_header) = (header_fields(&original), header_fields(&rewritten));
        for (key, value) in &old_header {
            let now = new_header.iter().find(|(k, _)| k == key);
            assert_eq!(
                now.map(|(_, v)| v),
                Some(value),
                "{}: header `{key}` changed on a rewrite",
                path.display()
            );
        }
        for (key, _) in &new_header {
            let added = !old_header.iter().any(|(k, _)| k == key);
            assert!(
                !added || ["arch", "rules_name", "rules_hash"].contains(&key.as_str()),
                "{}: a rewrite added header `{key}`",
                path.display()
            );
        }

        let payload = |bytes: &[u8]| {
            let header_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            bytes[12 + header_len as usize..].to_vec()
        };
        assert!(
            payload(&original) == payload(&rewritten),
            "{}: the weights changed on a rewrite",
            path.display()
        );
    }
}

/// The linear head is the tanh head with the tanh removed — the same accumulation, so
/// `tanh(linear)` is bit-identical to the tanh checkpoint's value. On both architectures.
#[test]
fn rnad_a_linear_value_head_is_the_tanh_head_without_the_tanh() {
    let config = GameConfig::default();
    let state = GameState::new(config.clone(), 7);
    let mut obs = vec![0.0f32; obs_dim(&config)];
    encode_observation(&state, state.acting_player(), &mut obs);

    let archs = [
        Arch { width: 16, blocks: 1, value_hidden: 8, ..Arch::default_for(&config) },
        Arch::lane_for(&config, 16, 1, 8),
    ];
    for (n, arch) in archs.into_iter().enumerate() {
        let mut tanh = Weights::random(11 + n as u64, arch);
        // Push the output well past ±1, where the two heads can be told apart.
        let bias = arch.params().iter().position(|(name, _)| name == "value2.bias").unwrap();
        tanh.params[bias][0] = 3.0;
        let linear = Weights {
            linear_value: true,
            ..tanh.clone()
        };

        let (tanh_path, linear_path) = (temp(&format!("tanh{n}")), temp(&format!("linear{n}")));
        tanh.save(&tanh_path, &config).unwrap();
        linear.save(&linear_path, &config).unwrap();

        let value = |path: &Path| {
            let evaluator = duel52_engine::nn::evaluator_for(path, &config).unwrap();
            let mut logits = vec![0.0f32; evaluator.action_dim()];
            let mut values = vec![0.0f32; 1];
            evaluator.eval_batch(&obs, 1, &mut logits, &mut values);
            (values[0], logits)
        };
        let ((squashed, logits_a), (raw, logits_b)) = (value(&tanh_path), value(&linear_path));
        assert!(raw.abs() > 1.0, "the test bias did not push the value past ±1 ({raw})");
        assert_eq!(raw.tanh().to_bits(), squashed.to_bits());
        assert_eq!(logits_a, logits_b, "the value head must not touch the policy");
    }
}

fn sampling_checkpoint() -> String {
    use std::sync::OnceLock;
    static PATH: OnceLock<String> = OnceLock::new();
    PATH.get_or_init(|| {
        let config = GameConfig::default();
        let arch = Arch { width: 16, blocks: 1, value_hidden: 8, ..Arch::default_for(&config) };
        let path = temp("netsample");
        Weights::random(20260913, arch).save(&path, &config).unwrap();
        path.to_string_lossy().into_owned()
    })
    .clone()
}

#[test]
fn rnad_netsample_parses_and_names_itself_back() {
    for text in ["netsample:models/a.d52nn", "netsample:models/a.d52nn@raw", "netsample:x@y.d52nn"] {
        let spec = AgentSpec::parse(text).unwrap();
        assert_eq!(spec.name(), text);
        assert!(spec.checkpoint().is_some());
    }
    assert_eq!(
        AgentSpec::parse("netsample:x@y.d52nn").unwrap(),
        AgentSpec::NetSample { checkpoint: "x@y.d52nn".into(), raw: false }
    );
    assert!(AgentSpec::parse("netsample").is_err());
    assert!(AgentSpec::parse("netsample:@raw").is_err());
}

/// Whole games, both seats, against `random`: every action it returns is one it was offered
/// (`apply_trusted` would panic otherwise in a debug build), and every game finishes.
#[test]
fn rnad_netsample_plays_legal_games_to_the_end() {
    for raw in [false, true] {
        let spec = AgentSpec::NetSample { checkpoint: sampling_checkpoint(), raw };
        for seed in 0..4u64 {
            let mut state = GameState::new(GameConfig::default(), seed);
            let mut net = spec.build(seed, 1);
            let mut random = RandomAgent::derived(seed, 2);
            if seed % 2 == 0 {
                play_game(&mut state, net.as_mut(), &mut random);
            } else {
                play_game(&mut state, &mut random, net.as_mut());
            }
            assert!(state.outcome.is_over());
        }
    }
}

/// Same seed, same game; and across seeds a random-init policy is mixed enough that the
/// sampler does not always pick the same opening — the difference from `netpolicy`.
#[test]
fn rnad_netsample_is_deterministic_under_its_seed_and_actually_samples() {
    let spec = AgentSpec::NetSample { checkpoint: sampling_checkpoint(), raw: true };
    let state = GameState::new(GameConfig::default(), 3);
    let legal = state.legal_actions();

    let pick = |seed: u64| spec.build(seed, 5).choose(&state, &legal);
    assert_eq!(pick(1), pick(1));

    let distinct: std::collections::HashSet<String> =
        (0..64).map(|s| pick(s).to_string()).collect();
    assert!(distinct.len() > 1, "64 seeds all chose the same opening action");
    assert_eq!(state.acting_player(), Player::P0);
}
