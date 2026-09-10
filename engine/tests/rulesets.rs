//! Structural invariants that must hold **whatever the card rules are**.
//!
//! `MODULAR_RULES.md` §8. This is the piece that makes rule modding safe to do quickly, and
//! it is the difference between a mod system and a way to generate broken games.
//!
//! # The registry is the directory
//!
//! Every `.toml` in `configs/rules/` is enumerated here. There is no list to add a ruleset
//! to — the file existing is the registration (`MODULAR_RULES.md` §5d). At twenty rulesets a
//! hand-maintained list is a list you forget to update, and the one you forget is the one
//! that fails to terminate inside a 24-hour run.
//!
//! The three shipped variants are included too, so `base`, `split` and `mirrored` are covered
//! by the same invariants rather than by three separate ad-hoc tests.
//!
//! # Why these particular invariants
//!
//! Each is a property the *engine* relies on, not a property of any ruleset. A power variant
//! that breaks one does not produce a differently-balanced game; it produces a game the rest
//! of the codebase silently lies about. The termination one is the sharpest: `game_rules.md`
//! §7's finiteness argument is a proof about *specific rules* — "powers fire on flip, a King
//! reactivates once, and nothing ever turns a card face-down again, so total power
//! activations are bounded" — and a variant that turned a card face-down would break the
//! proof while still passing every rules test.
//!
//! ⚠️ Keep the per-ruleset game counts low. This is a cross product of rulesets and
//! invariants, so it is the one suite that grows with the registry. These are structural
//! properties: a leak or a non-termination shows up in tens of games, not thousands. If it
//! gets slow, cut games per ruleset before cutting rulesets.

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use duel52_engine::agents::AgentSpec;
use duel52_engine::outcome::{DrawReason, Outcome};
use duel52_engine::rank::Rank;
use duel52_engine::{GameConfig, GameState, Player, Rng, Variant};

/// Games per ruleset for the behavioural invariants. Deliberately small — see the module
/// note about the cross product.
const GAMES: u64 = 24;

/// Every registered ruleset: the files in `configs/rules/`, plus the three shipped variants.
fn registry() -> Vec<(String, GameConfig)> {
    let mut out: Vec<(String, GameConfig)> = Variant::ALL
        .into_iter()
        .map(|v| (format!("variant:{}", v.label()), GameConfig::preset(v)))
        .collect();

    let dir = rules_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read the ruleset registry `{}`: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "the ruleset registry at `{}` is empty — this suite would pass vacuously",
        dir.display()
    );

    for path in files {
        let config = GameConfig::from_config_file(&path)
            .unwrap_or_else(|e| panic!("registered ruleset `{}` does not load: {e}", path.display()));
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        out.push((format!("rules/{name}"), config));
    }
    out
}

fn rules_dir() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is `engine/`, and the registry is a sibling of it.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the engine crate has a parent directory")
        .join("configs/rules")
}

/// Play one game out with a fixed agent, returning the finished state.
fn play_out(config: GameConfig, seed: u64) -> GameState {
    let mut state = GameState::new(config, seed);
    let mut rng = Rng::derive(seed, 0x5EED_0F_2011_E5E7);
    let mut guard = 0usize;
    while !state.outcome.is_over() {
        guard += 1;
        assert!(
            guard < 200_000,
            "a game made 200k decisions without finishing — `legal_actions` is probably \
             empty-but-not-over, which the invariant below is meant to catch first"
        );
        let legal = state.legal_actions();
        let action = *rng.choose(&legal).expect("a running game has actions");
        state.apply_trusted(action);
    }
    state
}

// ============================================================ registry hygiene ==

/// Every ruleset must validate, name itself, and be a *distinct game*.
///
/// The distinctness check is the one that earns its keep: two files that hash the same are
/// the same ruleset under two names, which means one of them does not do what its comment
/// says. `jack-2hp` is the case worth keeping in mind — it changes no card *power*, so a
/// hash that ignored the Tier-1 numbers would collide it with canonical and nothing else
/// would notice.
#[test]
fn every_registered_ruleset_is_valid_and_distinct() {
    let all = registry();
    let mut by_hash: Vec<(u64, String)> = Vec::new();

    for (name, config) in &all {
        config
            .validate()
            .unwrap_or_else(|e| panic!("{name} does not validate: {e}"));

        // Every power must sit on its own rank. `validate` checks it, but assert here too so
        // the failure names the ruleset.
        for rank in Rank::ALL {
            let power = config.power(rank);
            assert_eq!(power.rank(), rank, "{name}: {power} is not a {rank} power");
        }

        let h = config.rules_hash();
        if let Some((_, other)) = by_hash.iter().find(|(oh, _)| *oh == h) {
            // The three shipped variants are also reachable through the registry files, so a
            // file that resolves to exactly `variant:split` is expected rather than a bug.
            let expected_alias = name == "rules/canonical" && other == "variant:split";
            assert!(
                expected_alias,
                "{name} and {other} hash to the same ruleset ({h:016x}) — they describe the \
                 same game, so one of them is not doing what it says"
            );
        } else {
            by_hash.push((h, name.clone()));
        }
    }
}

/// A ruleset's identity must survive a round trip through the string that gets stamped into
/// every shard and game record. If it does not, an artifact says it was played under rules
/// that cannot be reconstructed.
#[test]
fn every_registered_ruleset_round_trips_through_its_config_string() {
    for (name, config) in registry() {
        let text = config.to_config_string();
        let back = GameConfig::from_config_str(&text)
            .unwrap_or_else(|e| panic!("{name}: its own config string does not parse: {e}"));
        assert_eq!(back, config, "{name}: config string round trip changed a field");
        assert_eq!(
            back.rules_hash(),
            config.rules_hash(),
            "{name}: rules_hash moved across a round trip"
        );
        assert_eq!(back.rules_name, config.rules_name, "{name}: name moved");
    }
}

/// The canonical ruleset must hash to exactly what the shipped variants hash to.
///
/// This is what makes the `PowerId` and Tier-1 refactor a *no-op* for provenance: naming
/// thirteen powers and eleven constants must not have changed what the default game is. If
/// this fails, every `FINDINGS.md` number written before the mod system silently refers to a
/// different ruleset than the one the engine now plays.
#[test]
fn the_canonical_ruleset_is_the_shipped_default() {
    let split = GameConfig::split_deck();
    assert!(split.is_canonical_rules());
    assert_eq!(split.rules_name.as_str(), "canonical-2026-09");
    assert!(
        split.power_diff().is_empty(),
        "the default config has a modded card: {:?}",
        split.power_diff()
    );
    let from_file = GameConfig::from_config_file(&rules_dir().join("canonical.toml"))
        .expect("canonical.toml loads");
    assert_eq!(
        from_file.rules_hash(),
        split.rules_hash(),
        "configs/rules/canonical.toml is not the shipped default"
    );
}

// ==================================================== structural invariants ==

/// **Determinism.** Same seed + same config → identical game, in every ruleset.
///
/// `CLAUDE.md`: "Everything is seeded and deterministic. Non-reproducible results are bugs."
/// A power that read wall-clock time or iterated a `HashMap` would break this and nothing
/// else.
#[test]
fn every_ruleset_is_deterministic() {
    for (name, config) in registry() {
        for seed in 0..6u64 {
            let a = play_out(config, seed);
            let b = play_out(config, seed);
            assert_eq!(a.outcome, b.outcome, "{name} seed {seed}: outcome differs");
            assert_eq!(a.ply, b.ply, "{name} seed {seed}: length differs");
            assert!(a == b, "{name} seed {seed}: final states differ");
        }
    }
}

/// **Termination.** No ruleset may reach `max_plies`.
///
/// `game_rules.md` §7 proves the game finite from *specific rules*. `max_plies` exists so a
/// rules bug degrades into a logged draw rather than an infinite loop — which means a
/// `PlyLimit` draw is a **bug report**, and this suite treats it as a test failure rather
/// than as a result (`MODULAR_RULES.md` §8).
///
/// The damage cascade has its own bound and panics rather than looping, so a death-trigger
/// cycle fails here loudly with a stack trace.
#[test]
fn no_ruleset_reaches_the_ply_cap() {
    for (name, config) in registry() {
        for seed in 0..GAMES {
            let state = play_out(config, seed);
            assert_ne!(
                state.outcome,
                Outcome::Draw(DrawReason::PlyLimit),
                "{name} seed {seed} hit the ply cap after {} plies. That is a bug report, not \
                 a draw: `game_rules.md` §7's finiteness argument does not cover this ruleset.",
                state.ply
            );
        }
    }
}

/// **`legal_actions()` is empty exactly when the game is over.**
///
/// Every caller leans on this — the search, the CLI, the record walker. A power that could
/// leave a player with nothing legal to do, without ending the turn, would strand them.
#[test]
fn every_ruleset_always_offers_a_move_until_the_game_ends() {
    for (name, config) in registry() {
        for seed in 0..8u64 {
            let mut state = GameState::new(config, seed);
            let mut rng = Rng::derive(seed, 0x1E6A_1AC7_0000_0001);
            while !state.outcome.is_over() {
                let legal = state.legal_actions();
                assert!(
                    !legal.is_empty(),
                    "{name} seed {seed}: no legal action at ply {} but the game is not over",
                    state.ply
                );
                let action = *rng.choose(&legal).expect("checked non-empty");
                state.apply_trusted(action);
            }
            assert!(
                state.legal_actions().is_empty(),
                "{name} seed {seed}: the game is over but actions are still offered"
            );
        }
    }
}

/// **Card census.** Nothing is lost or duplicated, whatever the powers do.
///
/// The 3's Trap is the reason this matters: it is the one canonical rule that takes a card
/// off the path to the discard pile, and a vengeance variant adds a second effect to the
/// same moment. A death trigger that forgot to either kill or restore its card would leave
/// it in play at zero hit points, or drop it entirely.
#[test]
fn every_ruleset_conserves_cards() {
    for (name, config) in registry() {
        for seed in 0..GAMES {
            let state = play_out(config, seed);
            let deck_total = if config.variant.is_split() {
                2 * config.split_deck_size()
            } else {
                config.full_deck_size()
            };

            let on_table: usize = state
                .lanes
                .iter()
                .map(|l| l.sides[0].len() + l.sides[1].len())
                .sum();
            let counted = on_table
                + state.hands[0].len()
                + state.hands[1].len()
                + state.piles[0].len()
                + state.piles[1].len()
                + state.discards[0].len()
                + state.discards[1].len()
                + state.removed[0].len()
                + state.removed[1].len();

            assert_eq!(
                counted, deck_total,
                "{name} seed {seed}: {counted} cards accounted for, deck holds {deck_total}"
            );
        }
    }
}

/// **No card outlives its hit points.**
///
/// The engine's own `debug_check_invariants` asserts this after every action in debug
/// builds, so this test is really about making sure every ruleset is actually *exercised*
/// under those assertions rather than only the default one.
#[test]
fn every_ruleset_leaves_no_dead_card_standing() {
    for (name, config) in registry() {
        for seed in 0..GAMES {
            let state = play_out(config, seed);
            for (l, lane) in state.lanes.iter().enumerate() {
                for p in Player::BOTH {
                    for card in lane.side(p) {
                        assert!(
                            !card.is_dead(&state.config),
                            "{name} seed {seed}: {} at lane {l} has {} damage of {} max HP",
                            card.rank,
                            card.damage,
                            card.max_hp(&state.config)
                        );
                    }
                }
            }
        }
    }
}

/// **The observation is a function of the information set, in every ruleset.**
///
/// `CLAUDE.md`'s structural rule, generalised. A power whose effect was visible in the
/// tensor but not in the information set would leak hidden information into the network
/// without anything crashing — the agent would simply learn something it must not know.
#[test]
fn every_ruleset_keeps_the_observation_a_function_of_the_information_set() {
    use duel52_engine::encode::{encode_observation, obs_dim};

    for (name, config) in registry() {
        let mut real = vec![0.0f32; obs_dim(&config)];
        let mut sampled = vec![0.0f32; obs_dim(&config)];
        for seed in 0..4u64 {
            let mut state = GameState::new(config, seed);
            let mut rng = Rng::derive(seed, 0xDE7E_2411_2E00_0001);
            let mut steps = 0;
            while !state.outcome.is_over() && steps < 90 {
                steps += 1;
                let observer = state.acting_player();
                encode_observation(&state, observer, &mut real);
                let world = state.determinize(observer, &mut rng);
                encode_observation(&world, observer, &mut sampled);
                assert_eq!(
                    real, sampled,
                    "{name} seed {seed} step {steps}: the observation distinguishes two \
                     worlds in the same information set — a rules mod is leaking"
                );
                let legal = state.legal_actions();
                let action = *rng.choose(&legal).expect("a running game has actions");
                state.apply_trusted(action);
            }
        }
    }
}

/// **No agent reads hidden information, in every ruleset.**
///
/// The same guard as `phase2_no_agent_reads_hidden_information`, run across the registry.
/// A power that made an agent's evaluation depend on a card it cannot see would show up
/// here as a disagreement between the real state and a world in the same information set.
///
/// Restricted to the deterministic hand-written rungs: a search agent's answer legitimately
/// depends on its RNG stream, and the existing test handles that separately.
#[test]
fn every_ruleset_keeps_agents_honest() {
    for (name, config) in registry() {
        for spec in [AgentSpec::Greedy] {
            for seed in 0..4u64 {
                let mut state = GameState::new(config, seed);
                let mut rng = Rng::derive(seed, 0x404E_5700_0000_0001);
                let mut steps = 0;
                while !state.outcome.is_over() && steps < 60 {
                    steps += 1;
                    let observer = state.acting_player();
                    let legal = state.legal_actions();
                    let world = state.determinize(observer, &mut rng);

                    let from_real = spec.build(seed, 1).choose(&state, &legal);
                    let from_world = spec.build(seed, 1).choose(&world, &legal);
                    assert_eq!(
                        from_real, from_world,
                        "{name} seed {seed} step {steps}: {} answers differently in two \
                         worlds of the same information set — it is reading hidden state",
                        spec.name()
                    );

                    let legal = state.legal_actions();
                    let action = *rng.choose(&legal).expect("a running game has actions");
                    state.apply_trusted(action);
                }
            }
        }
    }
}

/// **The layout never moves with the rules.**
///
/// `MODULAR_RULES.md` §1b, and the single most valuable property the project has: the
/// encoder is rank-agnostic, so no ruleset changes `obs_dim`, `action_dim`, or either layout
/// hash. That is what lets `--init-from` warm-start a rules experiment from the current
/// champion and turns a 24-hour run into a 3-hour one.
///
/// If this fails, the mod system just became an order of magnitude more expensive, and the
/// place to look is a new per-card feature in `encode.rs`.
#[test]
fn no_ruleset_moves_the_encoder_layout() {
    use duel52_engine::encode::{action_dim, action_layout_hash, obs_dim, obs_layout_hash};

    let mut seen: BTreeSet<(usize, usize, u64, u64)> = BTreeSet::new();
    let mut witness: Vec<String> = Vec::new();
    for (name, config) in registry() {
        // Only rulesets that agree on `encoding_slots` are comparable: that field sizes the
        // tensor and is deliberately *not* part of `rules_hash`.
        if config.encoding_slots != GameConfig::default().encoding_slots {
            continue;
        }
        seen.insert((
            obs_dim(&config),
            action_dim(&config),
            obs_layout_hash(&config),
            action_layout_hash(&config),
        ));
        witness.push(name);
    }
    assert_eq!(
        seen.len(),
        1,
        "the encoder layout differs across rulesets, so a checkpoint cannot warm-start \
         across them. Rulesets compared: {witness:?}; layouts seen: {seen:?}"
    );
}
