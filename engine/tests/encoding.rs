//! Phase 3: the observation encoder, the action encoding, and the reference network.
//!
//! Naming follows `CLAUDE.md` — a test that pins a ruling in `game_rules.md` is named for
//! its section, and a test about *machinery* carries a phase prefix instead, so a failure
//! reads as "the encoder broke" rather than "the rules broke".
//!
//! # The test that matters
//!
//! [`phase3_observation_is_a_function_of_the_information_set`] is the Phase 3 analogue of
//! Phase 2's `phase2_no_agent_reads_hidden_information`, and it is load-bearing for the same
//! reason. The encoder is handed engine-side ground truth, because it has to be — the state
//! *is* the ground truth — so nothing structural stops it reading the opponent's hand into
//! a float. A determinized world is in the same information set by construction, so an
//! honest encoder must produce **bit-identical** output from either. That is an exact
//! assertion, not a statistical one, and every leak this encoder could have shows up in it.

mod common;

use common::sample_positions;
use duel52_engine::encode::{
    action_blocks, action_dim, action_layout_hash, decode_action, encode_action,
    encode_observation, legal_mask, obs_dim, obs_layout_hash, slot_features,
};
use duel52_engine::nn::{Arch, Evaluator, MlpEvaluator, Weights};
use duel52_engine::testkit::Position;
use duel52_engine::{
    Action, AgentSpec, GameConfig, GameState, Player, Rank, Rng, Variant,
};

/// Encode one position, returning the buffer.
fn encode(state: &GameState, observer: Player) -> Vec<f32> {
    let mut out = vec![0.0f32; obs_dim(&state.config)];
    encode_observation(state, observer, &mut out);
    out
}

/// Index of the first float of a board slot, and the slot block's width.
fn slot_span(state: &GameState, lane: usize, side: usize, slot: usize) -> (usize, usize) {
    let f = slot_features(&state.config);
    let s = state.config.encoding_slots;
    (((lane * 2 + side) * s + slot) * f, f)
}

// ============================================================== information hiding ==

/// **The load-bearing test.** `DESIGN.md` §5–6: the observation must be a function of the
/// observer's information set and nothing else.
///
/// [`GameState::determinize`] resamples exactly the fields the observer is not entitled to
/// know — the opponent's hand, unread face-down ranks including the observer's *own* base
/// cards (`game_rules.md` §3), unknown draw-pile positions, and the removed-unseen pool. So
/// if the encoded tensor moves at all, some hidden field reached a float.
///
/// Exact f32 equality is deliberate. A tolerance would hide precisely the small leaks that
/// are hardest to find by reading the code — a belief count computed from ground truth, a
/// normaliser that happens to divide by a hidden quantity.
#[test]
fn phase3_observation_is_a_function_of_the_information_set() {
    let mut rng = Rng::new(0x0B5E_04E5_0000_0001);
    let mut checked = 0;
    for state in sample_positions() {
        for observer in Player::BOTH {
            let real = encode(&state, observer);
            for _ in 0..3 {
                let world = state.determinize(observer, &mut rng);
                let sampled = encode(&world, observer);
                if real != sampled {
                    let at = real
                        .iter()
                        .zip(&sampled)
                        .position(|(a, b)| a != b)
                        .expect("the vectors differ, so some index differs");
                    panic!(
                        "the observation for {observer} moved at float {at} \
                         ({} → {}) when only hidden information was resampled — \
                         the encoder is leaking. {}",
                        real[at],
                        sampled[at],
                        state.header()
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 200, "only {checked} observations were actually checked");
}

/// The targeted version of the test above. It is implied by that one, but it names the bug
/// directly: a rank the observer may not read must contribute *nothing* — not a smoothed
/// prior, not a reserved index — and `rank_unknown` must be the only thing set.
///
/// `game_rules.md` §3 and §4 decide who knows what: a card played face-down from hand is
/// known to its owner, a **base card is known to nobody including its owner**, a face-up
/// card is known to both, and a 4's Foresight sets one private bit.
#[test]
fn phase3_observation_never_reveals_hidden_ranks() {
    let mut hidden_seen = 0;
    for state in sample_positions() {
        for observer in Player::BOTH {
            let obs = encode(&state, observer);
            let r = state.config.rank_count();
            for lane in 0..state.config.lanes {
                for (side, owner) in [observer, observer.other()].into_iter().enumerate() {
                    for (slot, card) in state.lanes[lane].side(owner).iter().enumerate() {
                        let (base, _) = slot_span(&state, lane, side, slot);
                        // Layout: [occupied][rank one-hot × r][rank_unknown] …
                        let one_hot = &obs[base + 1..base + 1 + r];
                        let unknown = obs[base + 1 + r];
                        if card.rank_known_to(observer) {
                            assert_eq!(unknown, 0.0);
                            assert_eq!(one_hot.iter().sum::<f32>(), 1.0);
                            assert_eq!(one_hot[card.rank.index()], 1.0);
                        } else {
                            hidden_seen += 1;
                            assert_eq!(unknown, 1.0);
                            assert!(
                                one_hot.iter().all(|&v| v == 0.0),
                                "lane {lane} slot {slot} is hidden from {observer} but its \
                                 rank one-hot is not empty"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        hidden_seen > 100,
        "only {hidden_seen} hidden cards appeared, so the check proved little"
    );
}

/// A base card is hidden from **its owner too** (`game_rules.md` §3) — the first of
/// `CLAUDE.md`'s five easy-to-get-wrong facts, and the one an encoder is most likely to
/// break by reaching for `card.rank` on "my own" side.
#[test]
fn rule_3_the_encoder_hides_a_base_card_from_its_owner() {
    let state = GameState::new(GameConfig::default(), 7);
    let r = state.config.rank_count();
    for observer in Player::BOTH {
        let obs = encode(&state, observer);
        for lane in 0..state.config.lanes {
            for (side, owner) in [observer, observer.other()].into_iter().enumerate() {
                for (slot, card) in state.lanes[lane].side(owner).iter().enumerate() {
                    assert!(card.is_base, "a fresh deal holds nothing but base cards");
                    let (base, _) = slot_span(&state, lane, side, slot);
                    assert_eq!(obs[base + 1 + r], 1.0, "base card is readable by {observer}");
                    assert!(obs[base + 1..base + 1 + r].iter().all(|&v| v == 0.0));
                }
            }
        }
    }
}

// ================================================================ shape and determinism ==

/// Same state, same bytes. The encoder must have no hidden state, no iteration-order
/// dependence, and no dependence on how the position was reached.
#[test]
fn phase3_encoder_is_deterministic() {
    for state in sample_positions().into_iter().take(80) {
        for observer in Player::BOTH {
            let a = encode(&state, observer);
            let b = encode(&state, observer);
            assert_eq!(a, b);
            // A clone is a different allocation reached by a different path.
            let c = encode(&state.clone(), observer);
            assert_eq!(a, c);
        }
    }
}

/// The board's side axis is `[observer, opponent]`, so the two players' views of one
/// position are the same tensor with the sides swapped. Getting this backwards would train
/// a network that plays one seat well and the other badly.
#[test]
fn phase3_the_board_block_is_written_from_the_observers_side() {
    for state in sample_positions().into_iter().take(60) {
        let p0 = encode(&state, Player::P0);
        let p1 = encode(&state, Player::P1);
        let r = state.config.rank_count();
        for lane in 0..state.config.lanes {
            for slot in 0..state.lanes[lane].side(Player::P0).len() {
                // P0's own side is axis 0 in P0's view and axis 1 in P1's view.
                let (a, f) = slot_span(&state, lane, 0, slot);
                let (b, _) = slot_span(&state, lane, 1, slot);
                // Three of the features are observer-dependent by design and must be
                // excluded: the rank one-hot and `rank_unknown` (who may read the card —
                // `game_rules.md` §3, §4) and `is_mine` (which seat it belongs to). What
                // remains — `face_up` through `paired` — is public (§5), so it has to be
                // byte-identical in both views.
                let public = 1 + r + 1;
                assert_eq!(
                    p0[a + public..a + f - 1],
                    p1[b + public..b + f - 1],
                    "public features of lane {lane} slot {slot} differ between the two views"
                );
            }
        }
    }
}

/// The encoder asserts rather than truncating, and the message has to name the config key,
/// because the person who hits this is running a training job and needs to know which knob
/// to turn (`FINDINGS.md` F2.7).
#[test]
fn phase3_encoding_slot_bound_asserts() {
    let mut config = GameConfig::default();
    config.encoding_slots = 4;
    let mut position = Position::new(config);
    for _ in 0..5 {
        position.face_up(0, Player::P0, Rank::SEVEN);
    }
    let state = position.build();

    let mut out = vec![0.0f32; obs_dim(&config)];
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_observation(&state, Player::P0, &mut out)
    }))
    .expect_err("five cards must not fit in four encoding slots");
    let message = panic
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()).unwrap());
    assert!(
        message.contains("encoding_slots"),
        "the panic must name the config key to raise, got: {message}"
    );
}

// ===================================================================== action encoding ==

/// `legal.rs` enumerates pairs with `slot_a < slot_b`, so the canonicalisation in
/// [`encode_action`] never has to reorder an action the engine produced.
///
/// Worth pinning separately: if enumeration ever emitted both orders, two distinct actions
/// would land on one logit and the round-trip test below would fail with a confusing
/// message about decode rather than a clear one about enumeration.
#[test]
fn phase3_legal_pairs_are_already_canonical() {
    let mut seen = 0;
    for state in sample_positions() {
        for action in state.legal_actions() {
            if let Action::DeclarePair { slot_a, slot_b, .. } = action {
                assert!(slot_a < slot_b, "legal.rs emitted a pair as ({slot_a}, {slot_b})");
                seen += 1;
            }
        }
    }
    assert!(seen > 0, "no position in the sample offered a pair");
}

/// Every legal action encodes to a distinct index and decodes back to itself.
///
/// Injectivity is the property that matters. `DESIGN.md` §4's original rank-keyed encoding
/// failed it — two same-rank face-down cards with different damage share a `FLIP(rank,
/// lane)` — and in an AlphaZero loop a collision is not merely lossy: the policy target is a
/// visit distribution over engine actions, so two actions on one logit force an invented
/// rule for folding their visits and another for which one to play.
#[test]
fn phase3_action_encoding_round_trips() {
    let mut kinds = std::collections::HashSet::new();
    let mut total = 0;
    for state in sample_positions() {
        let mut seen = std::collections::HashMap::new();
        for action in state.legal_actions() {
            let index = encode_action(&action, &state);
            assert!(
                index < action_dim(&state.config),
                "{action} encoded to {index}, outside the policy head"
            );
            if let Some(other) = seen.insert(index, action) {
                panic!("`{action}` and `{other}` both encode to index {index}");
            }
            let back = decode_action(index, &state)
                .unwrap_or_else(|| panic!("index {index} for `{action}` decoded to nothing"));
            assert_eq!(back, action, "round trip changed the action at index {index}");
            kinds.insert(std::mem::discriminant(&action));
            total += 1;
        }
    }
    assert!(total > 3_000, "only {total} actions were checked");
    assert_eq!(
        kinds.len(),
        10,
        "the sample did not exercise all ten Action variants, so the round trip is untested \
         for some of them"
    );
}

/// The mask agrees with the engine in both directions: one true entry per legal action, and
/// every true entry decoding to an action the engine accepts.
///
/// `CLAUDE.md`: the engine is the sole authority on legality. The mask is built from
/// [`GameState::legal_actions`] for exactly that reason — a second copy of the rules living
/// in the mask would present as a policy that occasionally proposes an illegal move, at
/// which point the natural suspect is the network rather than the mask.
#[test]
fn phase3_mask_matches_legal_actions() {
    for state in sample_positions() {
        let legal = state.legal_actions();
        let mut mask = vec![false; action_dim(&state.config)];
        legal_mask(&state, &mut mask);

        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            legal.len(),
            "the mask has a different number of entries than there are legal actions at {}",
            state.header()
        );
        for (index, _) in mask.iter().enumerate().filter(|(_, &m)| m) {
            let action = decode_action(index, &state)
                .unwrap_or_else(|| panic!("masked index {index} decodes to nothing"));
            assert!(
                state.is_legal(action),
                "the mask allows `{action}`, which the engine rejects"
            );
        }
    }
}

/// A terminal position offers no actions, so the mask must be empty rather than defaulting
/// to "everything" — a policy sampled from an all-true mask in a finished game would look
/// like a rules bug at the far end of a training run.
#[test]
fn phase3_a_finished_game_masks_nothing() {
    let mut state = GameState::new(GameConfig::default(), 3);
    let mut rng = Rng::new(3);
    while !state.outcome.is_over() {
        let legal = state.legal_actions();
        state.apply_trusted(*rng.choose(&legal).unwrap());
    }
    let mut mask = vec![false; action_dim(&state.config)];
    legal_mask(&state, &mut mask);
    assert!(mask.iter().all(|&m| !m));
}

/// The block table in `encode.rs` is the thing `action_layout_hash` commits to, so it has to
/// describe the head that actually exists.
#[test]
fn phase3_action_blocks_tile_the_policy_head() {
    for variant in Variant::ALL {
        let config = GameConfig::preset(variant);
        let blocks = action_blocks(&config);
        let mut at = 0;
        for b in &blocks {
            assert_eq!(b.offset, at, "block {} does not start where the last one ended", b.name);
            at += b.len;
        }
        assert_eq!(at, action_dim(&config));
    }
}

// ================================================================= the reference network ==

fn test_arch(config: &GameConfig) -> Arch {
    // Small enough for the `opt-level = 1` test profile; the shape, not the capacity, is
    // what these tests are about.
    Arch {
        obs_dim: obs_dim(config),
        action_dim: action_dim(config),
        width: 32,
        blocks: 2,
        value_hidden: 16,
    }
}

/// f32 throughout, a fixed accumulation order, and no parallelism inside the reference
/// forward pass — so the same weights and the same input give bit-identical output, whatever
/// else the process is doing.
#[test]
fn phase3_forward_pass_is_deterministic() {
    let config = GameConfig::default();
    let arch = test_arch(&config);
    let weights = Weights::random(1234, arch);
    let evaluator = MlpEvaluator::new(weights);

    let state = GameState::new(config, 11);
    let obs = encode(&state, Player::P0);

    let run = || {
        let mut logits = vec![0.0f32; arch.action_dim];
        let mut values = vec![0.0f32; 1];
        evaluator.eval_batch(&obs, 1, &mut logits, &mut values);
        (logits, values)
    };
    let first = run();
    for _ in 0..3 {
        assert_eq!(run(), first, "the forward pass is not reproducible");
    }

    // Across threads, too: nothing in the evaluator may depend on where it runs.
    let evaluator = std::sync::Arc::new(evaluator);
    let obs = std::sync::Arc::new(obs);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let evaluator = evaluator.clone();
            let obs = obs.clone();
            let action_dim = arch.action_dim;
            std::thread::spawn(move || {
                let mut logits = vec![0.0f32; action_dim];
                let mut values = vec![0.0f32; 1];
                evaluator.eval_batch(&obs, 1, &mut logits, &mut values);
                (logits, values)
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), first);
    }
}

/// The batch interface is what the self-play loop will use (`PHASE3_STEP1.md` §1.6: G games
/// in flight, one simulation each, evaluated as a single batch), so a row's result must not
/// depend on what else is in the batch with it.
#[test]
fn phase3_batched_evaluation_matches_row_by_row() {
    let config = GameConfig::default();
    let arch = test_arch(&config);
    let evaluator = MlpEvaluator::new(Weights::random(99, arch));

    let states: Vec<GameState> = (0..5).map(|s| GameState::new(config, s)).collect();
    let rows: Vec<Vec<f32>> = states.iter().map(|s| encode(s, s.acting_player())).collect();

    let mut batch = Vec::new();
    for row in &rows {
        batch.extend_from_slice(row);
    }
    let n = rows.len();
    let mut logits = vec![0.0f32; n * arch.action_dim];
    let mut values = vec![0.0f32; n];
    evaluator.eval_batch(&batch, n, &mut logits, &mut values);

    for (i, row) in rows.iter().enumerate() {
        let mut one_logits = vec![0.0f32; arch.action_dim];
        let mut one_value = vec![0.0f32; 1];
        evaluator.eval_batch(row, 1, &mut one_logits, &mut one_value);
        assert_eq!(&logits[i * arch.action_dim..(i + 1) * arch.action_dim], &one_logits[..]);
        assert_eq!(values[i], one_value[0]);
    }
}

/// The value head is a `tanh`, so it is bounded — and a random-init network must not be
/// saturated, or the first training steps would have no gradient to work with.
#[test]
fn phase3_value_head_is_bounded() {
    let config = GameConfig::default();
    let arch = test_arch(&config);
    let evaluator = MlpEvaluator::new(Weights::random(7, arch));
    for seed in 0..8u64 {
        let state = GameState::new(config, seed);
        let obs = encode(&state, Player::P0);
        let mut logits = vec![0.0f32; arch.action_dim];
        let mut value = vec![0.0f32; 1];
        evaluator.eval_batch(&obs, 1, &mut logits, &mut value);
        assert!(value[0] > -1.0 && value[0] < 1.0, "value {} is saturated", value[0]);
        assert!(logits.iter().all(|v| v.is_finite()), "the policy head produced a non-finite logit");
    }
}

// ================================================================== checkpoint round trip ==

/// A checkpoint written by the engine and read back by the engine must be the same weights,
/// bit for bit. This is the half of the format the Rust side owns; `py/tests/test_parity.py`
/// checks that Python agrees about the other half.
#[test]
fn phase3_checkpoint_round_trips_through_the_file_format() {
    let config = GameConfig::default();
    let arch = test_arch(&config);
    let weights = Weights::random(2026, arch);

    let path = std::env::temp_dir().join(format!(
        "duel52-roundtrip-{}-{:?}.d52nn",
        std::process::id(),
        std::thread::current().id()
    ));
    weights.save(&path, &config).expect("save");
    let loaded = Weights::load(&path, &config).expect("load");
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.arch, weights.arch);
    assert_eq!(loaded.params, weights.params);
}

/// **The reason the format carries hashes at all.** Silent layout drift between the function
/// that was trained and the function that is evaluated is the nightmare failure mode of this
/// phase: nothing crashes, the agent is merely bad, and the natural suspect is the training
/// run. A refused load turns that into a one-line error.
#[test]
fn phase3_a_checkpoint_from_a_different_layout_is_refused() {
    let config = GameConfig::default();
    let arch = test_arch(&config);
    let path = std::env::temp_dir().join(format!("duel52-drift-{}.d52nn", std::process::id()));
    Weights::random(5, arch).save(&path, &config).expect("save");

    let mut narrower = config;
    narrower.encoding_slots = 12;
    let error = Weights::load(&path, &narrower).expect_err("a moved layout must be refused");
    std::fs::remove_file(&path).ok();
    assert!(
        error.contains("obs_layout_hash") || error.contains("obs_dim"),
        "the error must name the field that moved, got: {error}"
    );
}

/// The hashes have to be reachable from Python without reimplementing them, which is what
/// `duel52.encoding_spec()` is for. Here we only pin that they are stable across calls and
/// distinct from each other — a copy-paste that hashed the observation string twice would
/// otherwise leave the action layout unprotected.
#[test]
fn phase3_layout_hashes_are_stable_and_distinct() {
    let config = GameConfig::default();
    assert_eq!(obs_layout_hash(&config), obs_layout_hash(&config));
    assert_eq!(action_layout_hash(&config), action_layout_hash(&config));
    assert_ne!(obs_layout_hash(&config), action_layout_hash(&config));
}

// ======================================================================= the smoke agent ==

/// A random-init policy played by argmax finishes legal games in every variant.
///
/// Nothing here is a claim about strength — a random network played deterministically may
/// well lose to random, and `PHASE3_STEP1.md` §3 says so explicitly. What is being checked
/// is that the whole path holds together under real play: encode, forward, mask, argmax,
/// decode, apply.
#[test]
fn phase3_netpolicy_plays_legal_games() {
    let dir = std::env::temp_dir();
    for variant in Variant::ALL {
        let config = GameConfig::preset(variant);
        let arch = test_arch(&config);
        let path = dir.join(format!(
            "duel52-netpolicy-{}-{}.d52nn",
            std::process::id(),
            variant.label()
        ));
        Weights::random(4242, arch).save(&path, &config).expect("save");

        let spec = AgentSpec::parse(&format!("netpolicy:{}", path.display())).expect("parse");
        for seed in 0..3u64 {
            let mut state = GameState::new(config, seed);
            let mut net = spec.build(seed, 0);
            let mut opponent = AgentSpec::Random.build(seed, 1);
            duel52_engine::agents::play_game(&mut state, &mut *net, &mut *opponent);
            assert!(state.outcome.is_over(), "netpolicy left a {variant} game unfinished");
        }
        std::fs::remove_file(&path).ok();
    }
}

/// `netpolicy:<path>` must survive a path with capitals in it. [`AgentSpec::parse`]
/// lowercases the family name to accept `Greedy` as well as `greedy`, and lowercasing the
/// whole string would have quietly broken every checkpoint under a capitalised directory.
#[test]
fn phase3_netpolicy_spec_preserves_the_checkpoint_path() {
    let spec = AgentSpec::parse("netpolicy:/Users/Someone/Runs/Init.d52nn").expect("parse");
    assert_eq!(spec.name(), "netpolicy:/Users/Someone/Runs/Init.d52nn");
    assert_eq!(AgentSpec::parse(&spec.name()), Ok(spec));
}
