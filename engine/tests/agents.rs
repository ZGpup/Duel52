//! Phase 2: determinization, the agent ladder, and the Elo machinery.
//!
//! Naming follows `CLAUDE.md` — a test that pins a ruling in `game_rules.md` is named for
//! its section. The tests here that are about *machinery* rather than about a rule carry a
//! `phase2_` prefix instead, so a failure is immediately legible as "the ladder broke"
//! rather than "the rules broke".
//!
//! # The test that matters
//!
//! [`phase2_no_agent_reads_hidden_information`] is the reason this file exists. `DESIGN.md`
//! §6 warns that an agent handed the engine's ground-truth state can trivially cheat, and
//! until now the only thing stopping it was a comment. Because a determinized world is by
//! construction in the same information set as the real one, an honest agent given either
//! must return the same action — and that is an exact assertion, not a statistical one.

mod common;

use duel52_engine::agents::eval::{evaluate, EvalWeights, View};
use duel52_engine::testkit::*;
use duel52_engine::{
    elo, ladder, probe, Action, AgentSpec, GameConfig, GameState, IsmctsAgent, Outcome, Player,
    Player::P0, Player::P1, Rank, Rng, Variant,
};

/// Budgets small enough for the `opt-level = 1` test profile. The ladder's real budgets live
/// in [`AgentSpec::LADDER`]; nothing here is measuring strength, only correctness.
///
/// **Every agent must be in this list.** `PLAN.md` used to claim that adding an agent got it
/// covered by [`phase2_no_agent_reads_hidden_information`] automatically — it does not, and
/// never did: that test iterates this roster, not [`AgentSpec::LADDER`]. Adding a rung means
/// adding it here by hand. `PLAN.md` was corrected when Phase 3's `netpolicy` arrived.
///
/// A function rather than a `const` because `netpolicy` names a checkpoint file, which has
/// to exist before the roster can mention it.
fn test_roster() -> Vec<AgentSpec> {
    vec![
        AgentSpec::Random,
        AgentSpec::Greedy,
        AgentSpec::FlatMc { playouts: 24 },
        AgentSpec::Pimc {
            worlds: 2,
            depth: 1,
        },
        AgentSpec::Ismcts { iterations: 40 },
        AgentSpec::NetPolicy {
            checkpoint: test_checkpoint(),
        },
        AgentSpec::NetMcts {
            checkpoint: test_checkpoint(),
            sims: 24,
        },
    ]
}

/// A small random-init checkpoint, written once per test process.
///
/// `Weights::random` is what makes this possible without Python in the loop — see
/// `nn/weights.rs`. A test that needed `maturin develop` first would not run in plain
/// `cargo test`, and a test that does not run is not a test.
fn test_checkpoint() -> String {
    use std::sync::OnceLock;
    static PATH: OnceLock<String> = OnceLock::new();
    PATH.get_or_init(|| {
        let config = GameConfig::default();
        let arch = duel52_engine::nn::Arch {
            obs_dim: duel52_engine::encode::obs_dim(&config),
            action_dim: duel52_engine::encode::action_dim(&config),
            width: 24,
            blocks: 2,
            value_hidden: 12,
        };
        let path = std::env::temp_dir().join(format!("duel52-roster-{}.d52nn", std::process::id()));
        duel52_engine::nn::Weights::random(20260903, arch)
            .save(&path, &config)
            .expect("write the roster checkpoint");
        path.display().to_string()
    })
    .clone()
}

// `position_after` and `sample_positions` moved to `tests/common/` when the Phase 3 encoder
// tests needed the same sample — see `engine/tests/encoding.rs`.
use common::{position_after, sample_positions};

// ==================================================== determinization ==

/// `DESIGN.md` §6, and the property every search agent depends on: the actions available at
/// a decision node are a function of the acting player's *information set*, not of the
/// hidden cards.
///
/// This holds because every legality predicate reads a face-up rank (public), a slot
/// position (public), the acting player's own hand (known to them), or a global flag — and
/// never a hidden rank. If a future rule breaks that, PIMC and ISMCTS would start evaluating
/// actions that are not actually available, and this test is the tripwire.
#[test]
fn rule_6_determinization_preserves_the_legal_action_set() {
    let mut rng = Rng::new(4242);
    for state in sample_positions() {
        let observer = state.acting_player();
        let before = state.legal_actions();
        for _ in 0..4 {
            let world = state.determinize(observer, &mut rng);
            assert_eq!(
                world.legal_actions(),
                before,
                "determinization changed the legal actions at {}",
                state.header()
            );
            assert_eq!(world.phase(), state.phase());
            assert_eq!(world.acting_player(), state.acting_player());
        }
    }
}

/// `game_rules.md` §3, §4, §5: a determinized world must agree with the real one about
/// everything the observer is entitled to know — their own hand, every face-up rank, every
/// card's damage and position, the public discards, and the sizes of everything hidden.
#[test]
fn rule_4_determinization_preserves_everything_the_observer_knows() {
    let mut rng = Rng::new(99);
    for state in sample_positions() {
        for observer in Player::BOTH {
            let world = state.determinize(observer, &mut rng);

            assert_eq!(world.hands[observer.idx()], state.hands[observer.idx()]);
            assert_eq!(
                world.hands[observer.other().idx()].len(),
                state.hands[observer.other().idx()].len(),
                "hand size is public"
            );
            assert_eq!(world.discards, state.discards, "discards are public (§5)");
            assert_eq!(world.pending, state.pending);
            assert_eq!(world.ply, state.ply);
            assert_eq!(world.actions_remaining, state.actions_remaining);
            assert_eq!(world.base_unlocked, state.base_unlocked);
            assert_eq!(world.quiet_plies, state.quiet_plies);
            for i in 0..2 {
                assert_eq!(world.piles[i].len(), state.piles[i].len());
                assert_eq!(world.removed[i].len(), state.removed[i].len());
            }

            for lane in 0..state.lane_count() {
                for p in Player::BOTH {
                    let real = state.lanes[lane].side(p);
                    let sampled = world.lanes[lane].side(p);
                    assert_eq!(real.len(), sampled.len(), "occupancy is public");
                    for (r, s) in real.iter().zip(sampled) {
                        assert_eq!(r.id, s.id);
                        assert_eq!(r.face_up, s.face_up);
                        assert_eq!(r.damage, s.damage, "damage is public (§5)");
                        assert_eq!(r.is_base, s.is_base);
                        assert_eq!(r.entered_as_base, s.entered_as_base);
                        assert_eq!(r.frozen_until_ply, s.frozen_until_ply);
                        assert_eq!(r.pair_id, s.pair_id);
                        assert_eq!(r.attacks_used, s.attacks_used);
                        assert_eq!(r.attack_allowance, s.attack_allowance);
                        assert_eq!(r.known_to, s.known_to, "who knows what is not resampled");
                        if r.rank_known_to(observer) {
                            assert_eq!(r.rank, s.rank, "a rank the observer knows is fixed");
                        }
                    }
                }
            }

            // A bottomed card is a hard constraint on the sampled pile order, not something
            // to resample (`DESIGN.md` §6).
            for i in 0..2 {
                assert_eq!(
                    world.piles[i].known_from_bottom(observer),
                    state.piles[i].known_from_bottom(observer),
                    "a pile position the observer bottomed must keep its rank and place"
                );
            }
        }
    }
}

/// `game_rules.md` §9a: "Each player owns one colour — 26 cards, two suits, ranks A–K
/// twice." A sampled world has to be a *possible* world, so the deck composition must still
/// balance after resampling.
#[test]
fn rule_9a_determinization_respects_the_deck_composition() {
    let mut rng = Rng::new(7);
    for state in sample_positions() {
        for observer in Player::BOTH {
            let world = state.determinize(observer, &mut rng);
            assert_eq!(world.card_census(), world.expected_card_count());

            if world.config.variant.is_split() {
                for p in Player::BOTH {
                    let mut counts = [0u8; Rank::COUNT];
                    for r in &world.hands[p.idx()] {
                        counts[r.index()] += 1;
                    }
                    for r in world.piles[p.idx()].ranks() {
                        counts[r.index()] += 1;
                    }
                    for r in &world.removed[p.idx()] {
                        counts[r.index()] += 1;
                    }
                    for r in &world.discards[p.idx()] {
                        counts[r.index()] += 1;
                    }
                    for (_, _, c) in world.cards_of(p) {
                        counts[c.rank.index()] += 1;
                    }
                    assert_eq!(
                        counts,
                        [2u8; Rank::COUNT],
                        "{p} no longer owns two of every rank after determinization"
                    );
                }
            } else {
                let mut counts = [0u8; Rank::COUNT];
                for p in Player::BOTH {
                    for r in &world.hands[p.idx()] {
                        counts[r.index()] += 1;
                    }
                    for r in &world.discards[p.idx()] {
                        counts[r.index()] += 1;
                    }
                    for (_, _, c) in world.cards_of(p) {
                        counts[c.rank.index()] += 1;
                    }
                }
                for r in world.piles[0].ranks() {
                    counts[r.index()] += 1;
                }
                for r in &world.removed[0] {
                    counts[r.index()] += 1;
                }
                assert_eq!(counts, [4u8; Rank::COUNT], "the shared deck no longer balances");
            }
        }
    }
}

/// `game_rules.md` §3: "base cards are hidden from their owner too". So an observer's *own*
/// base card must be resampled, which is the counter-intuitive case and the one most likely
/// to be got wrong.
#[test]
fn rule_3_determinization_resamples_the_observers_own_base_cards() {
    let state = GameState::new_default(31);
    let mut rng = Rng::new(1);
    let base_id = state.lanes[0].side(P0)[0].id;
    assert!(state.card(base_id).expect("base card is in play").is_base);

    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..200 {
        let world = state.determinize(P0, &mut rng);
        seen.insert(world.card(base_id).expect("still in play").rank);
    }
    assert!(
        seen.len() > 3,
        "P0's own base card should take many ranks across samples, saw {}",
        seen.len()
    );
}

/// `game_rules.md` §2: the removed cards are "removed face-down, without revealing them", so
/// belief over them never resolves — they must be resampled like anything else hidden.
#[test]
fn rule_2_determinization_resamples_the_removed_pool() {
    let state = GameState::new_default(12);
    let mut rng = Rng::new(2);
    let real = state.removed[0].clone();
    let mut differed = 0;
    for _ in 0..100 {
        let world = state.determinize(P0, &mut rng);
        assert_eq!(world.removed[0].len(), real.len());
        if world.removed[0] != real {
            differed += 1;
        }
    }
    assert!(
        differed > 80,
        "the removed pool is unseen and must be resampled; it matched {} times in 100",
        100 - differed
    );
}

/// `game_rules.md` §9b: the mirrored variant *reveals* the removed multiset at setup, so it
/// is known information and must be held fixed — this is the whole reason §9b is "the
/// cleanest target for equilibrium analysis".
#[test]
fn rule_9b_determinization_keeps_a_revealed_removal_fixed() {
    let state = GameState::new(GameConfig::mirrored_removal(), 5);
    assert!(state.removed_revealed);
    let mut rng = Rng::new(3);
    for _ in 0..50 {
        let world = state.determinize(P0, &mut rng);
        assert_eq!(world.removed, state.removed, "§9b's removal is public");
    }
}

/// Determinization is a resample, not a shuffle of the visible board: applying it twice from
/// the same RNG state gives the same world, and it is idempotent on the information set.
#[test]
fn phase2_determinization_is_reproducible() {
    let state = position_after(GameConfig::split_deck(), 17, 40).expect("game still running");
    let observer = state.acting_player();
    let a = state.determinize(observer, &mut Rng::new(88));
    let b = state.determinize(observer, &mut Rng::new(88));
    assert_eq!(a, b, "same seed, same information set, same sampled world");

    // Sampling from a sampled world is the same operation, because the two share an
    // information set. This is what makes `phase2_no_agent_reads_hidden_information` exact.
    let c = a.determinize(observer, &mut Rng::new(5));
    let d = state.determinize(observer, &mut Rng::new(5));
    assert_eq!(c, d);
}

// ========================================================= honest agents ==

/// **The load-bearing test.** `DESIGN.md` §6: an agent must decide from its information set
/// and nothing else.
///
/// A determinized world is in the same information set as the real state by construction, so
/// an agent that reads only what it is entitled to must return the identical action from
/// either — same agent, same RNG stream, same decision. An agent that peeks at the
/// opponent's hand, the pile order, or a face-down rank fails immediately, because those are
/// exactly the fields determinization rewrites.
#[test]
fn phase2_no_agent_reads_hidden_information() {
    let mut rng = Rng::new(31337);
    let mut checked = 0;
    for state in sample_positions() {
        let observer = state.acting_player();
        let legal = state.legal_actions();
        if legal.len() < 2 {
            continue;
        }
        let world = state.determinize(observer, &mut rng);

        for spec in test_roster() {
            let real = spec.build(1234, 7).choose(&state, &legal);
            let sampled = spec.build(1234, 7).choose(&world, &legal);
            assert_eq!(
                real, sampled,
                "{spec} decided differently on two states in the same information set, so it \
                 is reading hidden information: {}",
                state.header()
            );
        }
        checked += 1;
    }
    assert!(checked > 100, "only {checked} positions actually exercised the check");
}

/// The evaluation function has the same obligation as the agents that use it: under
/// [`View::Informed`] it must not be able to tell two states in one information set apart.
#[test]
fn phase2_the_informed_evaluation_cannot_see_hidden_ranks() {
    let mut rng = Rng::new(64);
    let weights = EvalWeights::default();
    for state in sample_positions() {
        for observer in Player::BOTH {
            let world = state.determinize(observer, &mut rng);
            let a = evaluate(&state, observer, View::Informed(observer), &weights);
            let b = evaluate(&world, observer, View::Informed(observer), &weights);
            assert!(
                (a - b).abs() < 1e-6,
                "the informed evaluation moved from {a} to {b} when only hidden ranks changed"
            );
        }
    }
}

/// The counterpart: the omniscient view *does* read hidden ranks. If it did not, PIMC would
/// be paying for determinization and getting nothing for it.
#[test]
fn phase2_the_omniscient_evaluation_does_see_hidden_ranks() {
    let mut rng = Rng::new(65);
    let weights = EvalWeights::default();
    let mut moved = 0;
    for state in sample_positions().into_iter().take(60) {
        let world = state.determinize(P0, &mut rng);
        let a = evaluate(&state, P0, View::Omniscient, &weights);
        let b = evaluate(&world, P0, View::Omniscient, &weights);
        if (a - b).abs() > 1e-6 {
            moved += 1;
        }
    }
    assert!(moved > 10, "the omniscient view never noticed a resample");
}

// ============================================================ the ladder ==

/// Every rung finishes a game against every other, in every variant, without hitting the
/// ply-limit safety cap.
#[test]
fn phase2_every_agent_plays_a_legal_game_to_the_end() {
    for variant in Variant::ALL {
        let config = GameConfig::preset(variant);
        for spec in test_roster() {
            let stats = probe::play_spec_game(config, 3, spec.clone(), AgentSpec::Random);
            assert!(stats.outcome.is_over(), "{spec} in {variant} left a game unfinished");
            assert_ne!(
                stats.outcome,
                Outcome::Draw(duel52_engine::DrawReason::PlyLimit),
                "{spec} in {variant} hit the ply-limit cap, which indicates a rules bug"
            );
        }
    }
}

/// `CLAUDE.md`: "Everything is seeded and deterministic. Same seed + same config → identical
/// game." That has to survive the arrival of stateful search agents.
#[test]
fn phase2_agent_games_are_reproducible() {
    for spec in test_roster() {
        for seed in [0u64, 11, 909] {
            let cfg = GameConfig::split_deck();
            let a = probe::play_spec_game(cfg, seed, spec.clone(), spec.clone());
            let b = probe::play_spec_game(cfg, seed, spec.clone(), spec.clone());
            assert_eq!(a.outcome, b.outcome, "{spec} seed {seed}");
            assert_eq!(a.plies, b.plies, "{spec} seed {seed}");
            assert_eq!(a.decisions, b.decisions, "{spec} seed {seed}");
        }
    }
}

/// A benchmark whose numbers move when you change `--threads` is not a benchmark. Sharding
/// is by seed range and every agent is re-derived from its game seed, so the result must be
/// bit-identical however the work is split.
#[test]
fn phase2_the_ladder_is_thread_count_independent() {
    let a = ladder::run_match(
        GameConfig::split_deck(),
        AgentSpec::Greedy,
        AgentSpec::Random,
        1,
        16,
        1,
    );
    let b = ladder::run_match(
        GameConfig::split_deck(),
        AgentSpec::Greedy,
        AgentSpec::Random,
        1,
        16,
        4,
    );
    assert_eq!(a.wins, b.wins);
    assert_eq!(a.draws, b.draws);
    assert_eq!(a.games, b.games);
    assert_eq!(a.max_side_occupancy, b.max_side_occupancy);
}

/// Colour-paired deals: every deal is played twice with the seats swapped, so both agents
/// meet the same deals and a first-player edge cannot masquerade as strength.
#[test]
fn phase2_a_match_balances_colours() {
    let m = ladder::run_match(
        GameConfig::split_deck(),
        AgentSpec::Random,
        AgentSpec::Random,
        1,
        40,
        1,
    );
    assert_eq!(m.games, 40);
    assert_eq!(m.behaviour[0].games, 40, "each agent played every game");
    assert_eq!(m.behaviour[1].games, 40);
    assert_eq!(m.p0_seat_games, 40);
}

/// An odd game count would leave one deal played from one side only, which is exactly the
/// asymmetry the pairing exists to remove. It is rounded up rather than rejected.
#[test]
fn phase2_a_match_rounds_up_to_an_even_game_count() {
    let m = ladder::run_match(
        GameConfig::split_deck(),
        AgentSpec::Random,
        AgentSpec::Greedy,
        1,
        7,
        1,
    );
    assert_eq!(m.games, 8);
}

/// The whole point of the ladder: search beats no search. Small budgets, so this asserts a
/// direction rather than a margin — the measured margins belong in `FINDINGS.md`.
#[test]
fn phase2_greedy_beats_random_decisively() {
    let m = ladder::run_match(
        GameConfig::split_deck(),
        AgentSpec::Greedy,
        AgentSpec::Random,
        1,
        60,
        4,
    );
    assert!(
        m.score() > 0.75,
        "a hand-written evaluation should crush uniform random; scored {:.3}",
        m.score()
    );
}

/// ISMCTS with random rollouts and no evaluation function should also beat random outright.
/// If this fails, the availability bookkeeping or the reward indexing is wrong.
#[test]
fn phase2_ismcts_beats_random_decisively() {
    let m = ladder::run_match(
        GameConfig::split_deck(),
        AgentSpec::Ismcts { iterations: 120 },
        AgentSpec::Random,
        1,
        40,
        4,
    );
    assert!(
        m.score() > 0.75,
        "ISMCTS should crush uniform random; scored {:.3}",
        m.score()
    );
}

/// SO-ISMCTS's exploration term divides by an edge's **availability** — the iterations in
/// which it was legal — not by the parent's visit count (Cowling, Powley & Whitehouse 2012).
///
/// Tested by its consequence rather than by reaching into the tree: every iteration must
/// traverse exactly one root edge, so the root visit counts sum to the iteration budget. A
/// tree that miscounted availability would still sum correctly, but one that skipped or
/// double-counted a traversal — the way a naive UCB1 port does when the legal set changes
/// between iterations — would not.
#[test]
fn rule_6_ismcts_spends_every_iteration_on_exactly_one_root_edge() {
    // A node with real width — a mid-turn position can legitimately offer a single forced
    // sub-decision, which would make the second assertion vacuous.
    let state = (0..40u64)
        .filter_map(|seed| position_after(GameConfig::split_deck(), seed, 30))
        .find(|s| s.legal_actions().len() >= 8)
        .expect("some mid-game position offers eight or more actions");
    let legal = state.legal_actions();
    let mut agent = IsmctsAgent::new(9, 200);
    let visits = agent.root_visits(&state, &legal);
    assert_eq!(
        visits.iter().sum::<u32>(),
        200,
        "every iteration must descend exactly one root edge"
    );
    assert!(
        visits.iter().filter(|&&v| v > 0).count() > 1,
        "the search collapsed onto a single action"
    );
}

/// A search that cannot find a move that wins on the spot is not a search.
///
/// The position: base cards are unlocked, both hands are empty, and P0 has one attack that
/// empties P1's last two lanes' worth of material — `game_rules.md` §7's three conditions all
/// hold, so the kill ends the game.
#[test]
fn rule_7_search_agents_take_an_immediately_winning_attack() {
    for spec in [
        AgentSpec::Greedy,
        AgentSpec::FlatMc { playouts: 60 },
        AgentSpec::Pimc {
            worlds: 4,
            depth: 1,
        },
        AgentSpec::Ismcts { iterations: 200 },
    ] {
        let mut p = Position::empty();
        // Lane 0: P1 has nothing — already won.
        p.face_up(0, P0, Rank::FOUR);
        // Lane 1: one damaged P1 card, one hit from dying.
        p.face_up(1, P0, Rank::SEVEN);
        let victim = p.face_up(1, P1, Rank::FOUR);
        p.damage(1, P1, victim, 1);
        // Lane 2: P1 holds it comfortably, so lane 1 is the only winning line.
        p.face_up(2, P1, Rank::JACK);
        p.face_up(2, P0, Rank::FIVE);
        p.hand(P0, &[]);
        p.hand(P1, &[]);
        p.unlock();
        let state = p.build();

        assert_eq!(state.lanes_won_by(P0), 1, "lane 0 is already won");
        let legal = state.legal_actions();
        let action = spec.build(77, 1).choose(&state, &legal);
        assert_eq!(
            action,
            Action::Attack {
                lane: 1,
                attacker: 0,
                target: 0
            },
            "{spec} passed up a move that wins the game on the spot"
        );
    }
}

// =============================================================== plumbing ==

#[test]
fn phase2_agent_specs_round_trip_through_their_names() {
    for spec in AgentSpec::LADDER {
        assert_eq!(AgentSpec::parse(&spec.name()), Ok(spec));
    }
    assert_eq!(AgentSpec::parse("ismcts:1600"), Ok(AgentSpec::Ismcts { iterations: 1600 }));
    assert_eq!(
        AgentSpec::parse("pimc:12x2"),
        Ok(AgentSpec::Pimc {
            worlds: 12,
            depth: 2
        })
    );
    assert!(AgentSpec::parse("mcts").is_err());
    assert!(AgentSpec::parse("ismcts:lots").is_err());
}

/// A full round robin fits ratings, orders them, and anchors random at zero.
#[test]
fn phase2_the_ladder_produces_an_ordered_rating_table() {
    let result = ladder::run_ladder(
        GameConfig::split_deck(),
        &[AgentSpec::Random, AgentSpec::Greedy],
        1,
        40,
        4,
        "random",
        false,
    );
    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.elo.ratings[0], 0.0, "random anchors the scale");
    assert!(
        result.elo.ratings[1] > 100.0,
        "greedy should rate well above random, got {:+.0}",
        result.elo.ratings[1]
    );
    assert!(result.report().contains("Ratings"));
}

/// `FINDINGS.md`'s Phase 2 measurements are read off [`probe::GameStats`], so the
/// instrumentation has to agree with the game it watched.
#[test]
fn phase2_instrumentation_agrees_with_the_game_it_watched() {
    let stats = probe::play_spec_game(
        GameConfig::split_deck(),
        44,
        AgentSpec::Greedy,
        AgentSpec::Random,
    );
    assert!(stats.outcome.is_over());
    assert!(stats.plies > 1);
    assert!(stats.decisions >= stats.plies);
    assert!(
        stats.max_side_occupancy >= 1,
        "every lane starts with a base card"
    );
    // A card cannot be flipped more often than it was played, per rank, except for base
    // cards — which are never played but can be flipped once the pile empties.
    let played: u32 = stats.plays_by_rank[0].iter().sum();
    assert!(played > 0, "a full game must involve playing cards");

    // The unlock is deterministic under the house 2 (`FINDINGS.md` F1.6), so if this game
    // reached it, it reached it on the documented ply.
    if let Some(ply) = stats.ply_at_unlock {
        assert_eq!(ply, 25, "split-deck games unlock on ply 25 (F1.6)");
        assert!(stats.hand_at_unlock.iter().all(|&h| h <= 30));
    }
}

/// Concentration is a share of a total, so it lives in `1/lanes .. 1` and a player who never
/// acted has none.
#[test]
fn phase2_lane_concentration_is_a_well_formed_share() {
    let stats = probe::play_spec_game(
        GameConfig::split_deck(),
        8,
        AgentSpec::Greedy,
        AgentSpec::Greedy,
    );
    for p in [P0, P1] {
        if let Some(share) = stats.lane_concentration(p, 2) {
            assert!(
                (0.6..=1.0).contains(&share),
                "two of three lanes cannot hold less than two thirds of the plays: {share}"
            );
        }
    }
}

/// `GameStats::stuck_turns` agrees with an independent count of the same thing.
///
/// §4 has no pass, so a turn ending with actions unspent is the engine's doing and there is
/// no action for the probe to intercept — it is inferred from the ply counter instead
/// (`note_turn_ends`). Inference is exactly the kind of measurement that can be quietly
/// wrong and still look plausible in a table, and `FINDINGS.md` F2.4b reads a conclusion off
/// this column, so it gets checked against a count derived a completely different way:
/// replay the game, tally the action-costing decisions taken in each ply, and call a ply
/// stuck when it holds fewer than its allowance. A skipped turn holds none.
#[test]
fn phase2_the_stuck_turn_count_matches_an_independent_recount() {
    let config = GameConfig::split_deck();
    for spec in [AgentSpec::Random, AgentSpec::Greedy] {
        for seed in 0..40u64 {
            let stats = probe::play_spec_game(config, seed, spec.clone(), spec.clone());

            // Replay the same game, counting what each ply actually contained.
            let mut spent: Vec<u32> = Vec::new();
            let mut state = GameState::new(config, seed);
            // The same streams `play_spec_game` uses, so the replay is the same game.
            let mut p0 = spec.build(seed, probe::AGENT_STREAM[0]);
            let mut p1 = spec.build(seed, probe::AGENT_STREAM[1]);
            while !state.outcome.is_over() {
                let legal = state.legal_actions();
                let action = match state.to_move {
                    Player::P0 => p0.choose(&state, &legal),
                    Player::P1 => p1.choose(&state, &legal),
                };
                let ply = state.ply as usize;
                if spent.len() <= ply {
                    spent.resize(ply + 1, 0);
                }
                if action.costs_an_action() {
                    spent[ply] += 1;
                }
                state.apply_trusted(action);
            }
            if spent.len() <= state.ply as usize {
                spent.resize(state.ply as usize + 1, 0);
            }

            // The last ply is where the game ended, so it was cut short by the result
            // rather than by having nothing in it. Every ply before it is fair game.
            let mut recount = [0u32; 2];
            for (ply, &taken) in spent.iter().enumerate().take(state.ply as usize) {
                let allowance = if ply == 0 {
                    config.first_turn_actions
                } else {
                    config.actions_per_turn
                };
                if taken < allowance {
                    recount[ply % 2] += 1;
                }
            }

            assert_eq!(
                stats.stuck_turns, recount,
                "{spec:?} seed {seed}: the probe counted {:?} stuck turns, the replay found \
                 {recount:?}",
                stats.stuck_turns
            );
        }
    }
}

/// Elo is only defined up to an additive constant, so the anchor has to actually anchor.
#[test]
fn phase2_elo_anchors_where_it_is_told() {
    let table = elo::fit(
        vec!["a".into(), "b".into(), "c".into()],
        &[
            elo::Pairing::new(0, 1, 300, 700, 0),
            elo::Pairing::new(1, 2, 300, 700, 0),
        ],
        1,
    );
    assert_eq!(table.ratings[1], 0.0);
    assert!(table.ratings[0] < 0.0 && table.ratings[2] > 0.0);
}
