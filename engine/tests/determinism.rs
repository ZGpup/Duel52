//! Reproducibility and structural invariants.
//!
//! `CLAUDE.md`: "Everything is seeded and deterministic. Same seed + same config →
//! identical game. Non-reproducible results are bugs." Everything the project will report
//! rests on this, so it gets its own suite rather than a line in another one.

use duel52_engine::agents::{Agent, RandomAgent};
use duel52_engine::testkit::end_turn;
use duel52_engine::{
    Action, GameConfig, GameState, Outcome, Player, Player::P0, Player::P1, Rank, TwoPower,
    Variant,
};

/// Play a full game with fixed agent seeds and return everything that could differ.
fn fingerprint(config: GameConfig, seed: u64) -> (Vec<String>, Outcome, u32) {
    let mut s = GameState::new(config, seed);
    let mut p0 = RandomAgent::new(0xA1);
    let mut p1 = RandomAgent::new(0xB2);
    let mut trace = Vec::new();

    while !s.outcome.is_over() {
        let legal = s.legal_actions();
        let action = match s.to_move {
            P0 => p0.choose(&s, &legal),
            P1 => p1.choose(&s, &legal),
        };
        trace.push(format!("{} {}", s.to_move, action));
        s.apply_trusted(action);
    }
    (trace, s.outcome, s.ply)
}

/// The headline guarantee.
#[test]
fn same_seed_and_config_produce_an_identical_game() {
    for variant in Variant::ALL {
        for two_power in [TwoPower::Bottom, TwoPower::Discard] {
            let mut config = GameConfig::preset(variant);
            config.two_power = two_power;
            for seed in [0u64, 1, 7, 999, u64::MAX] {
                let a = fingerprint(config, seed);
                let b = fingerprint(config, seed);
                assert_eq!(
                    a, b,
                    "{variant}/{two_power:?} seed {seed} is not reproducible"
                );
            }
        }
    }
}

/// Different seeds must actually produce different games, or the seeding is broken in the
/// opposite direction.
#[test]
fn different_seeds_produce_different_games() {
    let config = GameConfig::split_deck();
    let traces: Vec<_> = (0..20u64).map(|s| fingerprint(config, s).0).collect();
    let distinct: std::collections::HashSet<_> = traces.iter().collect();
    assert!(
        distinct.len() >= 19,
        "20 seeds produced only {} distinct games",
        distinct.len()
    );
}

/// Cloning a position and playing both copies the same way must give identical results.
/// `DESIGN.md` §6's ISMCTS clones positions constantly; a clone that diverges would corrupt
/// every search statistic.
#[test]
fn cloned_positions_evolve_identically() {
    for variant in Variant::ALL {
        let mut original = GameState::new(GameConfig::preset(variant), 5);
        // Advance a few plies so the position is non-trivial.
        let mut agent = RandomAgent::new(3);
        for _ in 0..25 {
            if original.outcome.is_over() {
                break;
            }
            let legal = original.legal_actions();
            let action = agent.choose(&original, &legal);
            original.apply_trusted(action);
        }

        let mut a = original.clone();
        let mut b = original.clone();
        assert_eq!(a, b);

        let mut agent_a = RandomAgent::new(9);
        let mut agent_b = RandomAgent::new(9);
        while !a.outcome.is_over() {
            let action = agent_a.choose(&a, &a.legal_actions());
            a.apply_trusted(action);
            let action_b = agent_b.choose(&b, &b.legal_actions());
            b.apply_trusted(action_b);
        }
        assert_eq!(a, b, "{variant}: clones diverged");
    }
}

/// The card census must hold for the whole game: no card is ever created or destroyed, only
/// moved between the board, hands, piles, discards, and the removed pool.
#[test]
fn cards_are_conserved_throughout_a_game() {
    for variant in Variant::ALL {
        for seed in 0..12u64 {
            let config = GameConfig::preset(variant);
            let mut s = GameState::new(config, seed);
            let expected = s.expected_card_count();
            let mut agent = RandomAgent::new(seed);

            while !s.outcome.is_over() {
                assert_eq!(
                    s.card_census(),
                    expected,
                    "{variant} seed {seed} at ply {}: cards were lost or duplicated",
                    s.ply
                );
                let action = agent.choose(&s, &s.legal_actions());
                s.apply_trusted(action);
            }
            assert_eq!(s.card_census(), expected);
        }
    }
}

/// `two_power = discard` is the exception: RAW permanently removes a card from the pile
/// into the discard, so the census still holds — the card moves, it does not vanish.
#[test]
fn rule_10a_the_raw_two_moves_a_card_to_the_discard_without_losing_it() {
    let mut config = GameConfig::split_deck();
    config.two_power = TwoPower::Discard;

    for seed in 0..12u64 {
        let mut s = GameState::new(config, seed);
        let expected = s.expected_card_count();
        let mut agent = RandomAgent::new(seed);
        while !s.outcome.is_over() {
            assert_eq!(s.card_census(), expected, "seed {seed}");
            let action = agent.choose(&s, &s.legal_actions());
            s.apply_trusted(action);
        }
    }
}

/// A legal action must never be rejected by `apply`, and `apply` must never accept anything
/// outside `legal_actions`. This is the contract every agent relies on.
#[test]
fn every_enumerated_action_is_accepted_and_nothing_else_is() {
    for variant in Variant::ALL {
        let mut s = GameState::new(GameConfig::preset(variant), 21);
        let mut agent = RandomAgent::new(6);

        while !s.outcome.is_over() {
            let legal = s.legal_actions();
            assert!(!legal.is_empty(), "no legal action but the game is running");

            // Every enumerated action applies cleanly from a clone.
            for &action in &legal {
                let mut probe = s.clone();
                probe
                    .apply(action)
                    .unwrap_or_else(|e| panic!("{variant}: enumerated action rejected: {e}"));
            }

            // A plainly out-of-range action is refused rather than panicking.
            let bogus = Action::Attack {
                lane: 0,
                attacker: 200,
                target: 200,
            };
            assert!(s.apply(bogus).is_err(), "{variant}: bogus action accepted");

            let action = agent.choose(&s, &legal);
            s.apply_trusted(action);
        }
    }
}

/// A finished game refuses every action rather than continuing.
#[test]
fn a_finished_game_accepts_nothing() {
    let mut s = GameState::new_default(2);
    let mut agent = RandomAgent::new(1);
    while !s.outcome.is_over() {
        let action = agent.choose(&s, &s.legal_actions());
        s.apply_trusted(action);
    }
    assert!(s.legal_actions().is_empty());
    assert!(s.apply(Action::Play { rank: Rank::ACE, lane: 0 }).is_err());
}

/// Sub-decisions always belong to the player whose action opened them, and the game never
/// stalls inside one: every position with a pending sub-decision offers at least one way
/// out.
#[test]
fn sub_decisions_always_have_an_answer_and_belong_to_the_acting_player() {
    for variant in Variant::ALL {
        for seed in 0..12u64 {
            let mut s = GameState::new(GameConfig::preset(variant), seed);
            let mut agent = RandomAgent::new(seed);
            while !s.outcome.is_over() {
                if let Some(pending) = s.pending.last() {
                    assert_eq!(
                        pending.player(),
                        s.to_move,
                        "{variant} seed {seed}: a sub-decision is owed by the wrong player"
                    );
                    assert!(
                        !s.legal_actions().is_empty(),
                        "{variant} seed {seed}: stuck inside a sub-decision"
                    );
                }
                let action = agent.choose(&s, &s.legal_actions());
                s.apply_trusted(action);
            }
            assert!(
                s.pending.is_empty(),
                "{variant} seed {seed}: the game ended mid-resolution"
            );
        }
    }
}

/// §7: "The terminal check runs **after each action fully resolves**, including every
/// sub-decision that action opened. It never runs mid-resolution."
#[test]
fn rule_7_the_terminal_check_never_runs_mid_resolution() {
    for variant in Variant::ALL {
        for seed in 0..20u64 {
            let mut s = GameState::new(GameConfig::preset(variant), seed);
            let mut agent = RandomAgent::new(seed);
            while !s.outcome.is_over() {
                let action = agent.choose(&s, &s.legal_actions());
                s.apply_trusted(action);
                assert!(
                    !(s.outcome.is_over() && !s.pending.is_empty()),
                    "{variant} seed {seed}: the game was declared over with a \
                     sub-decision still owed"
                );
            }
        }
    }
}

/// §7: "**A lane win cannot be undone.**" Once a player is at or above the threshold on a
/// terminal check, the game is over — so the count can never be observed going back down.
#[test]
fn rule_7_a_lane_win_is_never_undone() {
    for variant in Variant::ALL {
        for seed in 0..20u64 {
            let mut s = GameState::new(GameConfig::preset(variant), seed);
            let mut agent = RandomAgent::new(seed);
            let mut best = [0usize; 2];
            while !s.outcome.is_over() {
                for p in Player::BOTH {
                    let won = s.lanes_won_by(p);
                    assert!(
                        won >= best[p.idx()] || won == 0 && best[p.idx()] == 0,
                        "{variant} seed {seed}: {p}'s lane count fell from {} to {won}",
                        best[p.idx()]
                    );
                    best[p.idx()] = best[p.idx()].max(won);
                    assert!(
                        won < s.config.lanes_to_win,
                        "{variant} seed {seed}: {p} is at the threshold but the game is \
                         still running"
                    );
                }
                let action = agent.choose(&s, &s.legal_actions());
                s.apply_trusted(action);
            }
        }
    }
}

/// §10a: "**Turns-to-unlock becomes fixed.** One draw per turn and no pile shrinkage means
/// the pile empties after exactly *pile size* turns, regardless of how many 2s are played.
/// The whole endgame trigger is deterministic at deal time."
///
/// P0 draws on plies 0, 2, 4, ...; P1 on 1, 3, 5, .... With a 13-card pile each, P0's
/// empties on ply 24 and P1's on ply 25, so the global unlock is ply 25 in every game that
/// reaches it. The 2 is pile-neutral, so no amount of scrying moves it.
#[test]
fn rule_10a_the_house_two_makes_turns_to_unlock_invariant() {
    let config = GameConfig::split_deck();
    let expected = 2 * (config.expected_pile_size() as u32 - 1) + 1;
    let mut checked = 0;

    for seed in 0..250u64 {
        let summary = duel52_engine::stats::play_random_game(config, seed);
        if let Some(ply) = summary.ply_at_unlock {
            assert_eq!(ply, expected, "seed {seed}: unlock ply must be fixed");
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} games reached the unlock");
}

/// The contrast that makes the house rule measurable: under `two_power = discard` the pile
/// shrinks, so turns-to-unlock stops being fixed. §10a: "remove one card and someone now
/// gets an extra draw, and the player who fires the 2 chooses who."
#[test]
fn rule_10a_the_raw_two_makes_turns_to_unlock_vary() {
    let mut config = GameConfig::split_deck();
    config.two_power = TwoPower::Discard;

    let unlock_plies: std::collections::HashSet<u32> = (0..250u64)
        .filter_map(|seed| duel52_engine::stats::play_random_game(config, seed).ply_at_unlock)
        .collect();

    assert!(
        unlock_plies.len() > 1,
        "RAW should produce a spread of unlock plies, got {unlock_plies:?}"
    );
}

/// `DESIGN.md` §8 targets ">=10k full random games/sec/core". This is a smoke test rather
/// than a benchmark — it runs in debug mode at opt-level 1 — so the bar is set well below
/// the target. It exists to catch an accidental quadratic, not to certify throughput.
#[test]
fn random_games_are_not_pathologically_slow() {
    let start = std::time::Instant::now();
    let stats = duel52_engine::run_random_games(GameConfig::split_deck(), 0, 200);
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(stats.games, 200);
    assert!(
        elapsed < 20.0,
        "200 games took {elapsed:.1}s, which suggests something is quadratic"
    );
}

/// The engine's own state invariants hold at every position of every game. These are the
/// checks in `GameState::debug_check_invariants` — dead cards never linger in play, pairs
/// are always exactly two cards on one side of one lane, a face-up card is always known to
/// both players, and `is_base` never outruns `entered_as_base`.
#[test]
fn state_invariants_hold_throughout_every_variant() {
    for variant in Variant::ALL {
        for two_power in [TwoPower::Bottom, TwoPower::Discard] {
            let mut config = GameConfig::preset(variant);
            config.two_power = two_power;
            for seed in 0..25u64 {
                // `apply_trusted` runs the invariant checks in debug builds, which is where
                // tests run, so simply playing the games exercises them.
                let summary = duel52_engine::stats::play_random_game(config, seed);
                assert!(summary.outcome.is_over());
            }
        }
    }
}

/// Hands stay sorted, which is what lets `legal_actions` collapse duplicate ranks by
/// checking only adjacent entries.
#[test]
fn hands_are_always_sorted() {
    for seed in 0..15u64 {
        let mut s = GameState::new_default(seed);
        let mut agent = RandomAgent::new(seed);
        while !s.outcome.is_over() {
            for p in Player::BOTH {
                let hand = s.hand(p);
                assert!(
                    hand.windows(2).all(|w| w[0] <= w[1]),
                    "seed {seed}: {p}'s hand is unsorted: {hand:?}"
                );
            }
            let action = agent.choose(&s, &s.legal_actions());
            s.apply_trusted(action);
        }
    }
}

/// The legality enumerator must not offer the same action twice — a duplicate would double
/// its probability under a random or uniform-prior policy, silently biasing every statistic
/// and every search.
#[test]
fn legal_actions_are_unique() {
    for variant in Variant::ALL {
        for seed in 0..12u64 {
            let mut s = GameState::new(GameConfig::preset(variant), seed);
            let mut agent = RandomAgent::new(seed);
            while !s.outcome.is_over() {
                let legal = s.legal_actions();
                let unique: std::collections::HashSet<_> =
                    legal.iter().map(|a| format!("{a:?}")).collect();
                assert_eq!(
                    unique.len(),
                    legal.len(),
                    "{variant} seed {seed}: duplicate legal actions at ply {}",
                    s.ply
                );
                let action = agent.choose(&s, &legal);
                s.apply_trusted(action);
            }
        }
    }
}

/// A sanity check on the deal: across many seeds every rank shows up, so no rank is being
/// silently dropped by the deck builder.
#[test]
fn every_rank_appears_across_many_deals() {
    for variant in Variant::ALL {
        let mut seen = [false; Rank::COUNT];
        for seed in 0..40u64 {
            let s = GameState::new(GameConfig::preset(variant), seed);
            for p in Player::BOTH {
                for r in s.hand(p) {
                    seen[r.index()] = true;
                }
                for r in s.pile(p).ranks() {
                    seen[r.index()] = true;
                }
            }
        }
        assert!(
            seen.iter().all(|&b| b),
            "{variant}: some rank never appeared in 40 deals"
        );
    }
}

/// `game_rules.md` §4: **actions are mandatory**, and there is no action that declines one.
///
/// The structural claim, checked on every decision of a full game: whenever the engine hands
/// a player a position, it offers at least one thing to do, and everything it offers in the
/// main phase costs an action. There is nothing in the action space whose purpose is to move
/// the game along — a policy head over these logits is a distribution over real choices.
///
/// This is what `settle`'s auto-advance buys, and it is worth a test rather than a comment:
/// the invariant is invisible until it breaks, and when it breaks it presents as an agent
/// panicking on an empty action list somewhere far away.
#[test]
fn rule_4_every_offered_action_is_a_real_action() {
    for variant in Variant::ALL {
        for seed in 0..10u64 {
            let mut s = GameState::new(GameConfig::preset(variant), seed);
            let mut agent = RandomAgent::new(seed);
            while !s.outcome.is_over() {
                let legal = s.legal_actions();
                assert!(
                    !legal.is_empty(),
                    "{variant} seed {seed} ply {}: a running game offered no decision",
                    s.ply
                );
                if s.pending.is_empty() {
                    assert!(
                        legal.iter().all(|a| a.costs_an_action()),
                        "{variant} seed {seed}: the main phase offered a free action"
                    );
                }
                let action = agent.choose(&s, &legal);
                s.apply_trusted(action);
            }
        }
    }
}

/// The other half of §4: a player holding a card can always play it, so a full hand is
/// never stuck. This is what makes the mutual-standoff draw unreachable — the non-attacking
/// actions are all finite (a hand drains, a card flips once, a card joins one pair and
/// "cannot leave one pair to join another"), so a player who refuses to attack runs out of
/// ways to refuse.
///
/// It is also what `settle`'s auto-advance relies on to stay off the hot path: it checks the
/// hand before enumerating anything.
#[test]
fn rule_4_a_player_holding_a_card_is_never_stuck() {
    for variant in Variant::ALL {
        for seed in 0..10u64 {
            let mut s = GameState::new(GameConfig::preset(variant), seed);
            let mut agent = RandomAgent::new(seed);
            while !s.outcome.is_over() {
                let legal = s.legal_actions();
                if s.pending.is_empty() && !s.hand(s.to_move).is_empty() {
                    assert!(
                        legal.iter().any(|a| matches!(a, Action::Play { .. })),
                        "{variant} seed {seed}: a card in hand is always a legal play"
                    );
                }
                let action = agent.choose(&s, &legal);
                s.apply_trusted(action);
            }
        }
    }
}

/// Config presets survive a round trip through the text format, so a config file recorded
/// alongside a finding reproduces the same game.
#[test]
fn configs_round_trip_so_findings_stay_reproducible() {
    for variant in Variant::ALL {
        for two_power in [TwoPower::Bottom, TwoPower::Discard] {
            let mut config = GameConfig::preset(variant);
            config.two_power = two_power;
            let text = config.to_config_string();
            let parsed = GameConfig::from_config_str(&text).expect("must re-parse");
            assert_eq!(parsed, config);

            // And the same seed under the re-parsed config gives the same game.
            assert_eq!(
                fingerprint(config, 17),
                fingerprint(parsed, 17),
                "{variant}/{two_power:?}"
            );
        }
    }
}

/// P1 never gets the short opening turn — it belongs to whoever moves first.
///
/// §4: "the first player's opening turn" is the *only* short turn a player gets by rule.
#[test]
fn only_the_first_player_gets_the_short_opening_turn() {
    let mut s = GameState::new_default(8);
    assert_eq!(s.to_move, P0);
    assert_eq!(s.actions_remaining, 2);
    end_turn(&mut s);
    assert_eq!(s.to_move, P1);
    assert_eq!(s.actions_remaining, 3);
    end_turn(&mut s);
    assert_eq!(s.to_move, P0);
    assert_eq!(s.actions_remaining, 3, "P0's second turn is a full one");
}
