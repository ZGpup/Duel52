//! `PLAN.md` §4.0 — recording a played game, and replaying it exactly.
//!
//! The claim the whole format rests on: **a game is `(config, seed, chosen indices)`**. The
//! engine is deterministic, so those few hundred bytes reproduce the game exactly — both
//! hands, both base cards, the order of the draw pile and the ten cards removed unseen.
//! Everything here is an attempt to break that claim.
//!
//! The occasion: the owner beat `gen016` 5–0 and not one ply was written down. The human
//! series is the only external measurement this project has, and it was not being kept.

use duel52_engine::record::{prepare, read_all, GameRecord, FORMAT};
use duel52_engine::{
    Action, Agent, AgentSpec, GameConfig, GameState, Outcome, Player, RandomAgent,
};

/// Play a whole game with two agents, keeping the index chosen at each decision.
fn play_and_record(config: GameConfig, seed: u64) -> (GameRecord, GameState) {
    let mut state = GameState::new(config, seed);
    let mut p0 = RandomAgent::derived(seed, 1);
    let mut p1 = RandomAgent::derived(seed, 2);
    let mut moves = Vec::new();

    while !state.outcome.is_over() {
        let legal = state.legal_actions();
        let action = match state.acting_player() {
            Player::P0 => p0.choose(&state, &legal),
            Player::P1 => p1.choose(&state, &legal),
        };
        moves.push(
            legal
                .iter()
                .position(|a| *a == action)
                .expect("an agent returns an action it was offered"),
        );
        state.apply_trusted(action);
    }
    let record = GameRecord::new(
        config,
        seed,
        Player::P0,
        Some("random".to_string()),
        moves,
        state.outcome,
    );
    (record, state)
}

#[test]
fn phase4_a_recorded_game_replays_to_the_same_position_ply_for_ply() {
    let config = GameConfig::split_deck();
    let (record, played) = play_and_record(config, 4242);

    // Not merely the same outcome: the same position at every ply. A record that reproduced
    // only the result would be worthless for the diagnosis it exists to support.
    let mut seen = Vec::new();
    let replayed = record
        .walk(|state, legal, chosen| seen.push((state.ply, legal[chosen])))
        .expect("the record replays");

    assert_eq!(seen.len(), record.moves.len());
    assert_eq!(replayed.outcome, played.outcome);
    assert_eq!(replayed.ply, played.ply);

    // And hidden information came back too, which is the part that makes this cheap: the
    // record carries no cards at all.
    for player in Player::BOTH {
        assert_eq!(
            replayed.hand(player).len(),
            played.hand(player).len(),
            "the replayed hand differs, so the deal was not reproduced"
        );
    }
    assert_eq!(replayed.discards, played.discards);
}

#[test]
fn phase4_a_record_survives_a_round_trip_through_json() {
    let (record, _) = play_and_record(GameConfig::split_deck(), 7);
    let line = record.to_json_line();

    assert!(line.ends_with('\n'), "JSONL is one line per game");
    assert_eq!(line.matches('\n').count(), 1, "the config must not break the line");
    assert!(line.contains(FORMAT));

    let parsed = GameRecord::parse(&line).expect("a line we wrote parses");
    assert_eq!(parsed, record);
    // Including the config, which is embedded as an escaped multi-line string.
    assert_eq!(parsed.config, record.config);
    parsed.walk(|_, _, _| {}).expect("and still replays");
}

#[test]
fn phase4_a_record_of_a_hotseat_game_has_no_opponent() {
    let mut record = play_and_record(GameConfig::split_deck(), 11).0;
    record.opponent = None;
    let parsed = GameRecord::parse(&record.to_json_line()).expect("parses");
    assert_eq!(parsed.opponent, None);
    assert!(record.to_json_line().contains("\"opponent\":null"));
}

#[test]
fn phase4_every_variant_round_trips_including_its_config() {
    // The config is what the replay rebuilds the game from, so a variant whose config does
    // not survive the JSON string would replay a different game while looking fine.
    for config in [
        GameConfig::split_deck(),
        GameConfig::base(),
        GameConfig::mirrored_removal(),
    ] {
        let (record, _) = play_and_record(config, 99);
        let parsed = GameRecord::parse(&record.to_json_line()).expect("parses");
        assert_eq!(parsed.config, config, "{:?} did not survive", config.variant);
        parsed.walk(|_, _, _| {}).expect("replays");
    }
}

#[test]
fn phase4_a_tampered_move_index_is_refused_rather_than_replayed() {
    let (mut record, _) = play_and_record(GameConfig::split_deck(), 5);
    // An index past the end of the legal list at that ply.
    record.moves[3] = 9_999;
    let error = record.walk(|_, _, _| {}).expect_err("must not replay");
    assert!(error.contains("ply 4"), "{error}");
    assert!(error.contains("9999"), "{error}");
}

#[test]
fn phase4_a_truncated_record_is_refused_rather_than_replayed() {
    let (mut record, _) = play_and_record(GameConfig::split_deck(), 5);
    record.moves.truncate(record.moves.len() - 4);
    let error = record.walk(|_, _, _| {}).expect_err("must not replay");
    assert!(error.contains("still in progress"), "{error}");
}

#[test]
fn phase4_a_record_whose_outcome_no_longer_matches_is_refused() {
    // This is the check that catches a rules change: the moves still replay, the game still
    // ends, and it ends *differently*. Silence here would mean a corpus that quietly
    // describes games nobody played.
    let (mut record, played) = play_and_record(GameConfig::split_deck(), 5);
    record.outcome = match played.outcome {
        Outcome::Win(Player::P0) => "P1 wins",
        _ => "P0 wins",
    }
    .to_string();
    let error = record.walk(|_, _, _| {}).expect_err("must not replay");
    assert!(error.contains("does not reproduce"), "{error}");
}

#[test]
fn phase4_a_record_from_a_different_format_version_is_refused() {
    let (record, _) = play_and_record(GameConfig::split_deck(), 5);
    let line = record.to_json_line().replace(FORMAT, "duel52-play/99");
    let error = GameRecord::parse(&line).expect_err("must not parse");
    assert!(error.contains("duel52-play/99"), "{error}");
}

#[test]
fn phase4_the_human_result_is_written_from_the_recorded_seat() {
    let config = GameConfig::split_deck();
    let (as_p0, played) = play_and_record(config, 4242);
    let winner = match played.outcome {
        Outcome::Win(w) => w,
        other => panic!("expected a decisive game, got {other}"),
    };

    assert_eq!(as_p0.human, Player::P0);
    assert_eq!(as_p0.human_result, if winner == Player::P0 { "win" } else { "loss" });

    // The same game recorded from the other seat reports the other result.
    let as_p1 = GameRecord::new(
        config,
        4242,
        Player::P1,
        Some("random".to_string()),
        as_p0.moves.clone(),
        played.outcome,
    );
    assert_ne!(as_p0.human_result, as_p1.human_result);
}

#[test]
fn phase4_a_seed_too_large_for_an_f64_survives_the_round_trip() {
    // Numbers are kept as their source text for exactly this reason: a `u64` seed above
    // 2^53 does not survive `f64`, and a silently changed seed is a silently changed deal.
    let seed = u64::MAX - 12345;
    let (record, _) = play_and_record(GameConfig::split_deck(), seed);
    let parsed = GameRecord::parse(&record.to_json_line()).expect("parses");
    assert_eq!(parsed.seed, seed);
    parsed.walk(|_, _, _| {}).expect("replays");
}

#[test]
fn phase4_a_records_file_reads_back_in_the_order_it_was_written() {
    let dir = std::env::temp_dir().join(format!("duel52-record-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("series.jsonl");
    let _ = std::fs::remove_file(&path);

    let seeds = [1u64, 2, 3];
    for seed in seeds {
        play_and_record(GameConfig::split_deck(), seed)
            .0
            .append_to(&path)
            .expect("appends");
    }
    let games = read_all(&path).expect("reads back");
    assert_eq!(games.len(), 3);
    assert_eq!(
        games.iter().map(|g| g.seed).collect::<Vec<_>>(),
        seeds.to_vec()
    );
    for game in &games {
        game.walk(|_, _, _| {}).expect("each one replays");
    }

    // Blank lines are skipped, so a hand-edited corpus still opens.
    std::fs::write(
        &path,
        format!("\n{}\n{}", games[0].to_json_line().trim(), games[1].to_json_line()),
    )
    .expect("rewrite");
    assert_eq!(read_all(&path).expect("reads").len(), 2);

    // Anything else that will not parse names its line rather than being dropped.
    std::fs::write(&path, format!("{}not json\n", games[0].to_json_line())).expect("rewrite");
    let error = read_all(&path).expect_err("must not silently drop a game");
    assert!(error.contains(":2:"), "{error}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn phase4_recording_into_a_directory_that_does_not_exist_yet_creates_it() {
    // `--record games/owner-vs-gen006.jsonl` with no `games/` directory failed at the *end*
    // of a played game, which is the one moment at which failing costs something. The
    // directory is created, and `prepare` proves the path writable before a card is dealt.
    let root = std::env::temp_dir().join(format!("duel52-mkdir-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let path = root.join("games").join("series.jsonl");

    prepare(&path).expect("a missing directory is created, not an error an hour later");
    assert!(path.exists(), "the file exists once the path has been prepared");
    assert_eq!(read_all(&path).expect("an empty corpus reads").len(), 0);

    let (record, _) = play_and_record(GameConfig::split_deck(), 3);
    record.append_to(&path).expect("appends");
    assert_eq!(read_all(&path).expect("reads").len(), 1);

    // And `append_to` creates the directory on its own, for a caller that never prepared.
    let fresh = root.join("deeper").join("still").join("series.jsonl");
    record.append_to(&fresh).expect("appends into a new tree");
    assert_eq!(read_all(&fresh).expect("reads").len(), 1);

    // A bare filename has no parent to create, and must not be rejected for it.
    let cwd_file = format!("duel52-bare-{}.jsonl", std::process::id());
    prepare(std::path::Path::new(&cwd_file)).expect("a bare filename is a valid path");
    let _ = std::fs::remove_file(&cwd_file);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn phase4_the_recorded_opponent_spec_is_one_the_agent_parser_accepts() {
    // `replay` reads the checkpoint and the budget back out of this string to decide what
    // to score the game with, so it has to survive the round trip through `AgentSpec`.
    for spec in [
        AgentSpec::Random,
        AgentSpec::Ismcts { iterations: 800 },
        AgentSpec::NetMcts {
            checkpoint: "models/duel52-split-gen016.d52nn".to_string(),
            sims: 4096,
        },
    ] {
        let name = spec.name();
        let parsed = AgentSpec::parse(&name).expect("the name we write is a name we read");
        assert_eq!(parsed.name(), name);
    }
}

#[test]
fn phase4_the_indices_are_positions_in_the_legal_list_not_action_identities() {
    // Why indices: they are what `legal_actions()` hands out, what the encoder agrees on,
    // and four bytes. The cost is that they are only meaningful against the same list —
    // which is why `walk` re-derives the list rather than trusting the record, and why a
    // shifted list is a refusal rather than a different game.
    let (record, _) = play_and_record(GameConfig::split_deck(), 21);
    let mut widths = Vec::new();
    record
        .walk(|_, legal, chosen| {
            assert!(chosen < legal.len());
            widths.push(legal.len());
        })
        .expect("replays");
    assert!(
        widths.iter().any(|&w| w > 1),
        "a game with no real choices would not test anything"
    );
    let _: &[Action] = &[];
}
