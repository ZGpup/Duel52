//! The analysis corpus: the per-card log, and the two files it is written to.
//!
//! Naming follows `CLAUDE.md` — tests about *machinery* rather than about a rule carry a
//! phase prefix, so a failure reads as "the corpus broke" rather than "the rules broke".
//! `analysis_` is that prefix here.
//!
//! # What these are actually protecting
//!
//! The corpus is the input to every number in `analysis/*.md`, and it is written once and
//! read for hours afterwards. Two failure modes are worth real tests:
//!
//! 1. **A card's life adding up to something impossible** — dying face-up without ever being
//!    flipped, appearing twice, or a flip the aggregate counters saw and the log did not.
//!    None of these crash. They produce a table that is merely wrong.
//! 2. **The file depending on how it was produced.** `--threads` and `--eval-batch` are
//!    speed knobs, and a corpus that differs between two settings of them is a corpus whose
//!    numbers cannot be reproduced. The ladder has had this guarantee since `PLAN.md` §4.2d;
//!    these hold it for the CSV, byte for byte.

use std::collections::HashSet;

use duel52_engine::analysis;
use duel52_engine::probe::{play_spec_game, FaceUpKind, GameStats};
use duel52_engine::{AgentSpec, GameConfig, Player, Rank, Variant};

/// A spread of finished games, cheap enough for the `opt-level = 1` test profile.
fn sample_games(variant: Variant, count: u64) -> Vec<GameStats> {
    let config = GameConfig::preset(variant);
    (1..=count)
        .map(|seed| play_spec_game(config, seed, AgentSpec::Random, AgentSpec::Random))
        .collect()
}

#[test]
fn analysis_every_card_appears_in_the_log_exactly_once() {
    for variant in [Variant::Base, Variant::SplitDeck, Variant::MirroredRemoval] {
        for stats in sample_games(variant, 12) {
            let mut seen = HashSet::new();
            for card in &stats.cards {
                assert!(
                    seen.insert(card.id),
                    "{variant}: card {:?} has two rows in the log",
                    card.id
                );
            }
            // Every row is a card that entered play, and the only two ways in are the deal's
            // base cards and a play from hand.
            let base = stats.cards.iter().filter(|c| c.entered_as_base).count();
            let played = stats.cards.iter().filter(|c| !c.entered_as_base).count();
            let plays: u32 = stats
                .plays_by_rank
                .iter()
                .flat_map(|by_rank| by_rank.iter())
                .sum();
            assert_eq!(
                played, plays as usize,
                "{variant}: {played} non-base rows against {plays} plays counted"
            );
            assert_eq!(
                base,
                2 * GameConfig::preset(variant).lanes,
                "{variant}: one base card per lane per player"
            );
        }
    }
}

/// A card's life has to be internally consistent, and the ways it cannot be are all silent.
#[test]
fn analysis_a_cards_life_is_consistent() {
    for variant in [Variant::Base, Variant::SplitDeck, Variant::MirroredRemoval] {
        for stats in sample_games(variant, 12) {
            for card in &stats.cards {
                assert_eq!(
                    card.face_up_ply.is_none(),
                    card.face_up_kind == FaceUpKind::Never,
                    "{variant}: a flip ply and a flip kind must agree about whether it happened",
                );
                if let Some(up) = card.face_up_ply {
                    assert!(
                        up >= card.entered_ply,
                        "{variant}: flipped on ply {up}, entered on {}",
                        card.entered_ply
                    );
                }
                if let Some(death) = card.death_ply {
                    assert!(death >= card.entered_ply);
                    if card.died_face_up {
                        let up = card
                            .face_up_ply
                            .expect("a card that died face-up was turned face-up first");
                        assert!(up <= death, "{variant}: flipped after it died");
                    }
                    if let Some(up) = card.face_up_ply {
                        assert_eq!(
                            card.died_face_up,
                            up <= death,
                            "{variant}: died_face_up must match the order of the two plies",
                        );
                    }
                } else {
                    assert!(!card.died_face_up, "a living card did not die face-up");
                }
                // §6: a face-down card with a death trigger springs rather than dying, so it
                // is face-up afterwards — and can then be killed again later in the same
                // turn, which is why this is about the flag and not about the plies.
                if card.face_up_kind == FaceUpKind::Trap && card.death_ply.is_some() {
                    assert!(
                        card.died_face_up,
                        "{variant}: a card that sprang its trap was face-up from then on",
                    );
                }
            }
        }
    }
}

/// The per-rank counters are a fold over the log, so they cannot disagree with it — this is
/// what pins them to it rather than to a second tally.
#[test]
fn analysis_the_rank_counters_are_the_card_log_folded() {
    for variant in [Variant::Base, Variant::SplitDeck, Variant::MirroredRemoval] {
        for stats in sample_games(variant, 12) {
            let mut chose = [[0u32; Rank::COUNT]; 2];
            let mut cascade = [[0u32; Rank::COUNT]; 2];
            let mut trap = [[0u32; Rank::COUNT]; 2];
            let mut down_at_end = [[0u32; Rank::COUNT]; 2];
            let mut died_up = [[0u32; Rank::COUNT]; 2];
            let mut died_down = [[0u32; Rank::COUNT]; 2];
            for card in &stats.cards {
                let (p, r) = (card.owner.idx(), card.rank.index());
                match card.face_up_kind {
                    FaceUpKind::Chose => chose[p][r] += 1,
                    FaceUpKind::Cascade => cascade[p][r] += 1,
                    FaceUpKind::Trap => trap[p][r] += 1,
                    FaceUpKind::Never => {}
                }
                match card.death_ply {
                    Some(_) if card.died_face_up => died_up[p][r] += 1,
                    Some(_) => died_down[p][r] += 1,
                    None => {
                        if card.face_up_ply.is_none() {
                            down_at_end[p][r] += 1;
                        }
                    }
                }
            }
            for p in 0..2 {
                assert_eq!(
                    chose[p], stats.flips_by_rank[p],
                    "{variant}: `Chose` is exactly what `Action::Flip` did",
                );
                assert_eq!(cascade[p], stats.flipped_by_cascade_by_rank[p], "{variant}");
                assert_eq!(trap[p], stats.triggers_sprung_by_rank[p], "{variant}");
                assert_eq!(down_at_end[p], stats.face_down_at_end_by_rank[p], "{variant}");
                assert_eq!(died_up[p], stats.died_face_up_by_rank[p], "{variant}");
                assert_eq!(died_down[p], stats.died_face_down_by_rank[p], "{variant}");
            }
        }
    }
}

/// A card is flipped, killed hidden, or still hidden at the end — and it is exactly one of
/// them. This is the question "if a card is face-down, how often does it flip and how often
/// is it killed" made answerable: the three add up to every card that entered play.
#[test]
fn analysis_the_fates_of_a_face_down_card_are_exhaustive() {
    for variant in [Variant::Base, Variant::SplitDeck, Variant::MirroredRemoval] {
        for stats in sample_games(variant, 12) {
            let flipped = stats
                .cards
                .iter()
                .filter(|c| c.face_up_ply.is_some())
                .count();
            let killed_hidden = stats
                .cards
                .iter()
                .filter(|c| c.face_up_ply.is_none() && c.death_ply.is_some())
                .count();
            let still_hidden = stats
                .cards
                .iter()
                .filter(|c| c.face_up_ply.is_none() && c.death_ply.is_none())
                .count();
            assert_eq!(
                flipped + killed_hidden + still_hidden,
                stats.cards.len(),
                "{variant}: the three fates must partition the cards that entered play",
            );
        }
    }
}

/// The opening hand is taken at the start of each player's **own** first turn, so both
/// players are holding the same number of cards when it is taken.
///
/// This is the asymmetry the corpus exists to avoid: `GameState::new` performs P0's opening
/// draw (`game_rules.md` §2), so a naive "hand at setup" gives P0 six cards and P1 five, and
/// every per-rank win rate inherits a first-player edge that has nothing to do with the card.
#[test]
fn analysis_the_opening_hand_is_taken_symmetrically() {
    for variant in [Variant::Base, Variant::SplitDeck, Variant::MirroredRemoval] {
        let config = GameConfig::preset(variant);
        for stats in sample_games(variant, 8) {
            let sizes: Vec<u32> = Player::BOTH
                .iter()
                .map(|p| stats.start_hand[p.idx()].iter().map(|&n| n as u32).sum())
                .collect();
            assert_eq!(sizes[0], sizes[1], "{variant}: both players open on one hand size");
            assert_eq!(
                sizes[0] as usize,
                config.hand_size + 1,
                "{variant}: `hand_size` dealt, plus the draw that opens the turn",
            );
            for p in Player::BOTH {
                let at_unlock: u32 = stats.unlock_hand[p.idx()].iter().map(|&n| n as u32).sum();
                match stats.ply_at_unlock {
                    Some(_) => assert_eq!(
                        at_unlock, stats.hand_at_unlock[p.idx()],
                        "{variant}: the unlock hand's counts must sum to its size",
                    ),
                    // Zeroed, not stale: a game that never unlocked has no unlock hand, and
                    // the corpus writes those columns empty on the strength of this.
                    None => assert_eq!(at_unlock, 0, "{variant}"),
                }
            }
        }
    }
}

/// §5: a declared pair is two cards, so a pair counted by rank is two cards marked in the
/// log — with the caveat that a card can be re-paired after a partner dies, so the marks are
/// a lower bound rather than an equality.
#[test]
fn rule_5_a_declared_pair_marks_both_of_its_cards() {
    let config = GameConfig::preset(Variant::SplitDeck);
    let mut saw_a_pair = false;
    for seed in 1..=40u64 {
        let stats = play_spec_game(config, seed, AgentSpec::Random, AgentSpec::Random);
        for p in Player::BOTH {
            let by_rank: u32 = stats.pairs_by_rank[p.idx()].iter().sum();
            assert_eq!(
                by_rank, stats.pairs_declared[p.idx()],
                "every declared pair is filed under a rank",
            );
            if by_rank == 0 {
                continue;
            }
            saw_a_pair = true;
            let marked = stats
                .cards
                .iter()
                .filter(|c| c.owner == p && c.ever_paired)
                .count();
            assert!(
                marked >= 2,
                "a declared pair marks its two cards, saw {marked}",
            );
        }
    }
    assert!(saw_a_pair, "random play declares pairs; the test measured nothing");
}

// ==================================================== the files on disk ==

fn extract_to(dir: &std::path::Path, threads: usize, eval_batch: usize) {
    analysis::extract(
        GameConfig::preset(Variant::SplitDeck),
        AgentSpec::Random,
        1,
        24,
        threads,
        eval_batch,
        dir,
    )
    .expect("the corpus is written");
}

/// Every value lands in the column the header names it by.
///
/// The gap the other file tests leave open: they check that the rows are rectangular, that
/// the file does not depend on the knobs, and that `meta.json` names the ruleset — and a
/// **transposed pair of columns** passes all three. `hand_at_unlock` and
/// `opp_hand_at_unlock` written the wrong way round would silently invert the hand-size
/// result; `enter_ply` and `faceup_ply` swapped would make every card's tenure negative and
/// the mean meaningless.
///
/// So this replays the same games through [`play_spec_game`] and compares the numbers, which
/// is the one check that has to know the column order.
#[test]
fn analysis_each_value_is_written_under_its_own_column() {
    let config = GameConfig::preset(Variant::SplitDeck);
    let dir = std::env::temp_dir().join("duel52-analysis-columns-mean-what-they-say");
    let _ = std::fs::remove_dir_all(&dir);
    // Game `g` of a corpus is dealt from `first_seed + g / 2`, and game 0 seats agent 0
    // first — which is exactly what `play_spec_game` builds.
    let first_seed = 7;
    analysis::extract(config, AgentSpec::Random, first_seed, 2, 1, 1, &dir)
        .expect("the corpus is written");
    let expected = play_spec_game(config, first_seed, AgentSpec::Random, AgentSpec::Random);

    let games = std::fs::read_to_string(dir.join("games.csv")).expect("games.csv");
    let mut lines = games.lines();
    let header: Vec<&str> = lines.next().expect("a header").split(',').collect();
    let column = |name: &str| header.iter().position(|c| *c == name).expect(name);
    let mut checked = 0;
    for line in lines {
        let row: Vec<&str> = line.split(',').collect();
        if row[column("game")] != "0" {
            continue;
        }
        checked += 1;
        let seat: usize = row[column("seat")].parse().expect("a seat");
        let me = Player::from_index(seat);
        assert_eq!(row[column("seed")], first_seed.to_string());
        assert_eq!(row[column("plies")], expected.plies.to_string());
        assert_eq!(row[column("decisions")], expected.decisions.to_string());
        assert_eq!(
            row[column("result")],
            match expected.outcome {
                duel52_engine::Outcome::Win(w) if w == me => "win",
                duel52_engine::Outcome::Win(_) => "loss",
                _ => "draw",
            }
        );
        // The pair that would invert the hand-size result if it were swapped.
        if let Some(unlock) = expected.ply_at_unlock {
            assert_eq!(row[column("unlock_ply")], unlock.to_string());
            assert_eq!(
                row[column("hand_at_unlock")],
                expected.hand_at_unlock[seat].to_string()
            );
            assert_eq!(
                row[column("opp_hand_at_unlock")],
                expected.hand_at_unlock[1 - seat].to_string()
            );
        }
        assert_eq!(
            row[column("hand_at_end")],
            expected.hand_at_end[seat].to_string()
        );
        assert_eq!(
            row[column("pairs")],
            expected.pairs_declared[seat].to_string()
        );
        for (r, rank) in Rank::ALL.iter().enumerate().take(config.rank_count()) {
            assert_eq!(
                row[column(&format!("start_{rank}"))],
                expected.start_hand[seat][r].to_string(),
                "start_{rank} for seat {seat}",
            );
        }
    }

    assert_eq!(checked, 2, "one row per player, or this test asserted nothing");

    // And the card rows, against the log they were written from.
    let cards = std::fs::read_to_string(dir.join("cards.csv")).expect("cards.csv");
    let mut lines = cards.lines();
    let header: Vec<&str> = lines.next().expect("a header").split(',').collect();
    let column = |name: &str| header.iter().position(|c| *c == name).expect(name);
    let rows: Vec<Vec<&str>> = lines
        .map(|l| l.split(',').collect::<Vec<&str>>())
        .filter(|r| r[column("game")] == "0")
        .collect();
    assert_eq!(rows.len(), expected.cards.len());
    for (row, card) in rows.iter().zip(&expected.cards) {
        assert_eq!(row[column("rank")], card.rank.index().to_string());
        assert_eq!(row[column("owner")], card.owner.idx().to_string());
        assert_eq!(row[column("base")], u8::from(card.entered_as_base).to_string());
        assert_eq!(row[column("enter_ply")], card.entered_ply.to_string());
        assert_eq!(row[column("faceup_kind")], card.face_up_kind.label());
        // An absent value is the empty string, never a zero — `enter_ply` of 0 is a real
        // first-turn play, and a `faceup_ply` of 0 would be too.
        assert_eq!(
            row[column("faceup_ply")],
            card.face_up_ply.map(|p| p.to_string()).unwrap_or_default()
        );
        assert_eq!(
            row[column("death_ply")],
            card.death_ply.map(|p| p.to_string()).unwrap_or_default()
        );
        assert_eq!(
            row[column("died_face_up")],
            card.death_ply
                .map(|_| u8::from(card.died_face_up).to_string())
                .unwrap_or_default()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every row has as many fields as the header has columns. A shifted column is not an error
/// anywhere downstream — it is a table of the wrong numbers.
#[test]
fn analysis_every_row_matches_its_header() {
    let dir = std::env::temp_dir().join("duel52-analysis-columns");
    let _ = std::fs::remove_dir_all(&dir);
    extract_to(&dir, 2, 1);
    for name in ["games", "cards"] {
        let text = std::fs::read_to_string(dir.join(format!("{name}.csv"))).expect("the file");
        let mut lines = text.lines();
        let columns = lines.next().expect("a header").split(',').count();
        let mut rows = 0;
        for line in lines {
            assert_eq!(
                line.split(',').count(),
                columns,
                "{name}.csv: `{line}` has the wrong number of fields",
            );
            rows += 1;
        }
        assert!(rows > 0, "{name}.csv has no rows");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The corpus must not depend on how many workers wrote it, or on how many games were in
/// flight while they did. Byte-for-byte, deliberately: a tolerance would pass exactly the
/// reordering these knobs must not cause.
#[test]
fn analysis_the_corpus_does_not_depend_on_threads_or_eval_batch() {
    let root = std::env::temp_dir().join("duel52-analysis-knobs");
    let _ = std::fs::remove_dir_all(&root);
    let baseline = root.join("t1b1");
    extract_to(&baseline, 1, 1);

    for (threads, batch) in [(4, 1), (1, 8), (3, 8)] {
        let other = root.join(format!("t{threads}b{batch}"));
        extract_to(&other, threads, batch);
        for name in ["games.csv", "cards.csv"] {
            let a = std::fs::read(baseline.join(name)).expect("the baseline file");
            let b = std::fs::read(other.join(name)).expect("the other file");
            assert!(
                a == b,
                "{name} differs at --threads {threads} --eval-batch {batch}",
            );
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// A corpus names the ruleset it was played under. `MODULAR_RULES.md` §6: a shard from
/// another ruleset is indistinguishable on shape alone, and the same is true of a CSV — the
/// encoder is rank-agnostic, so nothing about the file says which rules produced it except
/// this field.
#[test]
fn analysis_meta_names_the_ruleset() {
    let dir = std::env::temp_dir().join("duel52-analysis-meta");
    let _ = std::fs::remove_dir_all(&dir);
    extract_to(&dir, 2, 1);
    let meta = std::fs::read_to_string(dir.join("meta.json")).expect("meta.json");
    let config = GameConfig::preset(Variant::SplitDeck);
    assert!(meta.contains(&format!("\"rules_hash\": \"{:016x}\"", config.rules_hash())));
    assert!(meta.contains("\"agent\": \"random\""));
    assert!(meta.contains(&format!("\"schema\": {}", analysis::SCHEMA)));
    let _ = std::fs::remove_dir_all(&dir);
}
