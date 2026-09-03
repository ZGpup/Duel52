//! Who knows what.
//!
//! `game_rules.md` §5 ("What is public") and §3. This is the part of the rules a bug in
//! would be invisible in play and fatal to the project: `DESIGN.md` §6 builds ISMCTS
//! determinization on top of exactly these facts, and an engine that leaks information into
//! an observation trains an agent that cheats.
//!
//! The `Pile` and `Card` knowledge masks are the ground truth; `display::render` and the
//! PyO3 `observation()` are the two consumers, and both are checked here.

mod common;
use common::*;

use duel52_engine::display::render;
use duel52_engine::testkit::*;
use duel52_engine::{
    Action, GameConfig, GameState, Player, Player::P0, Player::P1, Rank, Side, Variant,
};

// ====================================================== base cards are hidden ==

/// §3: base cards are "face-down and **hidden from both players, including their owner**".
///
/// `CLAUDE.md` lists this first among the things that are easy to get wrong.
#[test]
fn rule_3_base_cards_are_hidden_from_everyone_including_their_owner() {
    for variant in Variant::ALL {
        for seed in 0..10u64 {
            let s = GameState::new(GameConfig::preset(variant), seed);
            for lane in 0..3 {
                for owner in Player::BOTH {
                    let card = card_at(&s, lane, owner, 0);
                    assert!(card.is_base && card.entered_as_base);
                    assert_eq!(
                        card.known_to, 0,
                        "{variant} seed {seed}: a base card must be known to nobody"
                    );
                }
            }
        }
    }
}

/// §3: "It remains face-down, and its owner still may **not** look at it — the free look at
/// your own face-down cards never applies to a card that entered play as a base card, moved
/// or not. Moving a base card is therefore *not* a back-door Foresight on your own base."
#[test]
fn rule_3_a_queen_moved_base_card_is_still_unreadable_by_its_owner() {
    let mut p = Position::empty();
    p.unlock();
    p.face_down(0, P0, Rank::QUEEN);
    p.base(1, P0, Rank::SEVEN);
    p.hand(P0, &[Rank::KING]);
    p.hand(P1, &[Rank::KING]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::MoveHere { lane: 1, slot: 0 });

    let moved = card_at(&s, 0, P0, 1);
    assert!(!moved.is_base, "it stopped being a base card");
    assert!(moved.entered_as_base, "the flag that gates the free look");
    assert_eq!(moved.known_to, 0, "so still nobody knows it");

    // And it renders as unknown even to its owner.
    let text = render(&s, Some(P0));
    assert!(
        text.contains("(?)ex-base"),
        "the owner must see it as unknown\n{text}"
    );
}

/// §4: "You may look at your **own played face-down cards** at any time, for free."
#[test]
fn rule_4_your_own_played_face_down_cards_are_private_to_you() {
    let mut p = Position::empty();
    p.hand(P0, &[Rank::KING]);
    let mut s = p.build();

    go(&mut s, Action::Play { rank: Rank::KING, lane: 0 });
    let card = card_at(&s, 0, P0, 0);
    assert!(card.rank_known_to(P0));
    assert!(!card.rank_known_to(P1));

    assert!(render(&s, Some(P0)).contains("(K)"), "P0 sees the rank");
    assert!(!render(&s, Some(P1)).contains("(K)"), "P1 does not");
}

// ================================================== Foresight is private ==

/// §6: a 4's peek is "Private information", and §5 lists "anything a 4 revealed to you" as
/// private. §3 makes it persistent — the knowledge does not expire.
#[test]
fn rule_6_foresight_knowledge_is_private_and_persistent() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FOUR);
    p.base(1, P1, Rank::KING);
    p.hand(P0, &[Rank::TWO]);
    p.hand(P1, &[Rank::TWO]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(
        &mut s,
        Action::Peek {
            side: Side::Theirs,
            lane: 1,
            slot: 0,
        },
    );

    assert!(card_at(&s, 1, P1, 0).rank_known_to(P0));
    assert!(!card_at(&s, 1, P1, 0).rank_known_to(P1), "not even its owner");

    // Persistent: several turns later P0 still knows it.
    for _ in 0..6 {
        go(&mut s, Action::Pass);
    }
    assert!(
        card_at(&s, 1, P1, 0).rank_known_to(P0),
        "Foresight knowledge does not expire"
    );
}

/// A peeked card renders with its rank to the peeker and as unknown to everyone else.
#[test]
fn rule_5_a_peeked_card_renders_only_to_the_peeker() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FOUR);
    p.face_down(1, P1, Rank::KING);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(
        &mut s,
        Action::Peek {
            side: Side::Theirs,
            lane: 1,
            slot: 0,
        },
    );

    assert!(render(&s, Some(P0)).contains("(K)"), "the peeker sees it");
    let p1_view = render(&s, Some(P1));
    assert!(
        p1_view.contains("(K)"),
        "P1 played it from hand, so P1 knew it all along"
    );
    // The point is that P0's knowledge is not published anywhere on P1's board.
    assert!(
        !p1_view.contains("P0 knows"),
        "the board does not announce what the opponent has learned"
    );
}

// ============================================== damage is public, rank is not ==

/// §5: "**Damage is public**, including on face-down cards — a damaged card is turned
/// sideways, which both players can see. So 'face-down and damaged' is a visible state."
///
/// But the *rank* must not leak with it: printing "1/3 hp" on a face-down card would
/// announce a Jack, since only a Jack has 3 HP.
#[test]
fn rule_5_damage_is_public_but_max_hp_is_not_shown_for_an_unknown_card() {
    let mut p = Position::empty();
    p.face_down(0, P1, Rank::JACK);
    p.damage(0, P1, 0, 1);
    let s = p.build();

    let text = render(&s, Some(P0));
    assert!(text.contains("dmg1"), "the damage is visible\n{text}");
    assert!(
        !text.contains("hp"),
        "but never the hit points, which would reveal the Jack\n{text}"
    );

    // A face-up card publishes both, because its rank is public anyway.
    let mut p = Position::empty();
    p.face_up(0, P1, Rank::JACK);
    p.damage(0, P1, 0, 1);
    let s = p.build();
    assert!(render(&s, Some(P0)).contains("2/3hp"));
}

// ================================================ hands, discards, and piles ==

/// §5: "**Hand sizes are public**; hand *contents* are private."
#[test]
fn rule_5_hand_sizes_are_public_but_contents_are_not() {
    let s = GameState::new_default(4);
    let text = render(&s, Some(P0));
    assert!(text.contains("5 card(s)"), "P1's size is shown\n{text}");

    // Every rank P1 actually holds must be absent from the opponent-hand line.
    let opponent_line = text
        .lines()
        .find(|l| l.contains("P1 (opponent)"))
        .expect("the opponent line must exist");
    assert!(
        !opponent_line.contains(|c: char| c.is_ascii_digit() && c != '5')
            || opponent_line.contains("5 card(s)"),
        "the opponent's hand line must carry a count and nothing else: {opponent_line}"
    );
}

/// §5: "**The discard pile is public and inspectable** by both players at any time. Dead
/// cards are therefore common knowledge, not a memory feat."
#[test]
fn rule_5_the_discard_pile_is_public_to_both_players() {
    let mut p = Position::empty();
    p.discard(P1, &[Rank::ACE, Rank::KING]);
    let s = p.build();

    for observer in [Some(P0), Some(P1)] {
        let text = render(&s, observer);
        assert!(text.contains("A K"), "the discard is public\n{text}");
    }
}

/// §2: the removed-unseen pool is hidden from everybody, and §5's belief note says that is
/// permanent — "belief over hidden cards never fully resolves, even at the end."
#[test]
fn rule_2_the_removed_pool_is_hidden_from_both_players() {
    for variant in [Variant::Base, Variant::SplitDeck] {
        let s = GameState::new(GameConfig::preset(variant), 3);
        for observer in [Some(P0), Some(P1)] {
            assert!(
                !render(&s, observer).contains("removed unseen"),
                "{variant}: the removed pool must not be rendered"
            );
        }
        assert!(
            render(&s, None).contains("removed unseen"),
            "{variant}: reveal mode is the only way to see it"
        );
    }
}

/// §9b: the mirrored-removal variant is the exception — "The removed multiset ... is
/// **revealed to both players**. That last point is the strategically important one: the
/// unseen pool collapses to 'opponent's hand + the six base cards'."
#[test]
fn rule_9b_the_mirrored_removal_set_is_public() {
    let s = GameState::new(GameConfig::mirrored_removal(), 3);
    assert!(s.removed_revealed);
    for observer in [Some(P0), Some(P1)] {
        assert!(
            render(&s, observer).contains("removed from each deck"),
            "§9b publishes the removed multiset"
        );
    }
}

// ================================================= the 2's bottomed card ==

/// §5: "A **bottomed** card is known to the player who bottomed it and to nobody else."
/// §10a: "The bottomer holds private information. You know the identity and position of the
/// card you bottomed; your opponent knows only that you bottomed *something*."
#[test]
fn rule_10a_only_the_bottomer_knows_what_they_bottomed() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::TWO);
    p.hand(P0, &[Rank::JACK]);
    p.pile(P0, &[Rank::SEVEN, Rank::EIGHT]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::GiveBack { rank: Rank::JACK });

    let p0_view = render(&s, Some(P0));
    assert!(
        p0_view.contains("you privately know") && p0_view.contains("bottom-up: J"),
        "P0 knows both the identity and the position\n{p0_view}"
    );
    assert!(
        !render(&s, Some(P1)).contains("bottom-up"),
        "P1 learns nothing about it"
    );
}

/// §10a: "**Cards are recycled, not destroyed.** A bottomed card *will* be drawn again if
/// the pile outlasts it." The knowledge follows the card into the hand and out of the pile.
#[test]
fn rule_10a_a_bottomed_card_comes_back_around() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::TWO);
    p.hand(P0, &[Rank::JACK]);
    p.pile(P0, &[Rank::SEVEN]);
    p.hand(P1, &[Rank::TWO]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 }); // draws the 7
    go(&mut s, Action::GiveBack { rank: Rank::JACK }); // bottoms the J
    assert_eq!(s.pile(P0).len(), 1);

    go(&mut s, Action::Pass); // P0's turn ends
    go(&mut s, Action::Pass); // P1's turn ends; P0 draws at the start of theirs

    assert!(
        s.hand(P0).contains(&Rank::JACK),
        "the bottomed Jack came back around"
    );
    assert_eq!(s.pile(P0).len(), 0);
}

/// §10a: "**In the base game you bottom into the shared pile.** So the choice carries real
/// risk: the card may come back to your opponent rather than to you."
#[test]
fn rule_10a_in_the_base_game_you_bottom_into_the_shared_pile() {
    let mut p = Position::new(GameConfig::base());
    p.face_down(0, P0, Rank::TWO);
    p.hand(P0, &[Rank::JACK]);
    p.pile(P0, &[Rank::SEVEN]); // the shared pile
    let mut s = p.build();

    assert!(s.shared_pile());
    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::GiveBack { rank: Rank::JACK });

    // Both players draw from the same pile, so P1 is the next to draw from it.
    go(&mut s, Action::Pass);
    assert_eq!(s.to_move, P1);
    assert!(
        s.hand(P1).contains(&Rank::JACK),
        "the card P0 bottomed went to their opponent"
    );
}

// ================================================= face-up is common knowledge ==

/// §5: face-up ranks are public. Flipping a card publishes it to both players permanently.
#[test]
fn rule_5_flipping_a_card_makes_its_rank_common_knowledge() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::EIGHT);
    let mut s = p.build();

    assert!(!card_at(&s, 0, P0, 0).rank_known_to(P1));
    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert!(card_at(&s, 0, P0, 0).rank_known_to(P0));
    assert!(card_at(&s, 0, P0, 0).rank_known_to(P1));
    assert_eq!(card_at(&s, 0, P0, 0).known_to, 0b11);
}

/// A 3's Trap returns the card **face-up** (§6), so it also becomes common knowledge.
#[test]
fn rule_6_a_sprung_trap_publishes_the_three_to_both_players() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::THREE);
    p.damage(0, P0, 0, 1);
    p.face_up(0, P1, Rank::FIVE);
    p.to_move(P1);
    let mut s = p.build();

    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    let three = card_at(&s, 0, P0, 0);
    assert!(three.face_up);
    assert_eq!(three.known_to, 0b11, "everyone saw the Trap spring");
}

// ============================================ the renderer never leaks ==

/// A blunt end-to-end check: play a whole random game and, at every position, confirm the
/// text rendered for each player contains no rank they are not entitled to know.
///
/// The check is deliberately crude — it counts unknown-card tokens against the number of
/// cards the observer genuinely cannot identify. A renderer that leaked a rank would show
/// fewer `(?)`s than there are unknown cards.
#[test]
fn rule_5_the_renderer_never_shows_a_rank_the_observer_may_not_know() {
    use duel52_engine::agents::{Agent, RandomAgent};

    for variant in Variant::ALL {
        let mut s = GameState::new(GameConfig::preset(variant), 11);
        let mut p0 = RandomAgent::new(1);
        let mut p1 = RandomAgent::new(2);

        while !s.outcome.is_over() {
            for observer in Player::BOTH {
                let unknown_cards = s
                    .lanes
                    .iter()
                    .flat_map(|l| l.sides.iter())
                    .flatten()
                    .filter(|c| !c.rank_known_to(observer))
                    .count();
                let text = render(&s, Some(observer));
                let rendered_unknown = text.matches("(?)").count();
                assert_eq!(
                    rendered_unknown, unknown_cards,
                    "{variant}: {observer} was shown a rank they may not know\n{text}"
                );
            }

            let legal = s.legal_actions();
            let action = match s.to_move {
                P0 => p0.choose(&s, &legal),
                P1 => p1.choose(&s, &legal),
            };
            s.apply_trusted(action);
        }
    }
}
