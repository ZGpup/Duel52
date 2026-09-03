//! One named test per card-power ruling in `game_rules.md` §6 (plus the §8 sequencing
//! notes that belong to a specific power).
//!
//! The constant powers — 8 Retaliate, 9 Nimble, 10 Twinstrike, J Taunt — live in
//! `rules_combat.rs`, because every ruling about them is a ruling about an attack.

mod common;
use common::*;

use duel52_engine::testkit::*;
use duel52_engine::{Action, Phase, Player::P0, Player::P1, Rank, Side, TwoPower};

// =============================================================== A · Action ==

/// §6: "Gain **1 action** this turn, usable however you like."
#[test]
fn rule_6_ace_grants_one_extra_action_on_flip() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::ACE);
    let mut s = p.build();

    assert_eq!(s.actions_remaining, 3);
    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    // One action spent on the flip, one granted back.
    assert_eq!(s.actions_remaining, 3, "the Ace must refund the flip");
}

/// §6: "On the turn it is flipped, the Ace itself **may attack twice** — genuinely two
/// attacks, at two targets or the same one twice."
#[test]
fn rule_6_ace_may_attack_twice_on_the_turn_it_is_flipped() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::ACE);
    p.face_up(0, P1, Rank::FOUR);
    p.face_up(0, P1, Rank::FIVE);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(card_at(&s, 0, P0, 0).attack_allowance, 2);

    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 1,
        },
    );
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P1, 1), 1);

    // A third attack is not available: the allowance is 2, not unlimited.
    refuse(
        &s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
}

/// §6: "Each attack costs its own action; the exception is to the once-per-card limit, not
/// to the action cost."
#[test]
fn rule_6_ace_second_attack_costs_its_own_action() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::ACE);
    p.face_up(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 }); // 3 - 1 + 1 = 3
    assert_eq!(s.actions_remaining, 3);
    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    assert_eq!(s.actions_remaining, 2);
    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    assert_eq!(s.actions_remaining, 1);
}

/// §6: the double attack belongs to "the turn it is flipped". A later turn is one attack.
#[test]
fn rule_6_ace_double_attack_expires_at_the_end_of_the_turn() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::ACE);
    p.face_up(0, P1, Rank::FOUR);
    p.hand(P1, &[Rank::TWO]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(card_at(&s, 0, P0, 0).attack_allowance, 2);
    go(&mut s, Action::Pass); // P0 ends the turn
    go(&mut s, Action::Pass); // P1 ends the turn; back to P0
    assert_eq!(s.to_move, P0);
    assert_eq!(
        card_at(&s, 0, P0, 0).attack_allowance,
        1,
        "the Ace's second attack was for that turn only"
    );
}

/// §6: "The double attack attaches to the **flip**, not to who caused it: an Ace flipped by
/// a 5 gets it the same as one flipped by its owner's own action, and gets the +1 action
/// too."
#[test]
fn rule_6_ace_flipped_by_a_five_still_gets_the_action_and_the_double_attack() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FIVE);
    p.face_down(0, P0, Rank::ACE);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 }); // flip the 5: 3 -> 2
    assert_eq!(s.actions_remaining, 2);
    assert_eq!(s.phase(), Phase::ResolveOrder);
    resolve_only(&mut s); // the 5 flips the Ace

    assert!(card_at(&s, 0, P0, 1).face_up);
    assert_eq!(card_at(&s, 0, P0, 1).attack_allowance, 2);
    assert_eq!(s.actions_remaining, 3, "the Ace grants its action either way");
}

// ================================================================= 2 · View ==

/// §6 + §10a house rule: "Draw a card, then put a card from your hand on the bottom of your
/// draw pile — a scry, not a discard."
#[test]
fn rule_10a_two_draws_then_bottoms_a_card_from_hand() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::TWO);
    p.hand(P0, &[Rank::ACE]);
    p.pile(P0, &[Rank::SEVEN, Rank::EIGHT, Rank::NINE]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::GiveBack);
    assert_eq!(
        s.hand(P0),
        &vec![Rank::ACE, Rank::SEVEN],
        "the draw happens before the give-back"
    );

    go(&mut s, Action::GiveBack { rank: Rank::ACE });
    assert_eq!(s.hand(P0), &vec![Rank::SEVEN]);
    assert_eq!(
        s.pile(P0).len(),
        3,
        "the house 2 is pile-neutral: one out, one in"
    );
    assert!(
        s.discards[P0.idx()].is_empty(),
        "the house 2 does not discard"
    );
}

/// §6: "You may bottom the card you just drew."
#[test]
fn rule_10a_two_may_give_back_the_card_it_just_drew() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::TWO);
    p.hand(P0, &[Rank::ACE]);
    p.pile(P0, &[Rank::SEVEN, Rank::EIGHT]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    allow(&s, Action::GiveBack { rank: Rank::SEVEN });
    go(&mut s, Action::GiveBack { rank: Rank::SEVEN });
    assert_eq!(s.hand(P0), &vec![Rank::ACE]);
}

/// §5: "A **bottomed** card is known to the player who bottomed it and to nobody else. They
/// know both its identity and its position — the bottom of that pile."
#[test]
fn rule_5_a_bottomed_card_is_known_only_to_the_player_who_bottomed_it() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::TWO);
    p.hand(P0, &[Rank::ACE]);
    p.pile(P0, &[Rank::SEVEN, Rank::EIGHT]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::GiveBack { rank: Rank::ACE });

    let pile = s.pile(P0);
    assert_eq!(
        pile.known_from_bottom(P0).first(),
        Some(&Some(Rank::ACE)),
        "P0 knows the bottom card is their Ace"
    );
    assert_eq!(
        pile.known_from_bottom(P1).first(),
        Some(&None),
        "P1 knows only that something was bottomed"
    );
}

/// §6 + §9: the 2 is "gated on the pile **you** draw from; if that pile is empty the power
/// does **nothing at all** — no draw, and no bottoming either, so it cannot be used to
/// refill an empty pile." So a 2 can go dead a turn before the *global* unlock.
#[test]
fn rule_9_two_is_gated_on_your_own_pile_not_the_global_unlock() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::TWO);
    p.hand(P0, &[Rank::ACE]);
    p.pile(P0, &[]); // P0's own pile is empty
    p.pile(P1, &[Rank::KING, Rank::KING]); // but P1's is not, so base stays locked
    let mut s = p.build();

    assert!(!s.base_unlocked, "the global unlock needs *both* piles empty");
    go(&mut s, Action::Flip { lane: 0, slot: 0 });

    assert_eq!(s.phase(), Phase::Main, "the power fizzled entirely");
    assert_eq!(s.hand(P0), &vec![Rank::ACE], "no draw");
    assert!(s.discards[P0.idx()].is_empty(), "and no give-back");
    assert_eq!(s.pile(P0).len(), 0, "the 2 cannot refill an empty pile");
}

/// §10a: `two_power = discard` is rules-as-written. The card leaves the game and the pile
/// shrinks — the parity lever the house rule exists to remove.
#[test]
fn rule_10a_raw_two_discards_and_shrinks_the_pile() {
    let mut config = duel52_engine::GameConfig::split_deck();
    config.two_power = TwoPower::Discard;

    let mut p = Position::new(config);
    p.face_down(0, P0, Rank::TWO);
    p.hand(P0, &[Rank::ACE]);
    p.pile(P0, &[Rank::SEVEN, Rank::EIGHT, Rank::NINE]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::GiveBack { rank: Rank::ACE });

    assert_eq!(s.hand(P0), &vec![Rank::SEVEN]);
    assert_eq!(s.pile(P0).len(), 2, "RAW: one drawn, nothing put back");
    assert_eq!(discard_ranks(&s, P0), vec![Rank::ACE]);
}

// ================================================================= 3 · Trap ==

/// §6: "If killed **while face-down**, it **returns to play face-up** with full 2 HP
/// instead of dying — immediately, in **the same lane**."
#[test]
fn rule_6_three_killed_face_down_returns_face_up_at_full_hp() {
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
    assert!(three.face_up, "the Trap returns it face-up");
    assert_eq!(three.damage, 0, "at full HP");
    assert_eq!(three.rank, Rank::THREE);
    assert!(
        discard_ranks(&s, P0).is_empty(),
        "it did not die, so nothing was discarded"
    );
}

/// §6: "it returns face-up so the Trap **cannot re-trigger**."
#[test]
fn rule_6_three_trap_cannot_retrigger_because_it_returns_face_up() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::THREE);
    p.damage(0, P0, 0, 1);
    // Three separate attackers: each card may attack only once per turn (§4).
    p.face_up(0, P1, Rank::FOUR);
    p.face_up(0, P1, Rank::FIVE);
    p.face_up(0, P1, Rank::SIX);
    p.to_move(P1);
    let mut s = p.build();

    for attacker in 0..3 {
        go(
            &mut s,
            Action::Attack {
                lane: 0,
                attacker,
                target: 0,
            },
        );
    }
    assert_eq!(occupancy(&s, 0, P0), 0, "the second death is permanent");
    assert_eq!(discard_ranks(&s, P0), vec![Rank::THREE]);
}

/// §6: "A base 3 killed post-unlock triggers this too, and returns as a normal (non-base)
/// card." §3 says the same from the other direction.
#[test]
fn rule_3_base_three_killed_post_unlock_returns_as_a_normal_card() {
    let mut p = Position::empty();
    p.unlock();
    p.base(0, P0, Rank::THREE);
    p.damage(0, P0, 0, 1);
    p.face_up(0, P1, Rank::FIVE);
    // Both hands non-empty so that emptying a lane cannot end the game mid-test (§7).
    p.hand(P0, &[Rank::KING]);
    p.hand(P1, &[Rank::KING]);
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
    assert_eq!(three.damage, 0);
    assert!(!three.is_base, "it comes back as a normal card");
    assert!(
        three.entered_as_base,
        "but the engine still remembers it entered as a base card"
    );
}

/// §6: the Trap is conditional on being face-down. A face-up 3 is just a 3.
#[test]
fn rule_6_a_face_up_three_simply_dies() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::THREE);
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
    assert_eq!(occupancy(&s, 0, P0), 0);
    assert_eq!(discard_ranks(&s, P0), vec![Rank::THREE]);
}

// ============================================================ 4 · Foresight ==

/// §6: "Look at any one face-down card on the board — including base cards, yours or your
/// opponent's. Private information."
#[test]
fn rule_6_four_reveals_a_card_privately_to_the_peeker_only() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FOUR);
    p.face_down(0, P1, Rank::KING);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::Foresight);
    go(
        &mut s,
        Action::Peek {
            side: Side::Theirs,
            lane: 0,
            slot: 0,
        },
    );

    let target = card_at(&s, 0, P1, 0);
    assert!(target.rank_known_to(P0), "the peeker learns it");
    assert!(!target.face_up, "peeking does not flip it");
    assert!(
        target.rank_known_to(P1),
        "its owner already knew — they played it from hand (§4)"
    );
}

/// §3 + §6: base cards are hidden from their owner too, which is precisely why the 4 can
/// usefully target *your own* base card.
#[test]
fn rule_6_four_may_peek_at_your_own_base_card() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FOUR);
    p.base(1, P0, Rank::QUEEN);
    let mut s = p.build();

    assert!(
        !card_at(&s, 1, P0, 0).rank_known_to(P0),
        "you do not know your own base card"
    );

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    allow(
        &s,
        Action::Peek {
            side: Side::Mine,
            lane: 1,
            slot: 0,
        },
    );
    go(
        &mut s,
        Action::Peek {
            side: Side::Mine,
            lane: 1,
            slot: 0,
        },
    );

    let base = card_at(&s, 1, P0, 0);
    assert!(base.rank_known_to(P0), "now you do");
    assert!(!base.rank_known_to(P1), "the opponent still does not");
    assert!(base.is_base, "and it is still a base card");
}

/// §8: "A power with **no legal target simply fizzles**, and the flip remains a **legal
/// action**."
#[test]
fn rule_8_four_with_no_face_down_card_fizzles_and_the_flip_is_still_legal() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FOUR);
    let mut s = p.build();

    allow(&s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::Main, "nothing left face-down to peek at");
    assert_eq!(s.actions_remaining, 2, "the flip still cost an action");
}

// ================================================================= 5 · Flip ==

/// §6: "Flip **all your face-down cards in its lane**."
#[test]
fn rule_6_five_flips_all_your_face_down_cards_in_its_lane() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FIVE);
    p.face_down(0, P0, Rank::SEVEN);
    p.face_down(0, P0, Rank::EIGHT);
    p.face_down(1, P0, Rank::NINE); // another lane: untouched
    p.face_down(0, P1, Rank::TEN); // the opponent's card: untouched
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::ResolveNext { lane: 0, slot: 1 });
    go(&mut s, Action::ResolveNext { lane: 0, slot: 2 });

    assert!(card_at(&s, 0, P0, 1).face_up);
    assert!(card_at(&s, 0, P0, 2).face_up);
    assert!(!card_at(&s, 1, P0, 0).face_up, "a 5 works in its own lane only");
    assert!(!card_at(&s, 0, P1, 0).face_up, "and only on your own cards");
}

/// §8: "A 5 resolving in the lane simply **skips** frozen cards; they are untouchable, not
/// merely passive."
#[test]
fn rule_8_five_skips_frozen_cards() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FIVE);
    p.face_down(0, P0, Rank::SEVEN);
    p.face_down(0, P0, Rank::EIGHT);
    p.freeze(0, P0, 2, /* through ply */ 0);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    resolve_only(&mut s); // only the 7 is available

    assert!(card_at(&s, 0, P0, 1).face_up);
    assert!(!card_at(&s, 0, P0, 2).face_up, "the frozen 8 was skipped");
    assert_eq!(s.phase(), Phase::Main);
}

/// §3 + §8: base cards "cannot be flipped" while any pile is non-empty, and the 5 does not
/// get around that.
#[test]
fn rule_3_five_cannot_flip_a_base_card_before_the_unlock() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FIVE);
    p.base(0, P0, Rank::SEVEN);
    let mut s = p.build();

    refuse(&s, Action::Flip { lane: 0, slot: 1 });
    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::Main, "nothing for the 5 to flip");
    assert!(!card_at(&s, 0, P0, 1).face_up);
}

/// §6 + §8: "Includes your base card in that lane **once the draw pile is empty**" — and it
/// is all-or-nothing, which is what makes a held 5 committal in the endgame.
#[test]
fn rule_6_five_flips_your_base_card_once_the_pile_is_empty() {
    let mut p = Position::empty();
    p.unlock();
    p.face_down(0, P0, Rank::FIVE);
    p.base(0, P0, Rank::SEVEN);
    p.hand(P0, &[Rank::KING]);
    p.hand(P1, &[Rank::KING]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    resolve_only(&mut s);
    assert!(
        card_at(&s, 0, P0, 1).face_up,
        "post-unlock the 5 forces the base card up whether you want it or not"
    );
}

/// §8: "**Resolution order is always the acting player's choice**, and always **adaptive**."
/// With two cards queued, both orders are offered.
#[test]
fn rule_8_five_resolution_order_is_the_players_choice() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::FIVE);
    p.face_down(0, P0, Rank::SEVEN);
    p.face_down(0, P0, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    let legal = s.legal_actions();
    assert_eq!(legal.len(), 2, "both queued cards are offered: {legal:?}");
    assert!(legal.contains(&Action::ResolveNext { lane: 0, slot: 1 }));
    assert!(legal.contains(&Action::ResolveNext { lane: 0, slot: 2 }));

    // Take the 8 first; the 7 is still owed afterwards.
    go(&mut s, Action::ResolveNext { lane: 0, slot: 2 });
    assert_eq!(s.phase(), Phase::ResolveOrder);
    assert_eq!(s.legal_actions(), vec![Action::ResolveNext { lane: 0, slot: 1 }]);
}

// =============================================================== 6 · Freeze ==

/// §6: "**All enemy cards in the lane are frozen**." Per card, and in that lane only.
#[test]
fn rule_6_six_freezes_enemy_cards_in_its_lane() {
    let mut p = Position::empty();
    p.ply(4);
    p.face_down(0, P0, Rank::SIX);
    p.face_up(0, P1, Rank::FOUR);
    p.face_down(0, P1, Rank::SEVEN);
    p.face_up(1, P1, Rank::FOUR); // another lane
    p.face_up(0, P0, Rank::FOUR); // your own card
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });

    assert!(card_at(&s, 0, P1, 0).is_frozen(4));
    assert!(card_at(&s, 0, P1, 1).is_frozen(4));
    assert!(!card_at(&s, 1, P1, 0).is_frozen(4), "other lanes are untouched");
    assert!(!card_at(&s, 0, P0, 1).is_frozen(4), "your own cards are safe");
}

/// §6: "**Cannot freeze a 9**, ever, including a 9 already in the lane when the 6 resolves."
///
/// **[ASSUMED]** The immunity is Nimble, and §6 says "Powers are inert while a card is
/// face-down", so a *face-down* 9 can be frozen. The rulebook's "ever" is about timing, not
/// about face-up-ness.
#[test]
fn rule_6_six_cannot_freeze_a_face_up_nine() {
    let mut p = Position::empty();
    p.ply(4);
    p.face_down(0, P0, Rank::SIX);
    p.face_up(0, P1, Rank::NINE);
    p.face_down(0, P1, Rank::NINE);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert!(!card_at(&s, 0, P1, 0).is_frozen(4), "Nimble, face-up");
    assert!(
        card_at(&s, 0, P1, 1).is_frozen(4),
        "[ASSUMED] a face-down 9's Nimble is inert, so it freezes"
    );
}

/// §8: "an enemy card frozen by a 6 is unfrozen at the **end of the frozen player's next
/// turn** — so exactly one of their turns is lost."
#[test]
fn rule_8_freeze_lasts_exactly_one_of_the_victims_turns() {
    let mut p = Position::empty();
    p.ply(4);
    p.face_down(0, P0, Rank::SIX);
    p.face_up(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    let card = card_at(&s, 0, P1, 0);
    assert!(card.is_frozen(4), "frozen now (P0's turn)");
    assert!(card.is_frozen(5), "and through P1's next turn");
    assert!(!card.is_frozen(6), "thawed at the end of that turn");
}

/// §8: "Freeze blocks exactly two things: **attacking**, and **being flipped** — by anyone."
#[test]
fn rule_8_freeze_blocks_attacking_and_being_flipped() {
    let mut p = Position::empty();
    p.ply(5);
    p.to_move(P1);
    p.face_up(0, P1, Rank::FOUR);
    p.face_down(0, P1, Rank::SEVEN);
    p.freeze(0, P1, 0, 5);
    p.freeze(0, P1, 1, 5);
    p.face_up(0, P0, Rank::FOUR); // something to attack
    let s = p.build();

    refuse(
        &s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    refuse(&s, Action::Flip { lane: 0, slot: 1 });
}

/// §8: "Cards that enter the lane *after* the 6 resolves are **not** frozen."
#[test]
fn rule_8_freeze_does_not_catch_cards_that_arrive_later() {
    let mut p = Position::empty();
    p.ply(4);
    p.face_down(0, P0, Rank::SIX);
    p.face_up(0, P1, Rank::FOUR);
    p.hand(P1, &[Rank::TEN]);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::Pass); // P0's turn ends; P1 is now to move on ply 5

    assert_eq!(s.to_move, P1);
    go(&mut s, Action::Play { rank: Rank::TEN, lane: 0 });
    let fresh = s.lanes[0].side(P1).last().unwrap();
    assert_eq!(fresh.rank, Rank::TEN);
    assert!(!fresh.is_frozen(5), "it arrived after the 6 resolved");
    assert!(
        card_at(&s, 0, P1, 0).is_frozen(5),
        "the card that was there is still frozen"
    );
}

/// §8: "Freeze does **not** block **reactivation**: a face-up frozen card that a King
/// empowers fires its power normally. It still cannot attack."
#[test]
fn rule_8_a_king_can_reactivate_a_frozen_card() {
    let mut p = Position::empty();
    p.ply(5);
    p.to_move(P1);
    p.face_up(0, P1, Rank::ACE);
    p.freeze(0, P1, 0, 5);
    p.face_down(0, P1, Rank::KING);
    p.face_up(0, P0, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 1 }); // flip the King: 3 -> 2
    resolve_only(&mut s); // reactivate the frozen Ace
    assert_eq!(
        s.actions_remaining, 3,
        "the frozen Ace still grants its action"
    );
    refuse(
        &s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
}

// ============================================================= 7 · Heal All ==

/// §6: "**Heal all your damaged cards 2 HP**, in all lanes, face-up and face-down."
#[test]
fn rule_6_seven_heals_all_your_damaged_cards_in_every_lane() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::FOUR);
    p.damage(0, P0, 1, 1);
    p.face_down(1, P0, Rank::EIGHT);
    p.damage(1, P0, 0, 1);
    p.face_up(0, P1, Rank::FOUR);
    p.damage(0, P1, 0, 1);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });

    assert_eq!(damage_at(&s, 0, P0, 1), 0, "face-up, same lane");
    assert_eq!(damage_at(&s, 1, P0, 0), 0, "face-down, another lane");
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "the opponent is not healed");
}

/// §6: "Healing is capped at the card's maximum HP — a Jack on 1 HP heals to 3, not 5."
#[test]
fn rule_6_seven_healing_is_capped_at_max_hp() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::JACK);
    p.damage(0, P0, 1, 2); // a Jack on 1 of 3 HP
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    let jack = card_at(&s, 0, P0, 1);
    assert_eq!(jack.damage, 0);
    assert_eq!(jack.hp_remaining(), 3, "healed to 3, not to 5");
}

/// §6: the 7 "Includes base cards **once the draw pile is empty**".
#[test]
fn rule_6_seven_heals_base_cards_only_after_the_unlock() {
    // Post-unlock: healed.
    let mut p = Position::empty();
    p.unlock();
    p.face_down(0, P0, Rank::SEVEN);
    p.base(1, P0, Rank::NINE);
    p.damage(1, P0, 0, 1);
    p.hand(P0, &[Rank::KING]);
    p.hand(P1, &[Rank::KING]);
    let mut s = p.build();
    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(damage_at(&s, 1, P0, 0), 0);

    // Pre-unlock: the gate holds. A damaged base card cannot actually arise before the
    // unlock (base cards are untouchable, §3), so this position is constructed rather than
    // reachable — it exists to pin the gate itself.
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::SEVEN);
    p.base(1, P0, Rank::NINE);
    p.damage(1, P0, 0, 1);
    let mut s = p.build();
    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(damage_at(&s, 1, P0, 0), 1, "still gated on the unlock");
}

// ================================================================= Q · Move ==

/// §6: "**Move one allied card from another lane into the Queen's lane**, face-down or
/// face-up. The moved card **keeps its damage**, does **not** reactivate its one-shot
/// power, keeps constant powers."
#[test]
fn rule_6_queen_moves_an_allied_card_and_it_keeps_its_damage() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::QUEEN);
    p.face_up(1, P0, Rank::NINE);
    p.damage(1, P0, 0, 1);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::QueenSource);
    go(&mut s, Action::MoveHere { lane: 1, slot: 0 });

    assert_eq!(occupancy(&s, 1, P0), 0);
    let moved = card_at(&s, 0, P0, 1);
    assert_eq!(moved.rank, Rank::NINE);
    assert_eq!(moved.damage, 1, "damage travels with the card");
    assert!(moved.face_up);
}

/// §6: the moved card "does **not** reactivate its one-shot power".
#[test]
fn rule_6_queen_moved_card_does_not_refire_its_one_shot_power() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::QUEEN);
    p.face_up(1, P0, Rank::FOUR); // a 4 would open a Foresight node if it refired
    p.face_down(2, P0, Rank::EIGHT); // a legal peek target, so a refire would be visible
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::MoveHere { lane: 1, slot: 0 });
    assert_eq!(s.phase(), Phase::Main, "the 4 did not fire again");
}

/// §6: the moved card "**may attack after the move** if it has not already attacked this
/// turn".
#[test]
fn rule_6_queen_moved_card_may_attack_after_the_move() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::QUEEN);
    p.face_up(1, P0, Rank::FIVE);
    p.face_up(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::MoveHere { lane: 1, slot: 0 });
    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 1,
            target: 0,
        },
    );
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
}

/// §6: ...and may **not** if it already attacked.
#[test]
fn rule_6_queen_moved_card_cannot_attack_twice_in_a_turn() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::QUEEN);
    p.face_up(1, P0, Rank::FIVE);
    p.spent(1, P0, 0);
    p.face_up(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::MoveHere { lane: 1, slot: 0 });
    refuse(
        &s,
        Action::Attack {
            lane: 0,
            attacker: 1,
            target: 0,
        },
    );
}

/// §3: "A base card that a Queen moves to another lane **stops being a base card**... It
/// remains face-down, and its owner still may **not** look at it."
#[test]
fn rule_3_queen_moved_base_card_stops_being_a_base_card_but_stays_hidden() {
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
    assert!(!moved.is_base, "no longer a base card");
    assert!(moved.entered_as_base, "but it entered as one");
    assert!(!moved.face_up);
    assert!(
        !moved.rank_known_to(P0),
        "moving a base card is not a back-door Foresight on your own base"
    );
    assert!(!moved.rank_known_to(P1));
}

/// §6: a Queen "Can move a base card **once the draw pile is empty**" — and not before.
#[test]
fn rule_3_queen_cannot_move_a_base_card_before_the_unlock() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::QUEEN);
    p.base(1, P0, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(
        s.phase(),
        Phase::Main,
        "the only card elsewhere is a locked base card, so the Queen fizzles"
    );
}

/// §8: "a frozen card a Queen moves to another lane **stays frozen** ... A Queen is
/// therefore not an escape hatch from a 6 — she relocates the problem."
#[test]
fn rule_8_queen_moved_card_stays_frozen() {
    let mut p = Position::empty();
    p.ply(5);
    p.to_move(P1);
    p.face_down(0, P1, Rank::QUEEN);
    p.face_up(1, P1, Rank::FIVE);
    p.freeze(1, P1, 0, 5);
    p.face_up(0, P0, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::MoveHere { lane: 1, slot: 0 });

    assert!(card_at(&s, 0, P1, 1).is_frozen(5), "the freeze travels");
    refuse(
        &s,
        Action::Attack {
            lane: 0,
            attacker: 1,
            target: 0,
        },
    );
}

/// §8: "a Queen with no allied card elsewhere" fizzles, and "often that is precisely why
/// you flip her: a Queen with no move available is still a body that can attack."
#[test]
fn rule_8_queen_with_no_allied_card_elsewhere_fizzles() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::QUEEN);
    p.face_up(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::Main);
    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "she is still a body that attacks");
}

/// §6: a Queen may not move herself — the source must be in *another* lane.
#[test]
fn rule_6_queen_cannot_move_herself() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::QUEEN);
    p.face_up(0, P0, Rank::FIVE); // same lane: not a legal source
    p.face_up(1, P0, Rank::SIX);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(
        s.legal_actions(),
        vec![Action::MoveHere { lane: 1, slot: 0 }],
        "only the other lane's card is a legal source"
    );
}

// ============================================================== K · Empower ==

/// §6: "All your face-up cards in this lane reactivate their powers", and the King+Ace
/// ruling: "A King reactivating an Ace **does grant another action** — **once**."
#[test]
fn rule_6_king_reactivates_ace_grants_one_action() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::ACE);
    p.face_down(0, P0, Rank::KING);
    let mut s = p.build();

    assert_eq!(s.actions_remaining, 3);
    go(&mut s, Action::Flip { lane: 0, slot: 1 }); // 3 -> 2
    assert_eq!(s.actions_remaining, 2);
    resolve_only(&mut s); // reactivate the Ace
    assert_eq!(s.actions_remaining, 3, "+1 action, once");
    assert_eq!(s.phase(), Phase::Main, "and it does not repeat");
}

/// §6: "The reactivated Ace also **regains its double attack** for that turn, as a **reset,
/// not a stack** ... An Ace that attacked once, then got Kinged, tops out at **three**
/// attacks that turn, not four."
#[test]
fn rule_6_king_resets_the_aces_attack_counter_rather_than_stacking_it() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::ACE);
    p.spent(0, P0, 0); // the Ace has already attacked once this turn
    p.face_down(0, P0, Rank::KING);
    p.face_up(0, P1, Rank::JACK); // 3 HP, so it survives to be hit repeatedly
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 1 });
    resolve_only(&mut s);

    let ace = card_at(&s, 0, P0, 0);
    assert_eq!(ace.attacks_used, 0, "counter reset to zero");
    assert_eq!(ace.attack_allowance, 2, "with an allowance of 2");

    // Two more attacks, and no third: one before the King plus two after is three.
    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    refuse(
        &s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
}

/// §6: "Does **not** affect other Kings." This is what makes an infinite loop impossible.
#[test]
fn rule_6_king_does_not_reactivate_other_kings() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::KING);
    p.face_down(0, P0, Rank::KING);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 1 });
    assert_eq!(
        s.phase(),
        Phase::Main,
        "the other King is not a reactivation target, so nothing is queued"
    );
}

/// §6: "Does **not** affect constant powers (8, 9, 10, J)" — nor the conditional 3.
#[test]
fn rule_6_king_does_not_affect_constant_powers_or_a_face_up_three() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::EIGHT);
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P0, Rank::JACK);
    p.face_up(0, P0, Rank::THREE);
    p.face_down(0, P0, Rank::KING);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 5 });
    assert_eq!(s.phase(), Phase::Main, "nothing here is reactivatable");
}

/// §6: "All your face-up cards in **this lane**" — the King does not reach across lanes,
/// and does not touch the opponent.
#[test]
fn rule_6_king_only_empowers_its_own_lane_and_its_own_side() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::KING);
    p.face_up(1, P0, Rank::ACE); // another lane
    p.face_up(0, P1, Rank::ACE); // the opponent's lane-0 card
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::Main);
    assert_eq!(s.actions_remaining, 2, "no Ace fired");
}

/// §6 + §8: a King reactivating a 5 makes the 5 flip the lane, and the nested resolution
/// finishes before control returns. This is the cascade §8 calls out by name.
#[test]
fn rule_8_king_reactivating_a_five_cascades_correctly() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FIVE);
    p.face_up(0, P0, Rank::ACE);
    p.face_down(0, P0, Rank::SEVEN); // only the 5 can flip this
    p.face_down(0, P0, Rank::KING);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 3 }); // flip the King
    assert_eq!(s.phase(), Phase::ResolveOrder);

    // The King queues the 5 and the Ace. Resolve the 5 first; its flip list lands on top.
    go(&mut s, Action::ResolveNext { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::ResolveOrder);
    assert_eq!(
        s.legal_actions(),
        vec![Action::ResolveNext { lane: 0, slot: 2 }],
        "the 5's own list resolves before the King's remaining list"
    );
    go(&mut s, Action::ResolveNext { lane: 0, slot: 2 });
    assert!(card_at(&s, 0, P0, 2).face_up);

    // Back to the King's list: the Ace is still owed.
    assert_eq!(
        s.legal_actions(),
        vec![Action::ResolveNext { lane: 0, slot: 1 }]
    );
    go(&mut s, Action::ResolveNext { lane: 0, slot: 1 });
    assert_eq!(s.actions_remaining, 3, "3 - 1 for the King + 1 from the Ace");
}
