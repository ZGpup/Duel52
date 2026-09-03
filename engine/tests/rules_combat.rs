//! Combat: damage, pairs, and the four constant powers.
//!
//! `game_rules.md` §5 (combat and pairs), the 8/9/10/J rows of §6, and the combat
//! sequencing notes of §8. This is the densest part of the rules — `DESIGN.md` §2 gives it
//! as the reason the engine is in Rust at all: "Rule interactions are dense with
//! conditionals (8-retaliate × 9-nimble × 10-twinstrike × J-taunt)."

mod common;
use common::*;

use duel52_engine::testkit::*;
use duel52_engine::{Action, Phase, Player::P0, Player::P1, Rank};

/// Attack from P0's slot `a` to P1's slot `t` in lane 0.
fn atk(a: u8, t: u8) -> Action {
    Action::Attack {
        lane: 0,
        attacker: a,
        target: t,
    }
}

// ================================================== basic damage and hit points ==

/// §5: "Every card has **2 hit points**, except the Jack, which has **3**." "A normal attack
/// deals **1 damage**."
#[test]
fn rule_5_two_attacks_kill_a_normal_card() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P0, Rank::FIVE);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "damaged but fully functional");
    go(&mut s, atk(1, 0));
    assert_eq!(occupancy(&s, 0, P1), 0);
    assert_eq!(discard_ranks(&s, P1), vec![Rank::SEVEN]);
}

/// §5: "A **face-down card is a blank 2-HP card** whatever its rank." So a face-down Jack
/// dies to two hits, exactly like a face-down 4 — the third hit point arrives on the flip.
#[test]
fn rule_5_a_face_down_jack_is_a_blank_two_hp_card() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P0, Rank::FIVE);
    p.face_down(0, P1, Rank::JACK);
    let mut s = p.build();

    assert_eq!(card_at(&s, 0, P1, 0).max_hp(), 2);
    go(&mut s, atk(0, 0));
    go(&mut s, atk(1, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "two hits kill a face-down Jack");
    assert_eq!(discard_ranks(&s, P1), vec![Rank::JACK]);
}

/// The corollary that makes the rule matter for belief modeling: because every face-down
/// card has the same hit points, **a Jack cannot be identified by chipping it**. Damage is
/// public (§5), so if a face-down card could survive two hits that fact alone would leak
/// its rank to both players.
#[test]
fn rule_5_you_cannot_identify_a_face_down_jack_by_damaging_it() {
    let mut p = Position::empty();
    p.face_down(0, P1, Rank::JACK);
    p.face_down(0, P1, Rank::FOUR);
    let s = p.build();

    let jack = card_at(&s, 0, P1, 0);
    let four = card_at(&s, 0, P1, 1);
    assert_eq!(jack.max_hp(), four.max_hp(), "indistinguishable");
    assert_eq!(jack.hp_remaining(), four.hp_remaining());
}

/// §5: "Damage persists through flipping" — and flipping a Jack *raises* its ceiling from
/// 2 to 3. So a face-down Jack on 1 damage becomes a face-up Jack on 2 of 3, and needs two
/// more hits rather than one.
#[test]
fn rule_5_flipping_a_damaged_jack_raises_its_ceiling_and_keeps_the_damage() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::JACK);
    p.damage(0, P0, 0, 1);
    let mut s = p.build();

    assert_eq!(card_at(&s, 0, P0, 0).hp_remaining(), 1, "1 of 2 while face-down");

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    let jack = card_at(&s, 0, P0, 0);
    assert_eq!(jack.damage, 1, "the damage persists");
    assert_eq!(jack.max_hp(), 3, "but the ceiling rose");
    assert_eq!(jack.hp_remaining(), 2, "so it now has 2 of 3");
}

/// §4: "**Each card may attack only once per turn.**"
#[test]
fn rule_4_each_card_attacks_only_once_per_turn() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::JACK);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    refuse(&s, atk(0, 0));
    assert_eq!(s.actions_remaining, 2, "and actions remain, so it is not that");
}

/// §4: "Put a card from hand **face-down** ... It is inactive: it can be attacked and
/// killed, but **cannot attack** and has no power."
#[test]
fn rule_4_face_down_cards_cannot_attack() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::TEN);
    p.face_up(0, P1, Rank::FOUR);
    let s = p.build();

    refuse(&s, atk(0, 0));
}

/// §5: "You may attack face-down enemy cards (they are legal targets; you just don't know
/// what they are)."
#[test]
fn rule_5_face_down_enemy_cards_are_legal_targets() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_down(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert!(!card_at(&s, 0, P1, 0).face_up, "attacking does not reveal it");
    assert!(
        !card_at(&s, 0, P1, 0).rank_known_to(P0),
        "and does not tell the attacker what it is"
    );
}

/// §5: "Lanes are otherwise independent" — an attack never crosses lanes.
#[test]
fn rule_1_attacks_do_not_cross_lanes() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(1, P1, Rank::SEVEN);
    let s = p.build();

    assert!(
        !s.legal_actions()
            .iter()
            .any(|a| matches!(a, Action::Attack { .. })),
        "no attack is available across lanes: {:?}",
        legal_names(&s)
    );
}

/// §5: "Damage persists through flipping."
///
/// Uses an 8 rather than a 7 as the flipped card: a 7 would heal itself on the way up, so
/// the test would pass for the wrong reason.
#[test]
fn rule_5_damage_persists_through_flipping() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::EIGHT);
    p.damage(0, P0, 0, 1);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert!(card_at(&s, 0, P0, 0).face_up);
    assert_eq!(damage_at(&s, 0, P0, 0), 1);
}

// ========================================================= 8 · Retaliate ==

/// §6: "Any card that attacks this 8 **takes 1 damage** — except a 9."
#[test]
fn rule_6_eight_retaliates_against_its_attacker() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "the 8 took the hit");
    assert_eq!(damage_at(&s, 0, P0, 0), 1, "and hit back");
}

/// §6 + §8: "Retaliate fires **even when the attack kills the 8**", and it "resolves *after*
/// the attacker's damage is applied".
#[test]
fn rule_8_retaliate_fires_even_when_the_attack_kills_the_eight() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::EIGHT);
    p.damage(0, P1, 0, 1); // one hit from death
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "the 8 died");
    assert_eq!(damage_at(&s, 0, P0, 0), 1, "and still retaliated");
}

/// §6: retaliate is a *constant* power, and "Powers are inert while a card is face-down".
#[test]
fn rule_6_a_face_down_eight_does_not_retaliate() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_down(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P0, 0), 0, "an inert power does not fire");
}

/// §6: "A card that attacks this 8 takes 1 damage — **except a 9**." Nimble.
#[test]
fn rule_6_a_nine_takes_no_retaliate_damage_from_an_eight() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P0, 0), 0, "Nimble");
}

/// Retaliate can kill the attacker. This is the mechanism behind the mutual-lane-win draw
/// of §7.
#[test]
fn rule_6_retaliate_can_kill_the_attacker() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.damage(0, P0, 0, 1);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(occupancy(&s, 0, P0), 0, "the attacker died to retaliate");
    assert_eq!(discard_ranks(&s, P0), vec![Rank::FOUR]);
}

// =============================================================== 9 · Nimble ==

/// §6: the 9 "**Deals 2 damage to Jacks**" — a single attack action for 2 damage.
#[test]
fn rule_6_a_nine_deals_two_damage_to_a_jack() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P1, Rank::JACK);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 2, "one action, two damage");
    assert_eq!(s.actions_remaining, 2, "and it cost one action");
}

/// §5 + §6: a 9 deals its ordinary **1** to a face-down Jack. A face-down card is a blank
/// 2-HP card, so there is no Jack there for the 9 to be good against — and the 9 kills it
/// in two hits like any other card.
///
/// Everything that keys on a target being a Jack reads the *live power*, never the bare
/// rank: the taunt, the twinstrike block, and this doubling all agree.
#[test]
fn rule_6_a_nine_deals_only_one_to_a_face_down_jack() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P0, Rank::NINE);
    p.face_down(0, P1, Rank::JACK);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "no doubling against a blank card");
    assert!(
        !card_at(&s, 0, P1, 0).face_up,
        "and the attack does not reveal it"
    );

    go(&mut s, atk(1, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "two ordinary hits kill it");
}

/// The contrast, so the rule is pinned from both sides: flip that same Jack and the 9's
/// doubling comes back on.
#[test]
fn rule_6_a_nine_deals_two_once_the_jack_is_face_up() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P1, Rank::JACK);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 2);
}

/// The 9's doubling keys on the target being a Jack, not on anything else.
#[test]
fn rule_6_a_nine_deals_normal_damage_to_everything_else() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
}

// ============================================================ J · Taunt ==

/// §6: "**Must be killed before any other card in his lane can be attacked.**"
#[test]
fn rule_6_jack_taunt_forces_attacks_onto_the_jack() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::SEVEN);
    let s = p.build();

    allow(&s, atk(0, 0));
    refuse(&s, atk(0, 1));
}

/// §6: taunt is constant, so a **face-down** Jack does not taunt.
#[test]
fn rule_6_a_face_down_jack_does_not_taunt() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_down(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::SEVEN);
    let s = p.build();

    allow(&s, atk(0, 1));
}

/// §5: "Where Jack taunt constrains the choice to Jacks and there is **more than one Jack
/// in the lane, the attacker picks which Jack to hit** — there is no 'oldest first'
/// ordering."
#[test]
fn rule_5_with_two_jacks_the_attacker_picks_which_to_hit() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::SEVEN);
    let s = p.build();

    allow(&s, atk(0, 0));
    allow(&s, atk(0, 1));
    refuse(&s, atk(0, 2));
}

/// Once the Jack dies, the taunt lifts.
#[test]
fn rule_6_taunt_lifts_when_the_jack_dies() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::NINE); // 2 damage to Jacks
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::JACK);
    p.damage(0, P1, 0, 1); // one 9-hit from death
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    refuse(&s, atk(1, 1));
    go(&mut s, atk(0, 0)); // the 9 kills the Jack
    assert_eq!(occupancy(&s, 0, P1), 1);
    allow(&s, atk(1, 0)); // now the 7 is reachable
}

// ========================================================== 10 · Twinstrike ==

/// §6: "When attacking, deals **1 damage each to two cards** in the opposing lane."
#[test]
fn rule_6_ten_twinstrikes_two_targets_for_one_each() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P1, Rank::FOUR);
    p.face_up(0, P1, Rank::FIVE);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(s.phase(), Phase::SplitTarget);
    assert_eq!(
        damage_at(&s, 0, P1, 0),
        0,
        "no damage lands until both targets are chosen"
    );

    go(&mut s, Action::SplitTarget { slot: 1 });
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P1, 1), 1);
    assert_eq!(s.actions_remaining, 2, "one action for both halves");
}

/// §6: with only one card in the lane there is nothing to split onto, so a lone 10 deals
/// its plain 1.
#[test]
fn rule_6_ten_with_a_single_target_deals_one() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(s.phase(), Phase::Main, "nothing to ask about");
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
}

/// §6: "**If an intended target is a 9 or a Jack, only that card is damaged** — both block
/// the split, but for different reasons." Against a single Jack: 1 to the Jack only.
#[test]
fn rule_6_ten_against_a_single_jack_deals_one_to_the_jack_only() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(s.phase(), Phase::Main, "taunt leaves nowhere for the second half");
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P1, 1), 0);
}

/// §6 + §8: "With **two Jacks**, taunt already confines both halves to Jacks, nothing can
/// leak past, and the 10 deals **1 to each**."
#[test]
fn rule_6_ten_against_two_jacks_deals_one_to_each() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(s.phase(), Phase::SplitTarget);
    assert_eq!(
        s.legal_actions(),
        vec![Action::SplitTarget { slot: 1 }],
        "the second half is confined to the other Jack"
    );
    go(&mut s, Action::SplitTarget { slot: 1 });
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P1, 1), 1);
    assert_eq!(damage_at(&s, 0, P1, 2), 0, "nothing leaked past the taunt");
}

/// §6 + §8: "With **two 9s** it is still **1 to one 9** — Nimble dodges the spread
/// personally, not positionally."
///
/// §8 is emphatic that this must not be unified with the Jack case: "they are different
/// mechanics that happen to share a symptom in the one-card case."
#[test]
fn rule_6_ten_against_two_nines_deals_one_to_one_nine() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P1, Rank::NINE);
    p.face_up(0, P1, Rank::NINE);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(s.phase(), Phase::Main, "Nimble refuses the spread");
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P1, 1), 0);
}

/// The other half of the same ruling: a 9 elsewhere in the lane is not an eligible second
/// target either.
#[test]
fn rule_6_a_nine_is_never_the_second_half_of_a_twinstrike() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P1, Rank::FOUR);
    p.face_up(0, P1, Rank::NINE);
    p.face_up(0, P1, Rank::FIVE);
    let mut s = p.build();

    go(&mut s, atk(0, 0)); // primary is the 4
    assert_eq!(
        s.legal_actions(),
        vec![Action::SplitTarget { slot: 2 }],
        "the 9 is skipped; only the 5 is eligible"
    );
}

/// Twinstrike is constant, so a face-down 10 does not split.
#[test]
fn rule_6_a_face_down_ten_does_not_twinstrike() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::TEN);
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::FOUR);
    p.face_up(0, P1, Rank::FIVE);
    let mut s = p.build();

    // The face-down 10 cannot attack at all; the 4 attacks normally.
    refuse(&s, atk(0, 0));
    go(&mut s, atk(1, 0));
    assert_eq!(s.phase(), Phase::Main);
    assert_eq!(damage_at(&s, 0, P1, 1), 0);
}

/// A twinstrike hitting two 8s takes retaliate from both.
///
/// **[ASSUMED]** §6 says "any card that attacks this 8 takes 1 damage" and the 10 attacked
/// both, so the damage adds — which kills a 2-HP 10. The rules do not address the case
/// directly; flagged in `apply.rs` and here.
#[test]
fn rule_6_assumed_twinstrike_into_two_eights_takes_two_retaliate() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P1, Rank::EIGHT);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    go(&mut s, Action::SplitTarget { slot: 1 });
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P1, 1), 1);
    assert_eq!(occupancy(&s, 0, P0), 0, "1 + 1 retaliate kills the 10");
}

// ==================================================================== pairs ==

/// §5: "A pair is **two face-up cards of matching rank that you control in the same lane**.
/// Both members must be face-up."
#[test]
fn rule_5_a_pair_needs_two_face_up_same_rank_cards_in_one_lane() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::SEVEN);
    p.face_down(0, P0, Rank::SEVEN); // face-down: not eligible
    p.face_up(0, P0, Rank::EIGHT); // different rank
    p.face_up(1, P0, Rank::SEVEN); // different lane
    let s = p.build();

    assert_eq!(
        count_legal(&s, |a| matches!(a, Action::DeclarePair { .. })),
        0,
        "no legal pair here: {:?}",
        legal_names(&s)
    );
}

/// §5: "A pair attacks together as a **single action** dealing **2 damage to one target**.
/// You **may not choose to split** the damage."
#[test]
fn rule_5_a_pair_deals_two_damage_for_one_action() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P1, Rank::JACK); // 3 HP, so it survives and we can read the damage
    let mut s = p.build();

    go(
        &mut s,
        Action::DeclarePair {
            lane: 0,
            slot_a: 0,
            slot_b: 1,
        },
    );
    assert_eq!(s.actions_remaining, 2, "declaring costs an action");

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 2);
    assert_eq!(s.actions_remaining, 1, "the pair attack is one action");
}

/// §5: "**Pairs must attack together** — a paired card cannot attack alone", and "A pair
/// attack is one attack for **both** members' once-per-turn budget".
#[test]
fn rule_5_a_pair_attack_spends_both_members_attacks() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::SEVEN);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::JACK);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    refuse(&s, atk(0, 0));
    refuse(&s, atk(1, 0));
}

/// §5: "You therefore cannot squeeze extra damage out of a lane by attacking with two cards
/// separately and *then* pairing them."
#[test]
fn rule_5_cannot_attack_separately_then_pair_for_extra_damage() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P1, Rank::JACK);
    let mut s = p.build();

    go(&mut s, atk(0, 0)); // one card attacks alone
    go(
        &mut s,
        Action::DeclarePair {
            lane: 0,
            slot_a: 0,
            slot_b: 1,
        },
    );
    refuse(&s, atk(0, 0));
    refuse(&s, atk(1, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "only the single attack landed");
}

/// §5: "A pair is **broken** if a member **dies**", and the survivor attacks alone again.
#[test]
fn rule_5_a_pair_breaks_when_a_member_dies() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::SEVEN);
    p.damage(0, P0, 1, 1);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::FOUR);
    p.to_move(P1);
    let mut s = p.build();

    go(&mut s, atk(0, 1)); // P1 kills the damaged member
    assert_eq!(occupancy(&s, 0, P0), 1);
    assert!(
        card_at(&s, 0, P0, 0).pair_id.is_none(),
        "the survivor is unpaired"
    );

    go(&mut s, Action::Pass);
    allow(&s, atk(0, 0));
}

/// §5: "A pair is **broken** if a Queen moves one member to another lane. Those are the
/// only two ways out: a pair **cannot be dissolved voluntarily**."
#[test]
fn rule_5_a_queen_breaks_a_pair_by_moving_a_member() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::SEVEN);
    p.pair(0, P0, 0, 1);
    p.face_down(1, P0, Rank::QUEEN);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 1, slot: 0 });
    go(&mut s, Action::MoveHere { lane: 0, slot: 0 });

    assert!(card_at(&s, 1, P0, 1).pair_id.is_none(), "the moved member");
    assert!(card_at(&s, 0, P0, 0).pair_id.is_none(), "and the one left behind");
}

/// §5: "a card **cannot leave one pair to join another**."
#[test]
fn rule_5_a_paired_card_cannot_join_another_pair() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::SEVEN);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P0, Rank::SEVEN); // a third 7, unpaired
    let s = p.build();

    refuse(
        &s,
        Action::DeclarePair {
            lane: 0,
            slot_a: 0,
            slot_b: 2,
        },
    );
    refuse(
        &s,
        Action::DeclarePair {
            lane: 0,
            slot_a: 1,
            slot_b: 2,
        },
    );
}

/// §5: "Three same-rank cards do not form a bigger group; the third stays unpaired and
/// attacks alone."
#[test]
fn rule_5_a_third_same_rank_card_stays_unpaired_and_attacks_alone() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::SEVEN);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P1, Rank::JACK);
    let mut s = p.build();

    go(&mut s, atk(2, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "alone, so 1 damage");
}

/// §5: "Two disjoint pairs of the same rank in one lane are legal, which can only arise in
/// the base game — the split deck gives each player exactly two of each rank."
#[test]
fn rule_5_two_disjoint_pairs_of_the_same_rank_are_legal() {
    let mut p = Position::new(duel52_engine::GameConfig::base());
    for _ in 0..4 {
        p.face_up(0, P0, Rank::SEVEN);
    }
    p.pair(0, P0, 0, 1);
    let mut s = p.build();

    go(
        &mut s,
        Action::DeclarePair {
            lane: 0,
            slot_a: 2,
            slot_b: 3,
        },
    );
    assert!(card_at(&s, 0, P0, 2).pair_id.is_some());
    assert_ne!(
        card_at(&s, 0, P0, 0).pair_id,
        card_at(&s, 0, P0, 2).pair_id,
        "two distinct pairs"
    );
}

// ============================================ pairs × rank modifiers (§5) ==

/// §5: "A pair of **9s** deals **4 damage to a Jack** (the 9's doubling applies to the
/// pair's 2), which one-shots a 3-HP Jack."
#[test]
fn rule_5_a_pair_of_nines_deals_four_to_a_jack() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P0, Rank::NINE);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::JACK);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "4 damage one-shots a 3-HP Jack");
    assert_eq!(discard_ranks(&s, P1), vec![Rank::JACK]);
}

/// §5: "Against anything else it is the normal 2."
#[test]
fn rule_5_a_pair_of_nines_deals_two_to_anything_else() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P0, Rank::NINE);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::SEVEN);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(occupancy(&s, 0, P1), 1, "exactly 2 damage: dead, not overkill");
}

/// §5: "A pair of **9s** attacking an **8** takes **no** retaliate damage — each member is
/// individually immune, and pairing does not forfeit Nimble. A 9-pair therefore kills an 8
/// outright for free."
#[test]
fn rule_5_a_pair_of_nines_kills_an_eight_for_free() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P0, Rank::NINE);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "the 8 died to 2 damage");
    assert_eq!(damage_at(&s, 0, P0, 0), 0, "and neither 9 took retaliate");
    assert_eq!(damage_at(&s, 0, P0, 1), 0);
}

/// §5: "If a pair attacks an 8, **both members take 1 retaliate damage**."
#[test]
fn rule_5_a_non_nine_pair_takes_retaliate_on_both_members() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::SEVEN);
    p.face_up(0, P0, Rank::SEVEN);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P0, 0), 1);
    assert_eq!(damage_at(&s, 0, P0, 1), 1);
}

/// §5: "A pair of **10s** twinstrikes: the pair's 2 damage is **split 1 + 1 across two
/// targets**, not doubled."
#[test]
fn rule_5_a_pair_of_tens_splits_one_and_one() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P0, Rank::TEN);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::SEVEN);
    p.face_up(0, P1, Rank::EIGHT); // an 8 so we can also check retaliate below
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    go(&mut s, Action::SplitTarget { slot: 1 });
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P1, 1), 1);
    assert_eq!(
        occupancy(&s, 0, P1),
        2,
        "1 + 1 across two fresh cards kills nothing — the cost of pairing 10s"
    );
}

/// §5: "**Damage is never lost** — whenever the split cannot happen, because it is blocked
/// (a 9, or a lone Jack) or because the lane holds only one legal target, the full **2**
/// lands on that single card."
#[test]
fn rule_5_a_pair_of_tens_consolidates_to_two_when_blocked_by_a_nine() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P0, Rank::TEN);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::NINE);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0)); // primary is the 9: Nimble refuses the spread
    assert_eq!(s.phase(), Phase::Main);
    assert_eq!(occupancy(&s, 0, P1), 1, "the full 2 killed the 9");
    assert_eq!(discard_ranks(&s, P1), vec![Rank::NINE]);
}

/// Same ruling, the lone-Jack case.
#[test]
fn rule_5_a_pair_of_tens_consolidates_to_two_when_blocked_by_a_lone_jack() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P0, Rank::TEN);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(s.phase(), Phase::Main, "taunt leaves nowhere to split");
    assert_eq!(damage_at(&s, 0, P1, 0), 2, "the full 2 lands on the Jack");
    assert_eq!(damage_at(&s, 0, P1, 1), 0);
}

/// Same ruling, the single-target case.
#[test]
fn rule_5_a_pair_of_tens_consolidates_to_two_against_a_lone_defender() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P0, Rank::TEN);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "2 damage, so the 7 dies");
}

/// The intersection of two rulings: a **pair of 10s against two Jacks**. §5 forces the
/// pair's 2 to split 1 + 1, and §8 says taunt has already confined both halves to Jacks
/// with nothing to leak past — so it is 1 to each Jack, and the third card is untouched.
#[test]
fn rule_5_a_pair_of_tens_against_two_jacks_deals_one_to_each() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P0, Rank::TEN);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(
        s.legal_actions(),
        vec![Action::SplitTarget { slot: 1 }],
        "the second half is confined to the other Jack"
    );
    go(&mut s, Action::SplitTarget { slot: 1 });
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P1, 1), 1);
    assert_eq!(damage_at(&s, 0, P1, 2), 0, "nothing leaked past the taunt");
}

/// A lone 10 whose split is blocked deals 1, not 2: its second point of damage *was* the
/// twinstrike bonus, so it goes away with the split. The §5 "damage is never lost" promise
/// is about the pair's 2, not about a single card's 1.
#[test]
fn rule_6_a_lone_ten_blocked_from_splitting_deals_one() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::TEN);
    p.face_up(0, P1, Rank::NINE);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(occupancy(&s, 0, P1), 2);
}

/// A pair may not attack if either member is frozen — freeze blocks attacking (§8), and a
/// pair attack requires both members to be able to attack (§5).
#[test]
fn rule_8_a_pair_cannot_attack_if_either_member_is_frozen() {
    let mut p = Position::empty();
    p.ply(5);
    p.to_move(P1);
    p.face_up(0, P1, Rank::SEVEN);
    p.face_up(0, P1, Rank::SEVEN);
    p.pair(0, P1, 0, 1);
    p.freeze(0, P1, 1, 5);
    p.face_up(0, P0, Rank::FOUR);
    let s = p.build();

    refuse(
        &s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
}

// ================================================= base cards are untouchable ==

/// §3: "While the draw pile is non-empty, base cards **cannot be attacked and cannot be
/// flipped**. They are untouchable."
#[test]
fn rule_3_base_cards_cannot_be_attacked_before_the_unlock() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.base(0, P1, Rank::SEVEN);
    let s = p.build();

    refuse(&s, atk(0, 0));
    assert_eq!(
        count_legal(&s, |a| matches!(a, Action::Attack { .. })),
        0,
        "there is nothing legal to attack"
    );
}

/// §3: "Once the draw pile is empty, base cards become **normal cards** in their lane: they
/// can be attacked, flipped, targeted by a 5, healed by a 7, and moved by a Queen."
#[test]
fn rule_3_base_cards_become_attackable_after_the_unlock() {
    let mut p = Position::empty();
    p.unlock();
    p.face_up(0, P0, Rank::FOUR);
    p.base(0, P1, Rank::SEVEN);
    p.hand(P0, &[Rank::KING]);
    p.hand(P1, &[Rank::KING]);
    let mut s = p.build();

    allow(&s, atk(0, 0));
    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert!(!card_at(&s, 0, P1, 0).face_up, "attacking still does not flip it");
}

/// §3: a base card can be flipped by its owner once the pile is empty — and only then.
#[test]
fn rule_3_you_can_flip_your_own_base_card_only_after_the_unlock() {
    let mut p = Position::empty();
    p.base(0, P0, Rank::SEVEN);
    let s = p.build();
    refuse(&s, Action::Flip { lane: 0, slot: 0 });

    let mut p = Position::empty();
    p.unlock();
    p.base(0, P0, Rank::SEVEN);
    p.hand(P0, &[Rank::KING]);
    p.hand(P1, &[Rank::KING]);
    let s = p.build();
    allow(&s, Action::Flip { lane: 0, slot: 0 });
}
