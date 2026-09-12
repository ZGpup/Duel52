//! The modded power variants, one named test per ruling.
//!
//! `CLAUDE.md`: "Every ruling in `game_rules.md` gets a named test." A mod is not in
//! `game_rules.md`, so these are named `mod_<ruleset>_<what it asserts>` rather than
//! `rule_N_…`, and each one states the canonical behaviour it departs from.
//!
//! The canonical tests in `rules_combat.rs` and `rules_powers.rs` stay exactly as they were
//! and now pin the *canonical ruleset* rather than "the rules" — `Position::empty()` builds
//! `GameConfig::default()`, which is canonical, so that happened without touching them.

mod common;
use common::*;

use duel52_engine::powers::PowerId;
use duel52_engine::testkit::*;
use duel52_engine::{Action, GameConfig, Player::P0, Player::P1, Rank};

/// Attack from P0's slot `a` to P1's slot `t` in lane 0.
fn atk(a: u8, t: u8) -> Action {
    Action::Attack {
        lane: 0,
        attacker: a,
        target: t,
    }
}

/// A canonical config with one card's power swapped.
fn with_power(power: PowerId) -> GameConfig {
    let mut cfg = GameConfig::default();
    cfg.powers[power.rank().index()] = power;
    cfg
}

// ==================================================== the 3, with vengeance ==

/// The mod `MODULAR_RULES.md` §3 is built around: the Trap still springs, and the attacker
/// that sprang it takes a hit.
#[test]
fn mod_three_vengeance_springs_the_trap_and_damages_the_killer() {
    let mut p = Position::new(with_power(PowerId::ThreeTrapVengeance1));
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P0, Rank::FIVE);
    p.face_down(0, P1, Rank::THREE);
    let mut s = p.build();

    go(&mut s, atk(0, 0)); // 1 damage: not lethal, so no Trap and no vengeance
    assert_eq!(damage_at(&s, 0, P0, 0), 0, "vengeance is a death trigger");
    assert_eq!(damage_at(&s, 0, P1, 0), 1);

    go(&mut s, atk(1, 0)); // lethal: the Trap springs
    let three = card_at(&s, 0, P1, 0);
    assert!(three.face_up, "the Trap returned it face-up");
    assert_eq!(three.damage, 0, "at full HP");
    assert_eq!(
        damage_at(&s, 0, P0, 1),
        1,
        "and the 5 that killed it took the vengeance"
    );
    assert_eq!(
        damage_at(&s, 0, P0, 0),
        0,
        "the 4 attacked earlier and is not the killer"
    );
}

/// Two damage kills a 2-HP attacker outright, so the sharper variant is a genuine trade
/// rather than a tax. This is what makes `trap_vengeance_two_damage` a different *shape*
/// worth its own name rather than a knob (`MODULAR_RULES.md` §5a).
#[test]
fn mod_three_vengeance_two_kills_the_attacker_outright() {
    let mut p = Position::new(with_power(PowerId::ThreeTrapVengeance2));
    p.face_up(0, P0, Rank::FOUR);
    p.face_down(0, P1, Rank::THREE);
    p.damage(0, P1, 0, 1); // one hit from dead
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert!(card_at(&s, 0, P1, 0).face_up, "the Trap sprang");
    assert_eq!(occupancy(&s, 0, P0), 0, "the 4 died to the vengeance");
    assert_eq!(discard_ranks(&s, P0), vec![Rank::FOUR]);
}

/// A pair attacks as one action and both members are the attacker (`game_rules.md` §5), the
/// same reading that makes both members take retaliate. So vengeance hits both.
#[test]
fn mod_three_vengeance_hits_both_members_of_a_pair() {
    let mut p = Position::new(with_power(PowerId::ThreeTrapVengeance1));
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P0, Rank::FOUR);
    p.pair(0, P0, 0, 1);
    p.face_down(0, P1, Rank::THREE);
    let mut s = p.build();

    go(&mut s, atk(0, 0)); // a pair deals 2 — lethal to a face-down card
    assert!(card_at(&s, 0, P1, 0).face_up, "the Trap sprang");
    assert_eq!(damage_at(&s, 0, P0, 0), 1, "first member took vengeance");
    assert_eq!(damage_at(&s, 0, P0, 1), 1, "and so did the second");
}

/// A **face-up** 3 has no Trap, so it has no vengeance either — the rider is attached to the
/// Trap, not to the rank. This is one of the two accidents that keeps the cascade finite:
/// vengeance hits attackers, attackers are always face-up, and a face-up 3 cannot spring
/// (`MODULAR_RULES.md` §3a).
#[test]
fn mod_three_vengeance_does_not_fire_for_a_face_up_three() {
    let mut p = Position::new(with_power(PowerId::ThreeTrapVengeance1));
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P0, Rank::FIVE);
    p.face_up(0, P1, Rank::THREE);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    go(&mut s, atk(1, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "a face-up 3 just dies");
    assert_eq!(damage_at(&s, 0, P0, 0), 0, "no vengeance");
    assert_eq!(damage_at(&s, 0, P0, 1), 0);
}

/// The ordering bug `MODULAR_RULES.md` §3a names: a 10 twinstrikes two face-down 3s, both
/// spring, and **both** vengeance hits must land. Under a nested-loop implementation the
/// second is silently dropped because the loop that spawned it has already moved on; the
/// damage queue is what makes this come out right.
///
/// Two hits of 1 kill the 10, which is the observable consequence.
#[test]
fn mod_three_vengeance_from_two_traps_in_one_twinstrike_both_land() {
    let mut p = Position::new(with_power(PowerId::ThreeTrapVengeance1));
    p.face_up(0, P0, Rank::TEN);
    p.face_down(0, P1, Rank::THREE);
    p.face_down(0, P1, Rank::THREE);
    p.damage(0, P1, 0, 1); // both are one hit from dead, so both halves are lethal
    p.damage(0, P1, 1, 1);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    go(&mut s, Action::SplitTarget { slot: 1 });

    assert!(card_at(&s, 0, P1, 0).face_up, "first Trap sprang");
    assert!(card_at(&s, 0, P1, 1).face_up, "second Trap sprang");
    assert_eq!(
        occupancy(&s, 0, P0),
        0,
        "two vengeance hits killed the 10 — if only one landed it would still be alive"
    );
}

/// The ablation is a plain card: no Trap, no vengeance, dies when killed.
#[test]
fn mod_three_none_dies_like_any_other_card() {
    let mut p = Position::new(with_power(PowerId::ThreeNone));
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P0, Rank::FIVE);
    p.face_down(0, P1, Rank::THREE);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    go(&mut s, atk(1, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "no Trap to spring");
    assert_eq!(discard_ranks(&s, P1), vec![Rank::THREE]);
}

// ============================================ the 8, retaliating only on survival ==

/// The inverse of `rule_8_retaliate_fires_even_when_the_attack_kills_the_eight`. Under this
/// ruleset an 8 that dies to the attack owes nothing.
#[test]
fn mod_eight_on_survival_does_not_retaliate_when_the_attack_kills_it() {
    let mut p = Position::new(with_power(PowerId::EightRetaliateOnSurvival));
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::EIGHT);
    p.damage(0, P1, 0, 1); // one hit from dead
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "the 8 died");
    assert_eq!(
        damage_at(&s, 0, P0, 0),
        0,
        "and took nothing with it — the canonical 8 would have dealt 1 here"
    );
}

/// The other half: an 8 that lives still hits back exactly as the canonical one does.
#[test]
fn mod_eight_on_survival_retaliates_when_it_survives() {
    let mut p = Position::new(with_power(PowerId::EightRetaliateOnSurvival));
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "the 8 took the attack");
    assert_eq!(damage_at(&s, 0, P0, 0), 1, "and hit back");
}

/// Nimble is unchanged by the mode: a 9 takes no retaliate damage from any 8.
#[test]
fn mod_eight_on_survival_still_does_nothing_to_a_nine() {
    let mut p = Position::new(with_power(PowerId::EightRetaliateOnSurvival));
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P0, 0), 0, "Nimble");
}

/// A pair's 2 damage kills a fresh 8 outright, so under this mode **neither member pays** —
/// where canonically both do (`rule_5_a_non_nine_pair_takes_retaliate_on_both_members`).
///
/// This is the sharpest practical consequence of the mod: pairing into an 8 goes from the
/// most expensive way to kill it to the cheapest.
#[test]
fn mod_eight_on_survival_charges_neither_member_of_a_pair_that_kills_it() {
    let mut p = Position::new(with_power(PowerId::EightRetaliateOnSurvival));
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P0, Rank::FOUR);
    p.pair(0, P0, 0, 1);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "a pair's 2 damage killed the 8");
    assert_eq!(damage_at(&s, 0, P0, 0), 0, "so neither member paid");
    assert_eq!(damage_at(&s, 0, P0, 1), 0);
}

/// The ablation: no retaliation at all, however healthy the 8 is.
#[test]
fn mod_eight_none_never_hits_back() {
    let mut p = Position::new(with_power(PowerId::EightNone));
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::EIGHT);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(damage_at(&s, 0, P1, 0), 1);
    assert_eq!(damage_at(&s, 0, P0, 0), 0);
}

// ======================================================= the Jack at 2 hit points ==

/// Tier 1: one config number. The Jack still taunts, and now dies to two hits.
#[test]
fn mod_jack_2hp_still_taunts_but_dies_to_two_hits() {
    let mut cfg = GameConfig::default();
    cfg.jack_hp = 2;

    let mut p = Position::new(cfg);
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P0, Rank::FIVE);
    p.face_up(0, P1, Rank::JACK);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    // Taunt is untouched: the 7 is not a legal target while the Jack stands.
    assert_eq!(
        s.legal_attack_targets(0, P1),
        vec![0],
        "the Jack still taunts"
    );
    assert_eq!(card_at(&s, 0, P1, 0).max_hp(&s.config), 2);

    go(&mut s, atk(0, 0));
    go(&mut s, atk(1, 0));
    assert_eq!(discard_ranks(&s, P1), vec![Rank::JACK], "two hits, not three");
}

/// The sharp edge this opens: a **lone** 9 now one-shots a Jack, where canonically it takes
/// a pair. `nimble_vs_taunt_multiplier` is the Tier-1 knob if that is too much.
#[test]
fn mod_jack_2hp_is_one_shot_by_a_lone_nine() {
    let mut cfg = GameConfig::default();
    cfg.jack_hp = 2;

    let mut p = Position::new(cfg);
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P1, Rank::JACK);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(
        occupancy(&s, 0, P1),
        0,
        "2 damage against 2 HP — canonically this leaves a Jack on 2 of 3"
    );
}

/// Belief is unaffected, which is the property that would have made this mod unacceptable.
/// `game_rules.md` §5 makes every **face-down** card a blank card whatever its rank, and
/// that is `default_hp`, not `jack_hp` — so a Jack still cannot be identified by chipping it.
#[test]
fn mod_jack_2hp_does_not_leak_a_face_down_jack() {
    let mut cfg = GameConfig::default();
    cfg.jack_hp = 2;

    let mut p = Position::new(cfg);
    p.face_down(0, P1, Rank::JACK);
    p.face_down(0, P1, Rank::FOUR);
    let s = p.build();

    assert_eq!(
        card_at(&s, 0, P1, 0).max_hp(&s.config),
        card_at(&s, 0, P1, 1).max_hp(&s.config),
        "indistinguishable while face-down, exactly as under the canonical rules"
    );
}
