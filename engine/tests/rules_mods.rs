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
use duel52_engine::{Action, GameConfig, Phase, Player::P0, Player::P1, Rank};

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

// ============================================================== the 4, a Bomb ==

/// The ruleset's premise: kill a face-down 4 and the card that did it dies too. Canonically
/// the 4 is a blank card while face-down and the attacker walks away.
#[test]
fn mod_four_bomb_kills_the_attacker_that_killed_it() {
    let mut p = Position::new(with_power(PowerId::FourBomb));
    p.face_up(0, P0, Rank::FIVE);
    p.face_up(0, P0, Rank::SIX);
    p.face_down(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, atk(0, 0)); // 1 damage: not lethal, so nothing goes off
    assert_eq!(damage_at(&s, 0, P0, 0), 0, "the Bomb is a death trigger");
    assert_eq!(damage_at(&s, 0, P1, 0), 1);

    go(&mut s, atk(1, 0)); // lethal
    assert_eq!(discard_ranks(&s, P1), vec![Rank::FOUR], "the Bomb died");
    assert_eq!(
        discard_ranks(&s, P0),
        vec![Rank::SIX],
        "and took the 6 that killed it"
    );
    assert_eq!(card_at(&s, 0, P0, 0).rank, Rank::FIVE, "the 5 hit it first and lives");
    assert_eq!(damage_at(&s, 0, P0, 0), 0);
}

/// "Kills", not "damages": a face-up Jack's third hit point does not save it, and a 9's
/// Nimble — which dodges an 8's retaliate — does not either. **[ASSUMED]**
#[test]
fn mod_four_bomb_kills_through_hit_points_and_nimble() {
    let mut p = Position::new(with_power(PowerId::FourBomb));
    p.face_up(0, P0, Rank::JACK);
    p.face_down(0, P1, Rank::FOUR);
    p.damage(0, P1, 0, 1);
    p.face_up(1, P0, Rank::NINE);
    p.face_down(1, P1, Rank::FOUR);
    p.damage(1, P1, 0, 1);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(occupancy(&s, 0, P0), 0, "a 3-HP Jack dies to the Bomb");
    go(
        &mut s,
        Action::Attack {
            lane: 1,
            attacker: 0,
            target: 0,
        },
    );
    assert_eq!(occupancy(&s, 1, P0), 0, "and so does a 9");
    assert_eq!(discard_ranks(&s, P0), vec![Rank::NINE, Rank::JACK]);
}

/// A pair attacks as one action and both members are the attacker (`game_rules.md` §5), the
/// reading that makes both pay retaliate and vengeance. So the Bomb takes both.
/// **[ASSUMED]**
#[test]
fn mod_four_bomb_takes_both_members_of_a_pair() {
    let mut p = Position::new(with_power(PowerId::FourBomb));
    p.face_up(0, P0, Rank::FIVE);
    p.face_up(0, P0, Rank::FIVE);
    p.pair(0, P0, 0, 1);
    p.face_down(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, atk(0, 0)); // a pair deals 2 — lethal to a face-down card
    assert_eq!(discard_ranks(&s, P1), vec![Rank::FOUR]);
    assert_eq!(occupancy(&s, 0, P0), 0, "both members died");
    assert_eq!(discard_ranks(&s, P0), vec![Rank::FIVE, Rank::FIVE]);
}

/// Like the 3's Trap, the Bomb is armed only while hidden. A face-up 4 is a plain card.
#[test]
fn mod_four_bomb_does_not_fire_for_a_face_up_four() {
    let mut p = Position::new(with_power(PowerId::FourBomb));
    p.face_up(0, P0, Rank::FIVE);
    p.face_up(0, P0, Rank::SIX);
    p.face_up(0, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    go(&mut s, atk(1, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "a face-up 4 just dies");
    assert_eq!(occupancy(&s, 0, P0), 2, "and takes nobody with it");
}

/// The Bomb **replaces** Foresight rather than adding to it. **[ASSUMED]** Canonically this
/// flip opens a Foresight node (`rule_6_four_reveals_a_card_privately_to_the_peeker_only`).
#[test]
fn mod_four_bomb_replaces_foresight() {
    let mut p = Position::new(with_power(PowerId::FourBomb));
    p.face_down(0, P0, Rank::FOUR);
    p.face_down(0, P1, Rank::KING);
    let mut s = p.build();

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::Main, "no peek to make");
    assert!(!card_at(&s, 0, P1, 0).rank_known_to(P0));
}

// ============================================================= the 2, a Blast ==

/// Both of `two-blast-four-bomb.toml`'s powers at once.
fn death_traps() -> GameConfig {
    let mut cfg = with_power(PowerId::TwoBlast1);
    cfg.powers[Rank::FOUR.index()] = PowerId::FourBomb;
    cfg
}

/// Kill a face-down 2 and every enemy card in its lane takes 1 — face-up or face-down, and
/// with no exception for a Jack's taunt or a 9's Nimble. **[ASSUMED]** Nothing in another
/// lane, and nothing on the 2's own side.
#[test]
fn mod_two_blast_damages_every_enemy_card_in_its_lane() {
    let mut p = Position::new(with_power(PowerId::TwoBlast1));
    p.face_up(0, P0, Rank::FIVE);
    p.face_up(0, P0, Rank::NINE);
    p.face_up(0, P0, Rank::JACK);
    p.face_down(0, P0, Rank::SIX);
    p.face_up(1, P0, Rank::SEVEN);
    p.face_down(0, P1, Rank::TWO);
    p.damage(0, P1, 0, 1);
    p.face_up(0, P1, Rank::KING);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(discard_ranks(&s, P1), vec![Rank::TWO]);
    for slot in 0..4 {
        assert_eq!(
            damage_at(&s, 0, P0, slot),
            1,
            "P0's lane-0 card in slot {slot} took the blast"
        );
    }
    assert_eq!(damage_at(&s, 1, P0, 0), 0, "another lane is untouched");
    assert_eq!(damage_at(&s, 0, P1, 0), 0, "and so is the 2's own side");
}

/// Armed only while hidden, like the Trap and the Bomb.
#[test]
fn mod_two_blast_does_not_fire_for_a_face_up_two() {
    let mut p = Position::new(with_power(PowerId::TwoBlast1));
    p.face_up(0, P0, Rank::FIVE);
    p.face_up(0, P0, Rank::SIX);
    p.face_up(0, P1, Rank::TWO);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    go(&mut s, atk(1, 0));
    assert_eq!(occupancy(&s, 0, P1), 0, "a face-up 2 just dies");
    assert_eq!(damage_at(&s, 0, P0, 0), 0, "no blast");
    assert_eq!(damage_at(&s, 0, P0, 1), 0);
}

/// The Blast **replaces** View rather than adding to it. **[ASSUMED]** Canonically this flip
/// draws from the pile and opens a GiveBack node.
#[test]
fn mod_two_blast_replaces_view() {
    let mut p = Position::new(with_power(PowerId::TwoBlast1));
    p.face_down(0, P0, Rank::TWO);
    let mut s = p.build();
    let (hand, pile) = (s.hands[P0.idx()].len(), s.pile(P0).len());

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    assert_eq!(s.phase(), Phase::Main, "no card to give back");
    assert_eq!(s.hands[P0.idx()].len(), hand, "nothing drawn");
    assert_eq!(s.pile(P0).len(), pile);
}

/// `game_rules.md` §3 makes base cards untouchable while a pile remains, so the Blast skips
/// them until the unlock — the exception the 7's Heal All makes. **[ASSUMED]**
#[test]
fn mod_two_blast_spares_base_cards_until_the_unlock() {
    for unlocked in [false, true] {
        let mut p = Position::new(with_power(PowerId::TwoBlast1));
        p.face_up(0, P0, Rank::FIVE);
        p.base(0, P0, Rank::SIX);
        p.face_down(0, P1, Rank::TWO);
        p.damage(0, P1, 0, 1);
        if unlocked {
            p.unlock();
        }
        let mut s = p.build();

        go(&mut s, atk(0, 0));
        assert_eq!(damage_at(&s, 0, P0, 0), 1, "the attacker takes the blast");
        assert_eq!(
            damage_at(&s, 0, P0, 1),
            u8::from(unlocked),
            "the base card takes it only once unlocked (unlocked = {unlocked})"
        );
    }
}

/// The first case in the engine of an attack setting off a trap on the **attacker's own**
/// side: the blast comes back across the lane and springs P0's face-down 3, on P0's turn.
/// Canonically the only damage to the acting side lands on face-up attackers, so this could
/// not happen — and `probe.rs` classified face-up transitions on that assumption.
#[test]
fn mod_two_blast_can_spring_the_attackers_own_trap() {
    let mut p = Position::new(with_power(PowerId::TwoBlast1));
    p.face_up(0, P0, Rank::FIVE);
    p.face_down(0, P0, Rank::THREE);
    p.damage(0, P0, 1, 1);
    p.face_down(0, P1, Rank::TWO);
    p.damage(0, P1, 0, 1);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    let three = card_at(&s, 0, P0, 1);
    assert!(three.face_up, "the blast killed P0's own 3 and its Trap sprang");
    assert_eq!(three.damage, 0, "at full HP");
    assert_eq!(damage_at(&s, 0, P0, 0), 1);
}

/// A blast can kill a face-down enemy 2, whose own blast comes straight back. What ends the
/// chain is that each 2 fires once and leaves play (`powers/two.rs`).
#[test]
fn mod_two_blast_chains_back_across_the_lane() {
    let mut p = Position::new(with_power(PowerId::TwoBlast1));
    p.face_up(0, P0, Rank::FIVE);
    p.face_down(0, P0, Rank::TWO);
    p.damage(0, P0, 1, 1);
    p.face_down(0, P1, Rank::TWO);
    p.damage(0, P1, 0, 1);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(discard_ranks(&s, P1), vec![Rank::TWO], "P0 killed P1's 2");
    assert_eq!(discard_ranks(&s, P0), vec![Rank::TWO], "whose blast killed P0's 2");
    assert_eq!(damage_at(&s, 0, P0, 0), 1, "the attacker took the first blast");
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "and P1's 7 took the second");
}

/// A 10 that twinstrikes a face-down 2 and a face-down 3, each one hit from dead: the 3
/// springs, the 2 dies, and the 2's blast hits the 10. Taken in both split orders, because
/// the two halves land one after the other through the damage queue.
#[test]
fn mod_two_blast_hits_a_ten_that_twinstrikes_it_alongside_a_trap() {
    for two_first in [true, false] {
        let mut p = Position::new(death_traps());
        p.face_up(0, P0, Rank::TEN);
        let (first, second) = if two_first {
            (Rank::TWO, Rank::THREE)
        } else {
            (Rank::THREE, Rank::TWO)
        };
        p.face_down(0, P1, first);
        p.face_down(0, P1, second);
        p.damage(0, P1, 0, 1);
        p.damage(0, P1, 1, 1);
        let mut s = p.build();

        go(&mut s, atk(0, 0));
        go(&mut s, Action::SplitTarget { slot: 1 });

        assert_eq!(discard_ranks(&s, P1), vec![Rank::TWO], "the 2 died (two_first = {two_first})");
        assert_eq!(ranks_in(&s, 0, P1), vec![Rank::THREE]);
        assert!(card_at(&s, 0, P1, 0).face_up, "the 3's Trap sprang");
        assert_eq!(damage_at(&s, 0, P0, 0), 1, "the 2's blast hit the 10 (two_first = {two_first})");
    }
}

/// A Bomb caught in a Blast has no killer to take with it: the 2 has already left play, and
/// a blast is not an attack. The bystander on the 2's side is untouched.
#[test]
fn mod_two_blast_sets_off_a_bomb_that_takes_nobody_with_it() {
    let mut p = Position::new(death_traps());
    p.face_up(0, P0, Rank::FIVE);
    p.face_down(0, P0, Rank::FOUR);
    p.damage(0, P0, 1, 1);
    p.face_down(0, P1, Rank::TWO);
    p.damage(0, P1, 0, 1);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, atk(0, 0));
    assert_eq!(discard_ranks(&s, P0), vec![Rank::FOUR], "the blast killed the Bomb");
    assert_eq!(discard_ranks(&s, P1), vec![Rank::TWO]);
    assert_eq!(card_at(&s, 0, P1, 0).rank, Rank::SEVEN);
    assert_eq!(damage_at(&s, 0, P1, 0), 0, "and the Bomb hit nothing");
    assert_eq!(damage_at(&s, 0, P0, 0), 1, "the attacker survives the blast alone");
}
