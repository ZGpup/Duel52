//! The 3 — **Trap**, and its vengeance variants.
//!
//! This is the one canonical power that fires from `damage_card` rather than from a flip,
//! and it is therefore the template for every "on death, do X" mod.

use super::{LethalOutcome, PowerCtx};
use crate::card::KNOWN_TO_BOTH;
use crate::damage::{DamageSource, Hit};
use crate::state::GameState;

/// The Trap, with an optional vengeance rider.
///
/// `game_rules.md` §6: "If killed **while face-down**, it returns to play face-up with full
/// 2 HP instead of dying — immediately, in the same lane. It comes back fully active with no
/// waiting period, and it returns face-up so the Trap **cannot re-trigger**." A base 3
/// killed post-unlock triggers this too and returns as a normal, non-base card (§3).
///
/// The card keeps its freeze, if it had one: the Trap is not a flip and nothing in §8 clears
/// a freeze early. **[ASSUMED]**
///
/// # `vengeance`
///
/// `0` is the rules as written. A non-zero value is the mod from `MODULAR_RULES.md` §3: the
/// 3 also deals that much damage to whatever killed it. Three things about it are worth
/// keeping in view, because they are what keep the cascade finite:
///
/// - It hits [`DamageSource::attackers`], which is empty for anything that is not a declared
///   attack. So vengeance cannot provoke vengeance, and two 3s cannot ping-pong.
/// - It hits the **attacker**, and attackers are always face-up (§4: a face-down card
///   "cannot attack"), and a face-up 3 has no Trap. So a vengeance hit can *kill* another 3
///   but can never *spring* one.
/// - It fires for **both** members of a pair, because a pair attacks as one action and both
///   members are the attacker (§5) — the same reading that makes both members take
///   retaliate.
///
/// The first of those is a property of the ruleset, not of this function. A second power
/// that triggered on death and dealt non-attack damage would break it, and the cross-ruleset
/// invariant suite is what would catch the result.
pub(crate) fn trap(
    state: &mut GameState,
    ctx: PowerCtx,
    source: DamageSource,
    vengeance: u8,
) -> LethalOutcome {
    if state.lanes[ctx.lane].sides[ctx.side][ctx.slot].face_up {
        return LethalOutcome::Die;
    }

    {
        let card = &mut state.lanes[ctx.lane].sides[ctx.side][ctx.slot];
        card.damage = 0;
        card.face_up = true;
        card.known_to = KNOWN_TO_BOTH;
        card.is_base = false;
        card.attacks_used = 0;
        card.attack_allowance = 1;
    }

    if vengeance > 0 {
        for attacker in source.attackers() {
            state.enqueue_damage(Hit {
                target: attacker,
                amount: vengeance,
                source: DamageSource::Vengeance { from: ctx.id },
            });
        }
    }

    LethalOutcome::Restored
}
