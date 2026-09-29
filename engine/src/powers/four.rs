//! The 4 — **Foresight**, and the **Bomb** variant.

use super::{LethalOutcome, PowerCtx};
use crate::damage::{DamageSource, Hit};
use crate::state::{GameState, Pending};

/// "Look at any one face-down card on the board — including base cards, yours or your
/// opponent's. Private information." (`game_rules.md` §6)
///
/// Fizzles when the board holds no face-down card at all (§8).
pub(crate) fn foresight(state: &mut GameState, ctx: PowerCtx) {
    if !state.face_down_cards().is_empty() {
        state.pending.push(Pending::Foresight { player: ctx.owner });
    }
}

/// **Bomb** — "if killed while face-down, the card that killed it dies too".
///
/// Not a rule of the canonical ruleset: it is the 3's Trap turned outward. The 3 survives
/// its death and the attacker walks away; the Bomb dies and takes the attacker with it. It
/// **replaces** Foresight rather than adding to it **[ASSUMED]** — the owner asked for "a
/// version of the 4 that is a bomb", like the 3 — so flipping a Bomb does nothing, and a
/// face-up 4 is a plain 2-HP card.
///
/// The rulings, each the same call the vengeance variants in [`super::three`] make:
///
/// - **"The card that killed it" is [`DamageSource::attackers`]**, so a pair loses **both**
///   members, as it pays retaliate and vengeance with both (§5). **[ASSUMED]**
/// - **It kills whatever the card's hit points.** A face-up Jack dies, and so does a 9 —
///   Nimble dodges an 8's retaliate and a 10's spread, and this is neither. **[ASSUMED]**
///   It is delivered as a hit for exactly the attacker's remaining HP, through the damage
///   queue, rather than by removing the card here: so it lands in FIFO order behind the
///   rest of the attack, and a card in play still never carries more damage than it has HP
///   (`encode.rs`'s damage one-hot depends on that).
/// - **Only an attack has a killer.** A Bomb caught in a 2's Blast takes nobody with it: the
///   2 that blasted it has already left play, and `attackers()` is empty for a blast. A
///   face-down card cannot be hit by retaliate or vengeance at all, since both target an
///   attacker and attackers are face-up (§4).
/// - A **base** 4 killed after the unlock detonates too, as a base 3 springs (§3).
///
/// The hit it deals is [`DamageSource::Vengeance`], which cannot provoke a second vengeance,
/// and it lands on a face-up card, which has no death trigger to set off. So a Bomb never
/// extends a cascade.
pub(crate) fn bomb(state: &mut GameState, ctx: PowerCtx, source: DamageSource) -> LethalOutcome {
    if state.lanes[ctx.lane].sides[ctx.side][ctx.slot].face_up {
        return LethalOutcome::Die;
    }
    let config = state.config;
    for attacker in source.attackers() {
        let Some(card) = state.card(attacker) else {
            continue;
        };
        let amount = card.hp_remaining(&config);
        state.enqueue_damage(Hit {
            target: attacker,
            amount,
            source: DamageSource::Vengeance { from: ctx.id },
        });
    }
    LethalOutcome::Die
}
