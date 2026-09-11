//! The 7 — **Heal All**, and the **Shield All** variant.

use super::PowerCtx;
use crate::card::STATUS_SHIELDED;
use crate::state::GameState;

/// "Heal all your damaged cards 2 HP, in all lanes, face-up and face-down." (`game_rules.md`
/// §6) Includes base cards once the pile is empty.
///
/// "Healing is capped at the card's maximum HP — a Jack on 1 HP heals to 3, not 5", which is
/// what `saturating_sub` on the damage counter expresses.
pub(crate) fn heal_all(state: &mut GameState, ctx: PowerCtx) {
    let unlocked = state.base_unlocked;
    let amount = state.config.seven_heal_amount;
    for lane_ref in state.lanes.iter_mut() {
        for card in lane_ref.side_mut(ctx.owner) {
            if card.is_base && !unlocked {
                continue;
            }
            card.damage = card.damage.saturating_sub(amount);
        }
    }
}

/// **Shield All** — "shield all your cards; each ignores the next damage it would take".
///
/// The first ruleset to claim one of `MODULAR_RULES.md` §7's reserve status flags, and the
/// reason the reserve is not a theoretical capacity. It is a *prevention* effect rather than
/// a *restoration* one, which is the thing the base layout could not express: healing is
/// visible in `damage`, a shield is not visible anywhere until it is spent, so before the
/// reserve there was nowhere to put it and the mechanic was Tier 3.
///
/// Deliberately the same shape as [`heal_all`] — every lane, the owner's cards only, base
/// cards excluded until the pile empties — so a ruleset that swaps one for the other is
/// measuring prevention against restoration and not two unrelated changes at once.
///
/// A shield does not stack: the flag is a bit, so shielding an already-shielded card is a
/// no-op rather than a second charge. `apply.rs`'s `apply_one_hit` spends it.
pub(crate) fn shield_all(state: &mut GameState, ctx: PowerCtx) {
    let unlocked = state.base_unlocked;
    for lane_ref in state.lanes.iter_mut() {
        for card in lane_ref.side_mut(ctx.owner) {
            if card.is_base && !unlocked {
                continue;
            }
            card.set_status(STATUS_SHIELDED);
        }
    }
}
