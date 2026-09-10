//! The 7 — **Heal All**.

use super::PowerCtx;
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
