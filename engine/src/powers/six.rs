//! The 6 — **Freeze**.

use super::PowerCtx;
use crate::state::GameState;

/// "All enemy cards in the lane are frozen: they may not attack, and cannot be flipped at
/// all. Cannot freeze a 9, ever." (`game_rules.md` §6)
///
/// Per card, not per lane: "Cards that enter the lane *after* the 6 resolves are not
/// frozen", and a frozen card a Queen relocates stays frozen (§8).
///
/// The 9's immunity is Nimble, and powers are inert while face-down (§6), so a **face-down**
/// 9 can be frozen. **[ASSUMED]** — the rulebook's "ever" is about timing (a 9 already in
/// the lane is still immune), not about face-up-ness.
pub(crate) fn freeze(state: &mut GameState, ctx: PowerCtx) {
    let ply = state.ply;
    let turns = state.config.six_freeze_turns;
    let config = state.config;
    let enemy = ctx.owner.other();
    for card in state.lanes[ctx.lane].side_mut(enemy) {
        if card.live_power(&config).is_some_and(|p| p.is_nimble()) {
            continue;
        }
        // §8: unfrozen "at the end of the frozen player's next turn — so exactly one of
        // their turns is lost". Plies strictly alternate, so the victim's next turn is
        // `ply + 1`.
        card.frozen_until_ply = Some(ply + turns);
    }
}
