//! The Ace — **Action**.

use super::PowerCtx;
use crate::state::GameState;

/// "Gain 1 action this turn, usable however you like. On the turn it is flipped, the Ace
/// itself may attack twice." (`game_rules.md` §6)
///
/// A King reactivating an Ace grants the action again — once — and *resets* the attack
/// counter rather than stacking it, so an Ace that attacked once and was then Kinged tops
/// out at three attacks that turn, not four (§6).
pub(crate) fn action(state: &mut GameState, ctx: PowerCtx) {
    state.actions_remaining += state.config.ace_bonus_actions;
    let allowance = state.config.ace_attack_allowance;
    let card = &mut state.lanes[ctx.lane].sides[ctx.side][ctx.slot];
    card.attacks_used = 0;
    card.attack_allowance = allowance;
}
