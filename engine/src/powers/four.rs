//! The 4 — **Foresight**.

use super::PowerCtx;
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
