//! The 2 — **View**.

use super::PowerCtx;
use crate::state::{GameState, Pending};

/// "Draw a card, then put a card from your hand on the bottom of your draw pile — a scry,
/// not a discard." (`game_rules.md` §6, house rule §10a)
///
/// Gated on the pile **you** draw from, not the global `base_unlocked` flag (§3, §9): "if
/// that pile is empty the power does nothing at all — no draw, and no bottoming either, so
/// it cannot be used to refill an empty pile." So a 2 can go dead a turn before the global
/// unlock.
pub(crate) fn view(state: &mut GameState, ctx: PowerCtx) {
    if state.pile(ctx.owner).is_empty() {
        return;
    }
    state.draw_one(ctx.owner);
    // The draw guarantees a non-empty hand, so this node always has an answer.
    state.pending.push(Pending::GiveBack { player: ctx.owner });
}
