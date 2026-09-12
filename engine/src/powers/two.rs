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

/// **View (your choice)** — as [`view`], but the player decides where the card goes rather
/// than the ruleset deciding for them.
///
/// The first ruleset to claim `MODULAR_RULES.md` §7's `CHOOSE_OPTION` block. It is the
/// interesting shape for that block because the choice is **modal and nameless**: "bottom or
/// discard" is not a card, not a lane and not a rank, so before the reserve there was no
/// logit that could carry it and `game_rules.md` §10a's two readings had to be a config
/// toggle — one ruleset or the other, never a decision inside a game.
///
/// The `CHOOSE_OPTION` node is opened by `do_give_back` once the rank is out of hand, not
/// here, so the two sub-decisions are ordered rank-then-destination and the card can never be
/// left in limbo by a fizzle. Two of the block's four logits are legal; the other two are
/// masked, which is the ordinary state of most of the policy head.
pub(crate) fn view_choose(state: &mut GameState, ctx: PowerCtx) {
    view(state, ctx);
}
