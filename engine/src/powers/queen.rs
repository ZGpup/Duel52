//! The Queen — **Move**.

use super::PowerCtx;
use crate::state::{GameState, Pending};

/// "Move one allied card from another lane into the Queen's lane, face-down or face-up."
/// (`game_rules.md` §6)
///
/// Fizzles with no allied card elsewhere — and §8 notes that is often exactly why you flip
/// her: "a Queen with no move available is still a body that can attack".
///
/// ⚠️ The Queen's destination is **fixed** (her own lane) and only the *source* is chosen,
/// which is why this needs no `CHOOSE_LANE` action block: the source is addressed as a card
/// whose address happens to include a lane, and "from another lane" is a mask.
/// `MODULAR_RULES.md` §1c — a Queen that moved a card *out* to a lane of your choice would
/// be a different, and much more expensive, power.
pub(crate) fn mv(state: &mut GameState, ctx: PowerCtx) {
    if !state.queen_move_sources(ctx.owner, ctx.lane).is_empty() {
        state.pending.push(Pending::QueenSource {
            player: ctx.owner,
            lane: ctx.lane as u8,
        });
    }
}
