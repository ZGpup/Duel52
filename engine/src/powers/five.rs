//! The 5 — **Flip**.

use super::PowerCtx;
use crate::state::{GameState, Pending, ResolveKind};

/// "Flip all your face-down cards in its lane. You choose the order in which their powers
/// resolve — one at a time, seeing each result before choosing the next." (`game_rules.md`
/// §6) Includes your base card in that lane once the pile is empty.
///
/// All-or-nothing: §8 stresses that post-unlock it flips your base card "whether you want it
/// flipped or not", which is what makes a held 5 committal in the endgame. The sole
/// exception is frozen cards, which it "simply skips".
///
/// The list is snapshotted here, so a face-down card that a Queen brings into the lane
/// *during* the cascade is not caught by it. **[ASSUMED]** — §8 says the player picks from
/// a queue, which reads as a fixed set.
pub(crate) fn flip_lane(state: &mut GameState, ctx: PowerCtx) {
    let queue = state.five_flip_targets(ctx.owner, ctx.lane, ctx.id);
    if !queue.is_empty() {
        state.pending.push(Pending::ResolveOrder {
            kind: ResolveKind::FiveFlip,
            player: ctx.owner,
            lane: ctx.lane as u8,
            remaining: queue,
        });
    }
}
