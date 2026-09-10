//! The King — **Empower**.

use super::PowerCtx;
use crate::state::{GameState, Pending, ResolveKind};

/// "All your face-up cards in this lane reactivate their powers. Does not affect other
/// Kings. Does not affect constant powers." (`game_rules.md` §6)
///
/// Because Kings cannot activate Kings, no infinite loop is possible — §6 says so
/// explicitly, and it is worth noting the engine relies on it rather than on a depth limit.
///
/// ⚠️ A power variant that made a King reactivate other Kings would break `game_rules.md`
/// §7's finiteness argument, not merely make the game longer. The cross-ruleset invariant
/// suite treats a `PlyLimit` draw as a test failure for exactly this reason.
pub(crate) fn empower(state: &mut GameState, ctx: PowerCtx) {
    let queue = state.king_reactivation_targets(ctx.owner, ctx.lane, ctx.id);
    if !queue.is_empty() {
        state.pending.push(Pending::ResolveOrder {
            kind: ResolveKind::KingEmpower,
            player: ctx.owner,
            lane: ctx.lane as u8,
            remaining: queue,
        });
    }
}
