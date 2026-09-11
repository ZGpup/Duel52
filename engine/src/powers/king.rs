//! The King — **Empower**.

use super::PowerCtx;
use crate::state::{GameState, LaneChoice, Pending, ResolveKind};

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

/// **Empower (any lane)** — as [`empower`], but the King reactivates a lane of its owner's
/// choosing rather than its own.
///
/// The first ruleset to claim `MODULAR_RULES.md` §7's `CHOOSE_LANE` block, and a
/// demonstration that one `2·L` block covers a power that may only name **its own** side:
/// `legal_lane_choices` emits nothing but `Side::Mine`, so the three `Theirs` logits are
/// never legal here, while a power that targeted an enemy lane would use the same block from
/// the other end. That is why the block is `2·L` rather than two blocks of `L`.
///
/// It also needs `Phase::ChooseLane`, which is why this is the variant that exercises reserve
/// item 1 as well: the base layout's `phase_onehot` has no position for it.
///
/// The King is still excluded from the queue it builds, so §6's "does not affect other Kings"
/// and `game_rules.md` §7's finiteness argument hold exactly as they do for [`empower`] —
/// choosing the lane changes *which* cards refire, not *whether* a King can refire a King.
/// A lane with nothing to reactivate is not offered at all (§8: a power with no legal target
/// fizzles), so if no lane has a target the whole power fizzles and no node is pushed.
pub(crate) fn empower_any_lane(state: &mut GameState, ctx: PowerCtx) {
    let any = (0..state.config.lanes)
        .any(|lane| !state.king_reactivation_targets(ctx.owner, lane, ctx.id).is_empty());
    if any {
        state.pending.push(Pending::ChooseLane {
            player: ctx.owner,
            kind: LaneChoice::KingEmpower { king: ctx.id },
        });
    }
}
