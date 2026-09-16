//! The 2 — **View**, and the **Blast** variant.

use super::{LethalOutcome, PowerCtx};
use crate::damage::{DamageSource, Hit};
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

/// **Blast** — "if killed while face-down, deals `damage` to every enemy card in its lane".
///
/// Not a rule of the canonical ruleset: a second face-down trap in the 3's family, and the
/// first whose payoff is a **lane** rather than one card. It **replaces** View rather than
/// adding to it **[ASSUMED]**, as [`super::four::bomb`] replaces Foresight, so flipping a
/// Blast draws nothing and a face-up 2 is a plain 2-HP card. `two_power` is inert under it.
///
/// What it hits, and the calls behind it:
///
/// - **Every enemy card in the lane, face-up or face-down**, whatever killed the 2 — an
///   attack, or another 2's Blast.
/// - **Not base cards before the unlock.** §3 makes them untouchable while a pile remains,
///   and `seven::heal_all` makes the same exception. **[ASSUMED]**
/// - **Not a taunt-respecting hit, and no Nimble dodge.** Taunt restricts what an *attack*
///   may target, and Nimble's immunities are to freeze, retaliate and the 10's spread; a
///   blast is none of those, so a Jack takes 1 and so does a 9. **[ASSUMED]**
///
/// The hits are queued in slot order and land after the 2 has left play and after
/// everything already in flight (`damage.rs`).
///
/// # Why this can chain, and why the chain ends
///
/// A blast is the first death trigger that can reach a **face-down** card, so it can set off
/// another trap: spring a 3, detonate a 4 (which takes nobody with it — see `bomb`), or kill
/// a face-down enemy 2 whose own blast comes straight back across the lane. What ends that
/// is not [`DamageSource::attackers`] but a counting argument: every death trigger in the
/// engine fires only on a **face-down** card, and afterwards the card has either left play
/// or turned face-up, and nothing ever turns a card face-down again (`game_rules.md` §7). So
/// each card fires at most once, and a cascade is bounded by the cards on the board.
pub(crate) fn blast(state: &mut GameState, ctx: PowerCtx, damage: u8) -> LethalOutcome {
    if state.lanes[ctx.lane].sides[ctx.side][ctx.slot].face_up {
        return LethalOutcome::Die;
    }
    let unlocked = state.base_unlocked;
    let enemy = ctx.owner.other();
    // Indexed rather than iterated, because enqueueing borrows the state. Enqueueing does not
    // touch the board, so the length cannot change under the loop.
    for i in 0..state.lanes[ctx.lane].side(enemy).len() {
        let card = &state.lanes[ctx.lane].side(enemy)[i];
        if card.is_base && !unlocked {
            continue;
        }
        let target = card.id;
        state.enqueue_damage(Hit {
            target,
            amount: damage,
            source: DamageSource::Blast { from: ctx.id },
        });
    }
    LethalOutcome::Die
}
