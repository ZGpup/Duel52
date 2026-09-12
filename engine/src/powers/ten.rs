//! The 10 — **Twinstrike**.
//!
//! "Attacks split 1 damage across two enemy cards." (`game_rules.md` §6)
//!
//! The split is a *sub-decision*, not part of the attack action: `apply.rs`'s `do_attack`
//! pushes [`crate::state::Pending::SplitTarget`] and the second target is chosen on its own
//! zero-cost decision node. Both targets are collected **before** any damage lands, so the
//! two halves are simultaneous and retaliate has no ambiguous ordering.
//!
//! Two rulings that are easy to get backwards:
//!
//! - A **pair** of 10s splits its 2 as 1 + 1 rather than doubling to 4 (§6).
//! - When the split is blocked, §5's "damage is never lost" promise applies to the *pair* —
//!   a pair of 10s consolidates to the full 2 on one card, but a lone 10's second point was
//!   the twinstrike bonus and goes away with it.
//!
//! ⚠️ A 10 that split **across lanes** would still be Tier 2: `CHOOSE_SLOT` spans every lane
//! already and `Phase::SplitTarget` already exists, so the change is a wider candidate list
//! and nothing else (`MODULAR_RULES.md` §2). The mask is free; the phase is not.
