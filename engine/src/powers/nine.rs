//! The 9 — **Nimble**.
//!
//! "Cannot be frozen; takes no retaliate damage; deals 2 damage to Jacks; dodges a
//! twinstrike's spread." (`game_rules.md` §6)
//!
//! Nimble is four immunities rather than one, and each is read where it applies rather than
//! from a central place:
//!
//! | Immunity | Read by |
//! |---|---|
//! | Cannot be frozen | [`crate::powers::six::freeze`] |
//! | No retaliate damage | `apply.rs`'s `resolve_attack` |
//! | Dodges a spread | [`crate::state::GameState::twinstrike_split_candidates`] |
//! | Double to a taunter | [`crate::state::GameState::attack_damage`] |
//!
//! ⚠️ `game_rules.md` §8: "Do not unify these into one 'blocker' concept in the engine; they
//! are different mechanics that happen to share a symptom in the one-card case." That
//! warning is about the 9 and the Jack both blocking a twinstrike split, and it is the
//! reason [`crate::powers::PowerId`] exposes `is_nimble` and `taunts` as separate predicates
//! rather than one `blocks_split`.
//!
//! There is no code here because every one of the four is a *query* on a live power, and the
//! module that asks the question is the one that owns the answer. A 9 variant that changed
//! only one of the four would add a predicate to `PowerId` and change one row of that table.
