//! The Jack — **Taunt**.
//!
//! "Must be killed before anything else in the lane; extra HP while face-up."
//! (`game_rules.md` §6)
//!
//! Taunt is read by [`crate::state::GameState::legal_attack_targets`]: if any *face-up* Jack
//! is attackable, only Jacks are. A face-down Jack does not taunt, because powers are inert
//! while face-down (§6).
//!
//! # The extra hit point is part of the power, not of the rank
//!
//! `game_rules.md` §5: "Every card has 2 hit points, except the Jack, which has 3" — but a
//! **face-down card is a blank 2-HP card whatever its rank**. The third point arrives with
//! the flip, exactly like the taunt does, which is why it is keyed on
//! [`crate::powers::PowerId::taunts`] and read from `config.jack_hp` rather than from the
//! rank. Setting `jack_hp = 2` is therefore a complete, coherent ruleset: a Jack that still
//! taunts but dies as fast as everything else.
//!
//! That the hit point is public is load-bearing for belief: damage is public (§5), so if a
//! face-down card could survive two hits, that fact would leak its rank to both players.
//! Any variant that gave a face-down card extra HP would leak, and the information-hiding
//! invariant in the cross-ruleset suite is what would catch it.
//!
//! ⚠️ A Jack that absorbed **one attack per turn** would be Tier 3, not Tier 2: `attacks_used`
//! counts attacks *made*, not absorbed, so it needs a new per-card counter and therefore a
//! new observation feature (`MODULAR_RULES.md` §2).
