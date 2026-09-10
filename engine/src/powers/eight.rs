//! The 8 — **Retaliate**.
//!
//! `MODULAR_RULES.md` §3b: the 8 is a better argument for card modules than the 3 is. Its
//! rule is not "when damaged, hit back" — it is a genuine back-edge in the damage graph with
//! three separate exceptions written into `resolve_attack` (read-before-damage, the 9's
//! immunity, and both members of a pair paying). Under the old shape those exceptions lived
//! in the turn machinery; here they live with the card.

use super::{PowerCtx, RetaliateMode};
use crate::card::CardId;
use crate::state::GameState;

/// Retaliate owed by the targets of one attack.
///
/// Two readings, because the two modes disagree about *when* the question is asked:
///
/// - [`RetaliateMode::Always`] is the rules as written. `game_rules.md` §6 + §8: "Retaliate
///   (8) resolves *after* the attacker's damage is applied, and fires even if that damage
///   killed the 8." So the set is read **before** damage and paid out after.
/// - [`RetaliateMode::OnSurvival`] asks after the damage has landed, so a dead 8 owes
///   nothing. The ids are carried across rather than the count.
#[derive(Clone, Debug, Default)]
pub(crate) struct Retaliation {
    /// Retaliators read before damage landed, which owe a hit whatever happens to them.
    pub committed: Vec<CardId>,
    /// Retaliators to re-check once damage has landed.
    pub on_survival: Vec<CardId>,
}

/// Read the retaliators among an attack's targets, **before** any damage is applied.
pub(crate) fn read(state: &GameState, hits: &[(CardId, u8)]) -> Retaliation {
    let config = state.config;
    let mut out = Retaliation::default();
    for &(id, _) in hits {
        let Some(card) = state.card(id) else { continue };
        let Some(power) = card.live_power(&config) else {
            continue;
        };
        match power.retaliate_mode() {
            RetaliateMode::Never => {}
            RetaliateMode::Always => out.committed.push(id),
            RetaliateMode::OnSurvival => out.on_survival.push(id),
        }
    }
    out
}

/// Every retaliator that is owed a hit, once the attack's damage has landed.
///
/// The `on_survival` ids are re-read here: a card that left play is gone, and one that is
/// still on the table has survived. A 3 whose Trap sprang counts as having survived, which
/// is the right reading — it is alive and face-up.
///
/// Returned as ids rather than a count so each hit can name the card that dealt it, which
/// is what [`crate::damage::DamageSource::Retaliate`] carries.
pub(crate) fn settle(state: &GameState, r: &Retaliation) -> Vec<CardId> {
    let config = state.config;
    let mut owed = r.committed.clone();
    owed.extend(r.on_survival.iter().copied().filter(|&id| {
        state
            .card(id)
            .and_then(|c| c.live_power(&config))
            .is_some_and(|p| p.retaliate_mode() == RetaliateMode::OnSurvival)
    }));
    owed
}

/// Damage one retaliate hit deals. A Tier-1 knob (`MODULAR_RULES.md` §2).
#[inline]
pub(crate) fn damage_per_hit(state: &GameState) -> u8 {
    state.config.eight_retaliate_damage
}

/// Nothing happens when an 8 is flipped; the power is constant.
#[allow(dead_code)]
pub(crate) fn on_flip(_state: &mut GameState, _ctx: PowerCtx) {}
