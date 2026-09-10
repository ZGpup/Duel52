//! Damage attribution, and the queue that carries it.
//!
//! # Why damage has a source at all
//!
//! Before `MODULAR_RULES.md` §5c, `damage_card(id, amount)` did not know who dealt the
//! damage. That was enough for the rules as written, because the only power that reacts to
//! being damaged — the 3's Trap — reacts by restoring *itself* and never looks outward. The
//! moment a card wants to hit back on death ("the 3 damages the card that killed it") the
//! attribution has to exist, and it has to exist at the point damage is *applied* rather
//! than at the point an attack is declared, because the two are separated by the retaliate
//! step.
//!
//! [`DamageSource`] is `Copy` and holds ids rather than references, so it survives the
//! slot-compaction that a kill triggers. `game_rules.md` and `CLAUDE.md` are both explicit
//! that anything remembered across a resolution step holds [`CardId`]s, never slots.
//!
//! # Why a queue rather than recursion
//!
//! A death trigger that deals damage can kill another card whose death trigger deals damage.
//! Recursion would make the ordering implicit in the call stack and would let one cascade
//! interleave with another's `for` loop — the failure mode `MODULAR_RULES.md` §3a describes,
//! where a 10 twinstrikes two face-down 3s and the second 3's vengeance is silently dropped
//! because the loop that spawned it has already moved on.
//!
//! A FIFO queue makes the order explicit and total: every hit of one attack lands, then
//! everything those hits triggered lands, then everything *those* triggered, and so on. It
//! also makes the depth bound observable rather than a property of the stack, which is what
//! [`DamageQueue::MAX_CASCADE`] asserts against.
//!
//! ⚠️ This module is deliberately **behaviour-neutral for the canonical ruleset**. With
//! `powers.three = "trap"` and `powers.eight = "retaliate"` nothing enqueues damage from
//! inside a drain, so the queue drains in exactly the order the old nested loops ran in.
//! The 354 tests that predate it are the proof.

use crate::card::CardId;

/// Who dealt a point of damage, and by what mechanism.
///
/// Held per *hit*, not per attack: a pair's two members are one source, but the retaliate
/// they provoke is a different source with a different card behind it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DamageSource {
    /// A declared attack (`game_rules.md` §4). `attacker` is the card the player chose;
    /// `partner` is the other member when the attack was made by a pair (§5), because a
    /// pair attacks as one action and anything that hits back hits **both** members.
    Attack {
        attacker: CardId,
        partner: Option<CardId>,
    },
    /// An 8's Retaliate (§6). The 8 is the source; the target is whoever attacked it.
    Retaliate { from: CardId },
    /// A death trigger hitting back — the `trap_vengeance_*` variants of the 3.
    ///
    /// Not a rule of the canonical ruleset. It is named here rather than folded into
    /// `Attack` so that a power can tell "I was attacked" from "I was caught in a death
    /// throe", which is what keeps the cascade depth bounded (`MODULAR_RULES.md` §3).
    Vengeance { from: CardId },
    /// Damage with no card behind it: `testkit` positions, and anything the engine applies
    /// as bookkeeping rather than as a rule.
    Unattributed,
}

impl DamageSource {
    /// An ordinary single-card attack.
    #[inline]
    pub const fn attack(attacker: CardId) -> DamageSource {
        DamageSource::Attack {
            attacker,
            partner: None,
        }
    }

    /// The cards that should be considered "the attacker" for a power that hits back.
    ///
    /// One id for a lone attacker, two for a pair, none for retaliate, vengeance or
    /// unattributed damage. **That last part is the load-bearing one**: it is what makes
    /// the cascade finite without a depth limit. A vengeance hit is not an attack, so it
    /// cannot provoke another vengeance, so a chain of death triggers cannot close into a
    /// cycle. `MODULAR_RULES.md` §3a walks through why this holds for every variant
    /// currently implemented, and what would break it.
    pub fn attackers(self) -> impl Iterator<Item = CardId> {
        let (a, b) = match self {
            DamageSource::Attack { attacker, partner } => (Some(attacker), partner),
            _ => (None, None),
        };
        a.into_iter().chain(b)
    }

    /// The single card behind this damage, whatever the mechanism. Used for reporting.
    #[inline]
    pub const fn dealt_by(self) -> Option<CardId> {
        match self {
            DamageSource::Attack { attacker, .. } => Some(attacker),
            DamageSource::Retaliate { from } => Some(from),
            DamageSource::Vengeance { from } => Some(from),
            DamageSource::Unattributed => None,
        }
    }

    /// Was this damage dealt by a declared attack, as opposed to a reaction to one?
    #[inline]
    pub const fn is_attack(self) -> bool {
        matches!(self, DamageSource::Attack { .. })
    }
}

/// One queued hit: apply `amount` to `target`, attributed to `source`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Hit {
    pub target: CardId,
    pub amount: u8,
    pub source: DamageSource,
}

/// A FIFO of hits waiting to land.
///
/// Lives on [`crate::state::GameState`] rather than being a local, because a death trigger
/// enqueues into it from deep inside a power body and must not need the queue threaded
/// through every signature to get there.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct DamageQueue {
    hits: std::collections::VecDeque<Hit>,
    /// Set while [`crate::state::GameState::drain_damage`] is running, so a nested call
    /// enqueues instead of starting a second drain. Without it, a death trigger firing
    /// mid-drain would run its own drain to completion and invert the order.
    draining: bool,
}

impl DamageQueue {
    /// Hits one drain may apply before the engine calls it a rules bug.
    ///
    /// Generous by design: the bound exists to turn a non-terminating cascade into a panic
    /// with a stack trace rather than a hung training run, not to constrain rule design. A
    /// legal position has at most a few dozen cards, and every hit either damages a card
    /// that has finite hit points or is dropped.
    pub const MAX_CASCADE: usize = 512;

    #[inline]
    pub fn push(&mut self, hit: Hit) {
        self.hits.push_back(hit);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<Hit> {
        self.hits.pop_front()
    }

    #[inline]
    pub fn is_draining(&self) -> bool {
        self.draining
    }

    #[inline]
    pub fn set_draining(&mut self, v: bool) {
        self.draining = v;
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }
}
