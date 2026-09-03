//! A card instance in play.
//!
//! Cards only become instances when they hit the table. In a hand, a draw pile, the discard
//! pile, or the removed-unseen pool a card is just a [`Rank`], because nothing else about it
//! can differ. See `DESIGN.md` §3.

use crate::player::Player;
use crate::rank::Rank;

/// Stable identifier for a card *while it is in play*.
///
/// Slot indices shift whenever a card dies (slots compact) or a Queen moves a card between
/// lanes. Anything the engine remembers across those events — a 5's pending flip list, a
/// King's pending reactivation list, an in-flight attack — holds `CardId`s, never slots.
/// Getting this wrong is the single easiest way to introduce a rules bug that only shows up
/// in a cascade.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CardId(pub u32);

/// Identifier for a declared pair. Two cards on the same side of the same lane carrying the
/// same `PairId` are a pair.
///
/// `game_rules.md` §5: "A card belongs to at most one pair, so pairs are a *matching*, not
/// a graph." Storing the id on the card enforces that structurally — a card has one
/// `Option<PairId>`, so it cannot be in two pairs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PairId(pub u32);

/// A card on the table.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Card {
    pub id: CardId,
    pub rank: Rank,
    /// Whose side of the lane this card sits on. Also who owns it for discard bookkeeping.
    pub owner: Player,

    /// Face-up cards have live powers; face-down cards' powers are inert (`game_rules.md`
    /// §6). Hit points are *not* a power, so a face-down Jack still has 3 HP.
    pub face_up: bool,

    /// Still a base card *right now*: untouchable while the draw pile is non-empty, and
    /// counted as "in the lane" for the lane-win check (`game_rules.md` §3, §7).
    ///
    /// A Queen that moves a base card clears this — "A base card that a Queen moves to
    /// another lane stops being a base card."
    pub is_base: bool,

    /// This card *entered play* as a base card, whether or not it still is one.
    ///
    /// This is what gates the owner's free look at their own face-down cards
    /// (`game_rules.md` §4). Splitting it from `is_base` is load-bearing: collapse the two
    /// and moving your own base card with a Queen becomes a repeatable one-action peek at
    /// your own base, which is the wrong game (`DESIGN.md` §3).
    pub entered_as_base: bool,

    /// Damage taken. Killed when `damage >= rank.max_hp()`.
    ///
    /// `game_rules.md` §5: damage is **public**, including on face-down cards — a damaged
    /// card is turned sideways and both players can see that.
    pub damage: u8,

    /// The last ply on which this card is still frozen, if frozen.
    ///
    /// `game_rules.md` §8: "an enemy card frozen by a 6 is unfrozen at the **end of the
    /// frozen player's next turn** — so exactly one of their turns is lost." Because plies
    /// strictly alternate, the 6 resolving on ply `t` sets this to `t + 1`, the victim's
    /// next turn. The card is frozen while `current_ply <= frozen_until_ply`.
    ///
    /// Per **card**, not per lane: a 6 does not catch cards that arrive later, and a frozen
    /// card a Queen moves stays frozen (§8).
    pub frozen_until_ply: Option<u32>,

    /// Attacks this card has made **this turn**.
    pub attacks_used: u8,

    /// Attacks this card is allowed **this turn**. Normally 1.
    ///
    /// A freshly flipped Ace gets 2 (`game_rules.md` §6), and a King reactivating an Ace
    /// *resets* `attacks_used` to 0 and the allowance to 2 rather than stacking — so an Ace
    /// that attacked once and was then Kinged tops out at three attacks, not four. A plain
    /// `attacked_this_turn: bool` cannot represent either case (`DESIGN.md` §3).
    pub attack_allowance: u8,

    /// The pair this card belongs to, if any.
    pub pair_id: Option<PairId>,

    /// Bitmask of players who know this card's rank. Bit `p` set means player `p` knows it.
    ///
    /// - Played face-down from hand → known to the owner only (`game_rules.md` §4: "You may
    ///   look at your own played face-down cards at any time, for free").
    /// - Dealt as a base card → known to **nobody**, owner included (§3). This is why the
    ///   4's Foresight can usefully target your own base cards.
    /// - Face-up → known to both.
    /// - A 4's Foresight sets the peeker's bit. That knowledge is private and persistent.
    pub known_to: u8,
}

impl Card {
    /// A card entering play from a hand: face-down, undamaged, known to its owner.
    pub fn played_from_hand(id: CardId, rank: Rank, owner: Player) -> Card {
        Card {
            id,
            rank,
            owner,
            face_up: false,
            is_base: false,
            entered_as_base: false,
            damage: 0,
            frozen_until_ply: None,
            attacks_used: 0,
            attack_allowance: 1,
            pair_id: None,
            known_to: owner.bit(),
        }
    }

    /// A base card dealt at setup: face-down and known to nobody.
    pub fn base_card(id: CardId, rank: Rank, owner: Player) -> Card {
        Card {
            id,
            rank,
            owner,
            face_up: false,
            is_base: true,
            entered_as_base: true,
            damage: 0,
            frozen_until_ply: None,
            attacks_used: 0,
            attack_allowance: 1,
            pair_id: None,
            known_to: 0,
        }
    }

    /// Remaining hit points.
    #[inline]
    pub fn hp_remaining(&self) -> u8 {
        self.rank.max_hp().saturating_sub(self.damage)
    }

    /// True once damage has reached max HP. Note that a face-down 3 in this condition does
    /// not die — its Trap fires instead (`game_rules.md` §6).
    #[inline]
    pub fn is_dead(&self) -> bool {
        self.damage >= self.rank.max_hp()
    }

    /// Frozen as of `ply`?
    #[inline]
    pub fn is_frozen(&self, ply: u32) -> bool {
        matches!(self.frozen_until_ply, Some(last) if ply <= last)
    }

    /// Does player `observer` know this card's rank?
    #[inline]
    pub fn rank_known_to(&self, observer: Player) -> bool {
        self.face_up || (self.known_to & observer.bit()) != 0
    }

    /// Can this card still attack this turn, ignoring pair and target constraints?
    ///
    /// Requires: face-up (a face-down card "cannot attack and has no power", §4), not
    /// frozen (freeze blocks attacking, §8), and attack budget left (§4).
    #[inline]
    pub fn can_attack(&self, ply: u32) -> bool {
        self.face_up && !self.is_frozen(ply) && self.attacks_used < self.attack_allowance
    }

    /// True if this card's *constant* power is live: it must be face-up.
    ///
    /// `game_rules.md` §6: "Powers are inert while a card is face-down." So a face-down 8
    /// does not retaliate, a face-down Jack does not taunt, a face-down 9 is not Nimble,
    /// and a face-down 10 does not twinstrike.
    #[inline]
    pub fn has_live_power(&self, rank: Rank) -> bool {
        self.face_up && self.rank == rank
    }

    /// Reset the per-turn attack budget. Called at the start of the owner's turn.
    #[inline]
    pub fn reset_turn_attacks(&mut self) {
        self.attacks_used = 0;
        self.attack_allowance = 1;
    }
}
