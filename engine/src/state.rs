//! The game state, and the queries the rules are written in terms of.
//!
//! Action *application* lives in `apply.rs` and legal-action *enumeration* lives in
//! `legal.rs`; both are written against the query helpers at the bottom of this file, so
//! there is exactly one implementation of "who can legally be attacked here" and the two
//! can never disagree.

use std::collections::VecDeque;

use crate::action::Phase;
use crate::card::{Card, CardId, PairId};
use crate::config::{GameConfig, Variant};
use crate::outcome::Outcome;
use crate::player::Player;
use crate::rank::{Rank, RankCounts};
use crate::rng::Rng;

/// One lane, with a side per player. Cards are ordered by play order and the vector
/// compacts when a card dies, so **slot indices are not stable** — see [`CardId`].
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Lane {
    pub sides: [Vec<Card>; 2],
}

impl Lane {
    #[inline]
    pub fn side(&self, p: Player) -> &Vec<Card> {
        &self.sides[p.idx()]
    }
    #[inline]
    pub fn side_mut(&mut self, p: Player) -> &mut Vec<Card> {
        &mut self.sides[p.idx()]
    }
}

/// An **ordered** draw pile.
///
/// `DESIGN.md` §3: "The draw pile is ordered, not a multiset: the 2's scry puts a known card
/// at a known position — the bottom — and the player who put it there knows both." A
/// multiset cannot represent that, and without it the 2 cannot be valued correctly.
///
/// `known_to` runs parallel to `cards`, one two-player bitmask per position, so the private
/// information created by a bottoming survives every later draw automatically.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Pile {
    /// Front is the **top** (next card drawn); back is the **bottom**.
    cards: VecDeque<Rank>,
    known_to: VecDeque<u8>,
}

impl Pile {
    pub fn from_ranks(ranks: Vec<Rank>) -> Pile {
        let known_to = vec![0u8; ranks.len()].into();
        Pile {
            cards: ranks.into(),
            known_to,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cards.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Take the top card. Returns the rank and the mask of players who already knew it.
    pub fn draw(&mut self) -> Option<(Rank, u8)> {
        let rank = self.cards.pop_front()?;
        let known = self.known_to.pop_front().unwrap_or(0);
        Some((rank, known))
    }

    /// Put a card on the **bottom**, known to `knower` (`game_rules.md` §10a).
    pub fn put_on_bottom(&mut self, rank: Rank, knower: Player) {
        self.cards.push_back(rank);
        self.known_to.push_back(knower.bit());
    }

    /// Ranks from top to bottom. Engine-side ground truth — never show this to a player.
    pub fn ranks(&self) -> impl Iterator<Item = Rank> + '_ {
        self.cards.iter().copied()
    }

    /// Every position top-first, as `(rank, known_to mask)`.
    ///
    /// Engine-side ground truth in the rank; the mask, by contrast, is **public** — that a
    /// player fired a 2 and put *something* on the bottom is visible to both. Determinization
    /// relies on exactly that split: it resamples the ranks it is not entitled to and leaves
    /// every mask alone (see [`GameState::determinize`]).
    pub fn entries(&self) -> impl Iterator<Item = (Rank, u8)> + '_ {
        self.cards.iter().copied().zip(self.known_to.iter().copied())
    }

    /// Overwrite the rank at `index`, counted from the top, leaving its `known_to` mask
    /// untouched. Only determinization does this: it is how a sampled world gets a pile
    /// order consistent with the observer's information set.
    pub(crate) fn set_rank(&mut self, index: usize, rank: Rank) {
        self.cards[index] = rank;
    }

    /// What `observer` knows about this pile, bottom-most first: `Some(rank)` for a card
    /// they put there, `None` for a card they cannot see. Feeds the "bottomed-card
    /// features" of `DESIGN.md` §5.
    pub fn known_from_bottom(&self, observer: Player) -> Vec<Option<Rank>> {
        self.cards
            .iter()
            .zip(self.known_to.iter())
            .rev()
            .map(|(r, k)| {
                if k & observer.bit() != 0 {
                    Some(*r)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Which list a [`Pending::ResolveOrder`] node is working through.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResolveKind {
    /// A 5 flipping your face-down cards in its lane (`game_rules.md` §6).
    FiveFlip,
    /// A King refiring your face-up powers in its lane (§6).
    KingEmpower,
}

impl ResolveKind {
    pub const fn label(self) -> &'static str {
        match self {
            ResolveKind::FiveFlip => "5: flip",
            ResolveKind::KingEmpower => "K: reactivate",
        }
    }
}

/// A decision a power has opened that must be answered before the turn continues.
///
/// Held as a **stack**: a power fired mid-resolution pushes its own sub-decisions on top,
/// and they finish before control returns to the list underneath. That is what makes a 5
/// flipping a King flipping more cards resolve in the right order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Pending {
    /// The 4's Foresight is waiting for a face-down card to look at.
    Foresight { player: Player },
    /// A 5 or King is waiting for the next card to resolve. `remaining` holds card ids,
    /// not slots, because slots move when cards die or a Queen relocates them.
    ResolveOrder {
        kind: ResolveKind,
        player: Player,
        lane: u8,
        remaining: Vec<CardId>,
    },
    /// A Queen is waiting for the allied card to pull into `lane`.
    QueenSource { player: Player, lane: u8 },
    /// A 2 is waiting for the card to give back.
    GiveBack { player: Player },
    /// A 10 is waiting for the second half of its twinstrike. No damage has been dealt yet
    /// — the engine collects both targets first so the split lands simultaneously.
    SplitTarget {
        player: Player,
        lane: u8,
        /// The attacking card, or both members if this is a pair attack.
        attackers: Vec<CardId>,
        primary: CardId,
    },
}

impl Pending {
    /// The player who must answer. Always the player to move: every power resolves on its
    /// owner's turn.
    pub fn player(&self) -> Player {
        match self {
            Pending::Foresight { player }
            | Pending::ResolveOrder { player, .. }
            | Pending::QueenSource { player, .. }
            | Pending::GiveBack { player }
            | Pending::SplitTarget { player, .. } => *player,
        }
    }

    pub fn phase(&self) -> Phase {
        match self {
            Pending::Foresight { .. } => Phase::Foresight,
            Pending::ResolveOrder { .. } => Phase::ResolveOrder,
            Pending::QueenSource { .. } => Phase::QueenSource,
            Pending::GiveBack { .. } => Phase::GiveBack,
            Pending::SplitTarget { .. } => Phase::SplitTarget,
        }
    }
}

/// A complete Duel 52 position.
///
/// This is engine-side **ground truth**: it contains the opponent's hand, the draw pile
/// order, and the identity of the removed-unseen cards. Anything shown to a player must go
/// through [`crate::observation`] or [`crate::display`], which filter by what that observer
/// is entitled to know.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GameState {
    pub config: GameConfig,
    /// The seed this game was dealt from. Recorded so any position can be reproduced.
    pub seed: u64,

    pub lanes: Vec<Lane>,
    /// Each player's hand, kept sorted by rank so that equal positions compare equal.
    pub hands: [Vec<Rank>; 2],
    /// Draw piles. In the split variants, `piles[p]` belongs to player `p`. In the base
    /// game there is a single shared pile in `piles[0]` and `piles[1]` stays empty — see
    /// [`GameState::pile_index`].
    pub piles: [Pile; 2],
    /// Discards, split by the owner of the card. Public to both players either way
    /// (`game_rules.md` §5): "The discard pile is public and inspectable."
    pub discards: [Vec<Rank>; 2],

    /// The cards removed face-down at setup. **Ground truth, normally hidden from
    /// everyone** — `game_rules.md` §2: because they are removed unseen, "a player's belief
    /// over hidden cards never fully resolves, even at the end of the game."
    ///
    /// Indexed by the deck the cards came out of. In the split variants that is the owning
    /// player's colour deck, 5 each (§9a). In the **base** game the ten cards come off a
    /// shared deck and belong to nobody, so they are all filed under index 0 and
    /// `removed[1]` stays empty — do not read ownership into it there.
    pub removed: [Vec<Rank>; 2],
    /// True only in the mirrored-removal variant (§9b), where the removed multiset is
    /// revealed to both players at setup.
    pub removed_revealed: bool,

    pub to_move: Player,
    /// Individual player turns since the start, from 0. P0 owns even plies.
    pub ply: u32,
    pub actions_remaining: u32,

    /// Set once every draw pile is empty, and never cleared (`game_rules.md` §3).
    ///
    /// Drives base-card unlock, the 5/7/Q reaching base cards, and lane-win condition 2.
    /// The 2's View is the one gated rule that does **not** read this flag — it is tied to
    /// the pile you personally draw from (§9), so a 2 can go dead a turn before the global
    /// unlock.
    ///
    /// Recomputed only at action boundaries. That matters under the house 2: firing a 2 on
    /// a one-card pile takes it to zero and back to one within a single resolution, and
    /// §10a is explicit that this must not count as the pile emptying.
    pub base_unlocked: bool,

    /// Consecutive plies with no damage and no kill (`game_rules.md` §7).
    pub quiet_plies: u32,

    /// Total cards each player has drawn, from the turn-start draw and from 2s alike.
    ///
    /// Not a rule — pure instrumentation, and it exists for one specific question.
    /// `game_rules.md` §10a claims the rules-as-written 2 "turns a filtering effect into a
    /// lever on the draw count" by shrinking a **shared** pile, and `PLAN.md` asks for that
    /// claim to be "measurable rather than assumed". Counting draws per player is what
    /// makes the mechanism visible rather than merely inferred from win rates.
    pub draws_taken: [u32; 2],

    pub outcome: Outcome,

    /// Sub-decisions still owed by the action being resolved. Empty in [`Phase::Main`].
    pub pending: Vec<Pending>,

    pub(crate) next_card_id: u32,
    pub(crate) next_pair_id: u32,
    /// Did anything take damage or die this ply? Drives `quiet_plies`.
    pub(crate) damage_this_ply: bool,
    /// Reserved for Phase 3 determinization sampling (`DESIGN.md` §6); unused during play,
    /// since nothing after the deal is random.
    pub(crate) rng: Rng,
}

impl GameState {
    // ------------------------------------------------------------------- basic queries --

    /// Which decision is on the table.
    pub fn phase(&self) -> Phase {
        if self.outcome.is_over() {
            Phase::Terminal
        } else {
            match self.pending.last() {
                Some(p) => p.phase(),
                None => Phase::Main,
            }
        }
    }

    /// The player who must choose now. Always `to_move` — sub-decisions belong to the
    /// player whose action opened them.
    pub fn acting_player(&self) -> Player {
        self.to_move
    }

    /// Which entry of `piles` player `p` draws from.
    ///
    /// Base game: a single shared pile (`game_rules.md` §2). Split variants: your own
    /// colour (§9a).
    #[inline]
    pub fn pile_index(&self, p: Player) -> usize {
        if self.config.variant.is_split() {
            p.idx()
        } else {
            0
        }
    }

    #[inline]
    pub fn pile(&self, p: Player) -> &Pile {
        &self.piles[self.pile_index(p)]
    }

    #[inline]
    pub fn pile_mut(&mut self, p: Player) -> &mut Pile {
        let i = self.pile_index(p);
        &mut self.piles[i]
    }

    #[inline]
    pub fn hand(&self, p: Player) -> &Vec<Rank> {
        &self.hands[p.idx()]
    }

    pub fn hand_counts(&self, p: Player) -> RankCounts {
        crate::rank::rank_counts(&self.hands[p.idx()])
    }

    /// Would every draw pile be empty right now?
    ///
    /// In the base game only `piles[0]` is ever used and `piles[1]` is empty from setup, so
    /// this one predicate covers both the base rule ("the draw pile is empty") and the
    /// split rule ("**both** piles are empty", §9).
    pub fn all_piles_empty(&self) -> bool {
        self.piles.iter().all(|p| p.is_empty())
    }

    /// Number of lanes, from config.
    #[inline]
    pub fn lane_count(&self) -> usize {
        self.config.lanes
    }

    // -------------------------------------------------------------------- card lookups --

    /// Find a card by id: `(lane, side index, slot)`. `None` if it has left play.
    ///
    /// Called after every damage step, because the card may have died in between.
    pub fn locate(&self, id: CardId) -> Option<(usize, usize, usize)> {
        for (l, lane) in self.lanes.iter().enumerate() {
            for (s, side) in lane.sides.iter().enumerate() {
                if let Some(slot) = side.iter().position(|c| c.id == id) {
                    return Some((l, s, slot));
                }
            }
        }
        None
    }

    pub fn card(&self, id: CardId) -> Option<&Card> {
        let (l, s, slot) = self.locate(id)?;
        Some(&self.lanes[l].sides[s][slot])
    }

    pub fn card_mut(&mut self, id: CardId) -> Option<&mut Card> {
        let (l, s, slot) = self.locate(id)?;
        Some(&mut self.lanes[l].sides[s][slot])
    }

    /// Borrow the card at a slot, if that slot exists.
    pub fn at(&self, lane: usize, owner: Player, slot: usize) -> Option<&Card> {
        self.lanes.get(lane)?.side(owner).get(slot)
    }

    /// Every card in play belonging to `p`, as `(lane, slot, &Card)`.
    pub fn cards_of(&self, p: Player) -> impl Iterator<Item = (usize, usize, &Card)> + '_ {
        self.lanes.iter().enumerate().flat_map(move |(l, lane)| {
            lane.side(p)
                .iter()
                .enumerate()
                .map(move |(slot, c)| (l, slot, c))
        })
    }

    /// The other member of a card's pair, if it is paired.
    ///
    /// A pair is always two cards on the same side of the same lane, so the search never
    /// leaves that vector.
    pub fn pair_partner(&self, lane: usize, owner: Player, slot: usize) -> Option<usize> {
        let side = self.lanes[lane].side(owner);
        let pid = side[slot].pair_id?;
        side.iter()
            .position(|c| c.pair_id == Some(pid) && c.id != side[slot].id)
    }

    /// The cards that would attack together if `slot` attacks: just itself, or both members
    /// if it is paired.
    ///
    /// `game_rules.md` §5: "Pairs must attack together — a paired card cannot attack alone."
    pub fn attack_group(&self, lane: usize, owner: Player, slot: usize) -> Vec<CardId> {
        let side = self.lanes[lane].side(owner);
        match self.pair_partner(lane, owner, slot) {
            Some(other) => vec![side[slot].id, side[other].id],
            None => vec![side[slot].id],
        }
    }

    /// Can the card at `slot` attack this turn, accounting for its pair?
    ///
    /// A pair attack is one attack for **both** members' once-per-turn budget (§5), so it is
    /// unavailable if either member has already attacked — you cannot attack with two cards
    /// separately and *then* pair them for extra damage.
    pub fn can_attack_from(&self, lane: usize, owner: Player, slot: usize) -> bool {
        let side = self.lanes[lane].side(owner);
        let Some(card) = side.get(slot) else {
            return false;
        };
        if !card.can_attack(self.ply) {
            return false;
        }
        match self.pair_partner(lane, owner, slot) {
            Some(other) => side[other].can_attack(self.ply),
            None => true,
        }
    }

    // ------------------------------------------------------------------ target queries --

    /// Slots on `defender`'s side of `lane` that may legally be attacked.
    ///
    /// Two filters, in this order:
    ///
    /// 1. **Base cards are untouchable while any draw pile is non-empty** (`game_rules.md`
    ///    §3). This is why lane wins are strictly an endgame event.
    /// 2. **Jack taunt** (§6): if any *face-up* Jack is attackable, only Jacks are. Taunt is
    ///    a power, so a face-down Jack does not taunt (§6: "Powers are inert while a card is
    ///    face-down"). With more than one Jack the attacker chooses which to hit — there is
    ///    no "oldest first" ordering (§5).
    pub fn legal_attack_targets(&self, lane: usize, defender: Player) -> Vec<usize> {
        let side = self.lanes[lane].side(defender);
        let targetable: Vec<usize> = (0..side.len())
            .filter(|&i| self.base_unlocked || !side[i].is_base)
            .collect();

        let jacks: Vec<usize> = targetable
            .iter()
            .copied()
            .filter(|&i| side[i].has_live_power(Rank::JACK))
            .collect();

        if jacks.is_empty() {
            targetable
        } else {
            jacks
        }
    }

    /// Slots that could take the **second** half of a 10's twinstrike, given the primary.
    ///
    /// `game_rules.md` §6 and §8. The 9 and the Jack both block the split, for different
    /// reasons, and the difference shows up in the two-card case:
    ///
    /// - **9 — Nimble.** Blocks *personally*. A 9 never takes a spread half, so a 9 as the
    ///   primary kills the split outright and a 9 elsewhere in the lane is not an eligible
    ///   second target. Two 9s in a lane still means "1 damage to one 9".
    /// - **Jack — Taunt.** Blocks by *leakage*: taunt confines the attack to Jacks, so with
    ///   one Jack there is nowhere for the second half to go. With **two** Jacks both halves
    ///   are already confined to Jacks and nothing leaks past, so the 10 deals 1 to each.
    ///
    /// `game_rules.md` §8 warns explicitly: "Do not unify these into one 'blocker' concept
    /// in the engine; they are different mechanics that happen to share a symptom in the
    /// one-card case."
    ///
    /// Both checks read `has_live_power`, so a **face-down** 9 or Jack blocks nothing.
    pub fn twinstrike_split_candidates(
        &self,
        lane: usize,
        defender: Player,
        primary_slot: usize,
    ) -> Vec<usize> {
        let side = self.lanes[lane].side(defender);
        let Some(primary) = side.get(primary_slot) else {
            return Vec::new();
        };
        // Nimble dodges the spread personally.
        if primary.has_live_power(Rank::NINE) {
            return Vec::new();
        }
        self.legal_attack_targets(lane, defender)
            .into_iter()
            .filter(|&i| i != primary_slot)
            .filter(|&i| !side[i].has_live_power(Rank::NINE))
            .collect()
    }

    /// How much damage an attack by `attacker_rank` deals to one target.
    ///
    /// Base is 1 for a single card, 2 for a pair. The only rank modifier that changes the
    /// *amount* is the 9's "deals 2 damage to Jacks" (`game_rules.md` §6), which doubles —
    /// so a lone 9 deals 2 to a Jack and a **pair of 9s deals 4**, one-shotting a 3-HP Jack
    /// (§5).
    ///
    /// It applies only to a **face-up** Jack. §5 is explicit that a face-down card is a
    /// blank 2-HP card whatever its rank, so there is no Jack there for the 9 to be good
    /// against — a 9 hitting a face-down Jack deals its ordinary 1, and kills it in two
    /// like anything else. This is consistent with the twinstrike rule: everything that
    /// keys on a target being a Jack reads `has_live_power`, never the bare rank.
    pub fn attack_damage(&self, attacker_rank: Rank, target: &Card, is_pair: bool) -> u8 {
        let base = if is_pair { 2 } else { 1 };
        if attacker_rank == Rank::NINE && target.has_live_power(Rank::JACK) {
            base * 2
        } else {
            base
        }
    }

    /// Face-down cards anywhere on the board — the legal targets for a 4's Foresight.
    ///
    /// `game_rules.md` §6: "Look at any one face-down card on the board — including base
    /// cards, yours or your opponent's." Cards the peeker already knows are still legal
    /// (wasteful, but legal); the rules impose no such filter.
    pub fn face_down_cards(&self) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for (l, lane) in self.lanes.iter().enumerate() {
            for (s, side) in lane.sides.iter().enumerate() {
                for (slot, card) in side.iter().enumerate() {
                    if !card.face_up {
                        out.push((l, s, slot));
                    }
                }
            }
        }
        out
    }

    /// Cards a Queen in `queen_lane` could pull in: allied cards in some **other** lane.
    ///
    /// `game_rules.md` §6: "Move one allied card from another lane into the Queen's lane,
    /// face-down or face-up." Base cards only once the pile is empty; a moved base card
    /// becomes a normal card (§3).
    pub fn queen_move_sources(&self, player: Player, queen_lane: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (l, lane) in self.lanes.iter().enumerate() {
            if l == queen_lane {
                continue;
            }
            for (slot, card) in lane.side(player).iter().enumerate() {
                if card.is_base && !self.base_unlocked {
                    continue;
                }
                out.push((l, slot));
            }
        }
        out
    }

    // ------------------------------------------------------------------- id allocation --

    pub(crate) fn fresh_card_id(&mut self) -> CardId {
        let id = CardId(self.next_card_id);
        self.next_card_id += 1;
        id
    }

    pub(crate) fn fresh_pair_id(&mut self) -> PairId {
        let id = PairId(self.next_pair_id);
        self.next_pair_id += 1;
        id
    }

    // ---------------------------------------------------------------- integrity checks --

    /// Assert the invariants that would otherwise turn a rules bug into a silent wrong
    /// answer. Run after every action in debug builds.
    pub(crate) fn debug_check_invariants(&self) {
        if cfg!(debug_assertions) {
            for (l, lane) in self.lanes.iter().enumerate() {
                for p in Player::BOTH {
                    let side = lane.side(p);
                    assert!(
                        side.len() <= self.config.max_slots_per_side,
                        "lane {l} side {p} holds {} cards, over the encoding cap of {}. \
                         The rules impose no limit; raise config.max_slots_per_side.",
                        side.len(),
                        self.config.max_slots_per_side
                    );
                    for c in side {
                        assert_eq!(c.owner, p, "card {:?} filed on the wrong side", c.id);
                        assert!(
                            c.damage < c.max_hp(),
                            "dead card {:?} still in play with {} damage against {} max HP",
                            c.id,
                            c.damage,
                            c.max_hp()
                        );
                        assert!(
                            !(c.is_base && !c.entered_as_base),
                            "card {:?} is a base card but did not enter as one",
                            c.id
                        );
                        assert!(
                            !(c.face_up && c.known_to != 0b11),
                            "face-up card {:?} must be known to both players",
                            c.id
                        );
                        // A pair is a matching: exactly one partner, same lane and side.
                        if let Some(pid) = c.pair_id {
                            let members = side.iter().filter(|o| o.pair_id == Some(pid)).count();
                            assert_eq!(
                                members, 2,
                                "pair {pid:?} in lane {l} has {members} members, not 2"
                            );
                        }
                    }
                }
            }
            if let Some(top) = self.pending.last() {
                assert_eq!(
                    top.player(),
                    self.to_move,
                    "a pending sub-decision belongs to the player who is not to move"
                );
            }

            // A running game must always offer a way forward. Without this, a bug that
            // left an unanswerable sub-decision on the stack would present as the engine
            // silently hanging, which is far harder to diagnose than a failed assertion.
            if !self.outcome.is_over() {
                assert!(
                    !self.legal_actions().is_empty(),
                    "no legal action is available but the game is not over: {}",
                    self.header()
                );
            }
        }
    }

    /// Total cards accounted for. Used by the setup tests to prove nothing was lost or
    /// duplicated during the deal.
    pub fn card_census(&self) -> usize {
        let in_play: usize = self
            .lanes
            .iter()
            .map(|l| l.sides[0].len() + l.sides[1].len())
            .sum();
        let in_hands = self.hands[0].len() + self.hands[1].len();
        let in_piles = self.piles[0].len() + self.piles[1].len();
        let in_discards = self.discards[0].len() + self.discards[1].len();
        let removed = self.removed[0].len() + self.removed[1].len();
        in_play + in_hands + in_piles + in_discards + removed
    }

    /// Every removed card, regardless of which deck it came from.
    pub fn all_removed(&self) -> impl Iterator<Item = Rank> + '_ {
        self.removed[0].iter().chain(self.removed[1].iter()).copied()
    }

    /// Total cards the configuration says exist.
    pub fn expected_card_count(&self) -> usize {
        if self.config.variant.is_split() {
            2 * self.config.split_deck_size()
        } else {
            self.config.full_deck_size()
        }
    }

    /// True when this game uses one shared pile (the base variant).
    pub fn shared_pile(&self) -> bool {
        !self.config.variant.is_split()
    }

    /// Human-readable one-liner for logs.
    pub fn header(&self) -> String {
        format!(
            "ply {} · {} to move · {} action(s) left · pile {} · base {} · quiet {}/{}",
            self.ply,
            self.to_move,
            self.actions_remaining,
            if self.shared_pile() {
                format!("{}", self.piles[0].len())
            } else {
                format!("P0:{} P1:{}", self.piles[0].len(), self.piles[1].len())
            },
            if self.base_unlocked {
                "UNLOCKED"
            } else {
                "locked"
            },
            self.quiet_plies,
            self.config.stalemate_quiet_plies,
        )
    }
}

/// Construction lives in `setup.rs`; this is only the shell the dealer fills in.
pub(crate) fn empty_state(config: GameConfig, seed: u64) -> GameState {
    GameState {
        lanes: vec![Lane::default(); config.lanes],
        hands: [Vec::new(), Vec::new()],
        piles: [Pile::default(), Pile::default()],
        discards: [Vec::new(), Vec::new()],
        removed: [Vec::new(), Vec::new()],
        removed_revealed: config.variant == Variant::MirroredRemoval,
        to_move: Player::P0,
        ply: 0,
        actions_remaining: 0,
        base_unlocked: false,
        quiet_plies: 0,
        draws_taken: [0, 0],
        outcome: Outcome::Ongoing,
        pending: Vec::new(),
        next_card_id: 0,
        next_pair_id: 0,
        damage_this_ply: false,
        rng: Rng::derive(seed, 0xD0E1_5205_2000_0001),
        config,
        seed,
    }
}
