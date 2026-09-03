//! Determinization — sampling a concrete world from an information set.
//!
//! `DESIGN.md` §6, step 1: "Sample a **determinization** — a concrete hidden state
//! consistent with the acting player's information set: opponent hand, hidden base cards,
//! draw pile order, *and which 10 cards were removed*. The removed pool must be sampled;
//! treating it as known is the classic way to get a subtly wrong agent here."
//!
//! This is the seam that separates an honest search agent from a cheating one. An agent
//! gets handed the engine's ground-truth [`GameState`] — it has to, because the engine is
//! the sole authority on legality — so nothing structurally stops it from reading the
//! opponent's hand. The discipline every Phase 2 search agent follows instead is:
//!
//! 1. call [`GameState::determinize`] with itself as the observer,
//! 2. search the **sampled** world,
//! 3. never read a hidden field of the real one.
//!
//! `engine/tests/agents.rs` enforces that mechanically rather than by inspection, via the
//! property in [`GameState::determinize`]'s "What this guarantees" note.
//!
//! # The pools
//!
//! Which cards can be where is a per-deck constraint, so the sampler works one **pool** at
//! a time:
//!
//! - **Base variant** (`game_rules.md` §2) — one shared 52-card deck, so one pool covering
//!   both players, both piles, and the ten removed cards.
//! - **Split variants** (§9a, §9b) — each player owns a 26-card colour deck holding two of
//!   every rank, and `Card::owner` never changes, so there are two independent pools. A
//!   card that is hidden on P1's side can only be a card missing from P1's colour deck.
//!
//! Within a pool the sampler subtracts every rank the observer can *place* and deals the
//! remainder uniformly over the positions they cannot.

use crate::card::CardId;
use crate::player::Player;
use crate::rank::{Rank, RankCounts};
use crate::rng::Rng;
use crate::state::GameState;

/// One position whose rank `observer` is not entitled to know.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HiddenSlot {
    /// A face-down card in play. Every card in play that the observer cannot read is
    /// face-down by construction — a face-up card is known to both players — which is why
    /// resampling one can never break the `damage < max_hp` invariant: a face-down card is
    /// a blank 2-HP card whatever its rank (`game_rules.md` §5).
    InPlay(CardId),
    /// A position in the opponent's hand.
    Hand(Player, usize),
    /// A position in a draw pile, counted from the top.
    Pile(usize, usize),
    /// A position in a removed-unseen pool.
    Removed(usize, usize),
}

impl GameState {
    /// Sample a world consistent with what `observer` knows.
    ///
    /// The returned state is a full [`GameState`] that can be played out by the engine like
    /// any other. Everything `observer` is entitled to know is preserved exactly;
    /// everything else is redealt uniformly at random subject to the deck composition.
    ///
    /// # What is held fixed
    ///
    /// - The whole board except the ranks of face-down cards the observer cannot read —
    ///   positions, damage, freeze, pairs, attack budgets, `is_base`, `entered_as_base`.
    /// - `known_to` masks, everywhere. Determinization changes *which rank sits where*,
    ///   never *who knows what*. So a card the observer peeked at with a 4 keeps its rank,
    ///   and a pile position the observer bottomed keeps both its rank and its position.
    /// - The observer's own hand, both discards, every pile length, hand sizes, and every
    ///   global flag.
    /// - The removed pool, in the mirrored variant only (§9b reveals it at setup).
    ///
    /// # What is resampled
    ///
    /// Face-down cards the observer cannot read — including **their own base cards**, which
    /// are hidden from their owner too (§3) — the opponent's hand, unknown pile positions,
    /// and, outside §9b, the removed-unseen pool.
    ///
    /// # What this guarantees
    ///
    /// The sampled state is in the same information set as `self` for `observer`. Two
    /// consequences the search agents lean on:
    ///
    /// - **The legal action set at this decision node is unchanged.** Every legality
    ///   predicate reads face-up ranks (public), slot positions (public), the acting
    ///   player's own hand (known to them), or a global flag — never a hidden rank. So an
    ///   agent may enumerate actions on the real state and evaluate them on sampled worlds.
    ///   `rule_6_determinization_preserves_the_legal_action_set` pins this.
    /// - **An agent that only reads sampled worlds is verifiably not cheating**, because
    ///   feeding it `self` and feeding it `self.determinize(observer, ..)` must produce the
    ///   same decision. That is the test, not a comment.
    ///
    /// # Known limitation, base variant only
    ///
    /// `game_rules.md` §10a lets a player bottom a card into the pile they draw from. In
    /// the base game that pile is *shared*, so the opponent may draw a card whose rank you
    /// know — and `GameState::draw_one` drops the mask when a card enters a hand, because a
    /// hand is modelled as a multiset of ranks. The sampler therefore treats the opponent's
    /// hand as fully unknown in the base variant even in the rare case where one card of it
    /// is not. This makes a base-variant agent very slightly *weaker* than the information
    /// it could in principle hold; it never makes one illegal, and it cannot arise in the
    /// split variants (the project default), where you only ever draw from your own pile.
    ///
    /// # Hand-built positions
    ///
    /// A position from [`crate::testkit`] is not a dealt game — it holds whatever cards the
    /// test asked for, and the deck arithmetic need not balance. `DESIGN.md` §7 and
    /// `PLAN.md` Phase 4 both want search agents runnable on exactly those, so the sampler
    /// stays usable there: any hidden position the deck cannot account for is filled with a
    /// uniformly random rank instead. The strict balance check applies only to real deals,
    /// where a mismatch would mean the engine had lost or duplicated a card.
    ///
    /// # Panics
    ///
    /// In debug builds, if the deck arithmetic fails to balance on a position that came from
    /// a real deal.
    pub fn determinize(&self, observer: Player, rng: &mut Rng) -> GameState {
        let mut out = self.clone();
        // A dealt game accounts for every card; a hand-built probe position does not.
        let dealt = self.card_census() == self.expected_card_count();

        for pool in 0..self.pool_count() {
            let (mut bag, slots) = self.hidden_pool(observer, pool, dealt);
            debug_assert!(
                !dealt || bag.len() == slots.len(),
                "pool {pool}: {} unplaced cards for {} hidden positions — the deck \
                 arithmetic does not balance",
                bag.len(),
                slots.len()
            );
            rng.shuffle(&mut bag);
            for (i, &slot) in slots.iter().enumerate() {
                let rank = match bag.get(i) {
                    Some(&rank) => rank,
                    // Only reachable on a hand-built position; see above.
                    None => Rank::from_index(rng.index(self.config.rank_count())),
                };
                out.place_hidden(slot, rank);
            }
        }

        // `legal.rs` collapses duplicate ranks by scanning adjacent entries, so a hand that
        // is not sorted would silently enumerate the same `Play` twice.
        for p in Player::BOTH {
            out.hands[p.idx()].sort_unstable();
        }

        out.debug_check_invariants();
        out
    }

    /// The multiset of ranks `observer` **cannot place** — the belief feature of
    /// `DESIGN.md` §5.
    ///
    /// A rank is counted once for every copy that could be sitting in a face-down card the
    /// observer cannot read, the opponent's hand, an unknown pile position, or the
    /// removed-unseen pool. It is exactly the bag [`GameState::determinize`] deals from, so
    /// the encoder and the sampler can never disagree about what is unknown — and it is a
    /// function of the observer's information set for the same reason determinization is.
    ///
    /// `game_rules.md` §2: these counts never reach zero uncertainty, because the ten cards
    /// removed at setup stay indistinguishable from cards in a hand or a base slot. §9b is
    /// the exception: there the removed multiset is public, so it is subtracted out.
    pub fn unseen_counts(&self, observer: Player) -> RankCounts {
        let dealt = self.card_census() == self.expected_card_count();
        let mut counts: RankCounts = [0; Rank::COUNT];
        for pool in 0..self.pool_count() {
            let (bag, _) = self.hidden_pool(observer, pool, dealt);
            for rank in bag {
                counts[rank.index()] += 1;
            }
        }
        counts
    }

    /// How many independent deck pools this variant has: one shared deck, or one per player.
    #[inline]
    fn pool_count(&self) -> usize {
        if self.config.variant.is_split() {
            2
        } else {
            1
        }
    }

    /// The unplaced ranks of one pool, and the positions they have to fill.
    ///
    /// Runs the deck composition down by every card the observer can place, and collects a
    /// slot for every card they cannot. Because every card is in exactly one of *play*,
    /// *a hand*, *a pile*, *a discard* or *the removed pool*, the two sides balance by
    /// construction — which is what the caller's assertion checks.
    fn hidden_pool(
        &self,
        observer: Player,
        pool: usize,
        dealt: bool,
    ) -> (Vec<Rank>, Vec<HiddenSlot>) {
        let split = self.config.variant.is_split();
        let members: &[Player] = if split {
            match pool {
                0 => &[Player::P0],
                _ => &[Player::P1],
            }
        } else {
            &[Player::P0, Player::P1]
        };
        let copies = if split {
            self.config.copies_per_rank_per_player()
        } else {
            self.config.copies_per_rank
        };

        let mut counts = vec![copies as i32; self.config.rank_count()];
        let mut slots: Vec<HiddenSlot> = Vec::new();

        // --- in play ---------------------------------------------------------------------
        for lane in &self.lanes {
            for &p in members {
                for card in lane.side(p) {
                    if card.rank_known_to(observer) {
                        counts[card.rank.index()] -= 1;
                    } else {
                        slots.push(HiddenSlot::InPlay(card.id));
                    }
                }
            }
        }

        // --- hands -----------------------------------------------------------------------
        // Yours is known exactly; theirs is a size and nothing more.
        for &p in members {
            if p == observer {
                for r in &self.hands[p.idx()] {
                    counts[r.index()] -= 1;
                }
            } else {
                for i in 0..self.hands[p.idx()].len() {
                    slots.push(HiddenSlot::Hand(p, i));
                }
            }
        }

        // --- discards --------------------------------------------------------------------
        // `game_rules.md` §5: "The discard pile is public and inspectable."
        for &p in members {
            for r in &self.discards[p.idx()] {
                counts[r.index()] -= 1;
            }
        }

        // --- the draw pile ---------------------------------------------------------------
        // Pool index doubles as pile index: the split variants give player `k` pile `k`,
        // and the base variant has its single shared pile at index 0 with pool count 1.
        for (i, (rank, mask)) in self.piles[pool].entries().enumerate() {
            if mask & observer.bit() != 0 {
                counts[rank.index()] -= 1;
            } else {
                slots.push(HiddenSlot::Pile(pool, i));
            }
        }

        // --- removed unseen ---------------------------------------------------------------
        // §9b reveals the removed multiset at setup; §2 and §9a never do, and
        // `game_rules.md` §2 is explicit that this is why "a player's belief over hidden
        // cards never fully resolves, even at the end of the game."
        if self.removed_revealed {
            for r in &self.removed[pool] {
                counts[r.index()] -= 1;
            }
        } else {
            for i in 0..self.removed[pool].len() {
                slots.push(HiddenSlot::Removed(pool, i));
            }
        }

        let mut bag = Vec::with_capacity(slots.len());
        for (i, &n) in counts.iter().enumerate() {
            debug_assert!(
                n >= 0 || !dealt,
                "pool {pool}: the observer can place {} cards of rank {} but the deck only \
                 holds {copies}",
                copies as i32 - n,
                Rank::from_index(i),
            );
            for _ in 0..n.max(0) {
                bag.push(Rank::from_index(i));
            }
        }
        (bag, slots)
    }

    /// Write one sampled rank into its position.
    fn place_hidden(&mut self, slot: HiddenSlot, rank: Rank) {
        match slot {
            HiddenSlot::InPlay(id) => {
                if let Some(card) = self.card_mut(id) {
                    card.rank = rank;
                }
            }
            HiddenSlot::Hand(p, i) => self.hands[p.idx()][i] = rank,
            HiddenSlot::Pile(pile, pos) => self.piles[pile].set_rank(pos, rank),
            HiddenSlot::Removed(pool, pos) => self.removed[pool][pos] = rank,
        }
    }
}
