//! Building positions by hand.
//!
//! Two consumers:
//!
//! - **The rules tests.** `CLAUDE.md` requires "every ruling in `game_rules.md` gets a
//!   named test". Most rulings are about a specific interaction — a 9-pair against a Jack,
//!   a 5 meeting a frozen card — and reaching those by dealing and playing would be
//!   hopeless. So tests place the cards directly.
//! - **Phase 4.** `PLAN.md` lists "probe the value net on hand-constructed positions" as a
//!   deliverable, which needs exactly this.
//!
//! This module can build positions that could never arise in a real game. That is the
//! point, but it also means it bypasses every rule about *how* cards get where they are.
//! Nothing outside tests and analysis should use it, and nothing here reimplements a rule:
//! it sets fields, and the engine still decides what is legal from there.
//!
//! # Defaults, and the two that bite
//!
//! [`Position::new`] gives you an empty board, empty hands, empty discards, `ply` 0, three
//! actions, and **`P0` to move**. Two defaults are chosen rather than obvious:
//!
//! 1. **Each draw pile holds three filler cards, so base cards start locked.** The engine
//!    latches `base_unlocked` the moment every pile is empty (`game_rules.md` §3), and an
//!    empty-piled position would unlock itself on the first action. Call
//!    [`Position::unlock`] to empty the piles and unlock deliberately.
//! 2. **Hands start empty**, which is convenient for asserting an exact legal-action list —
//!    but note that with `base_unlocked` set and a hand empty, an empty lane side is
//!    immediately a won lane (§7). Give a player a card in hand if a test needs the game to
//!    stay alive.

use crate::card::{Card, CardId};
use crate::config::GameConfig;
use crate::player::Player;
use crate::rank::Rank;
use crate::state::{empty_state, GameState, Pile};

/// Filler rank for the piles that keep base cards locked. Arbitrary; a test that cares
/// what gets drawn should set the pile itself with [`Position::pile`].
const FILLER: Rank = Rank::KING;

/// A position under construction.
pub struct Position {
    state: GameState,
}

impl Position {
    /// An empty position with base cards locked and `P0` to move.
    pub fn new(config: GameConfig) -> Position {
        let mut state = empty_state(config, 0);
        // Non-empty piles keep `base_unlocked` false. See the module docs.
        for i in 0..2 {
            state.piles[i] = Pile::from_ranks(vec![FILLER; 3]);
        }
        state.actions_remaining = config.actions_per_turn;
        Position { state }
    }

    /// An empty position in the project's default configuration (split deck).
    pub fn empty() -> Position {
        Position::new(GameConfig::default())
    }

    // -------------------------------------------------------------- placing cards --

    fn push(&mut self, lane: usize, owner: Player, card: Card) -> usize {
        let side = self.state.lanes[lane].side_mut(owner);
        side.push(card);
        side.len() - 1
    }

    fn next_id(&mut self) -> CardId {
        self.state.fresh_card_id()
    }

    /// Place a face-up card. Returns its slot index.
    pub fn face_up(&mut self, lane: usize, owner: Player, rank: Rank) -> usize {
        let id = self.next_id();
        let mut card = Card::played_from_hand(id, rank, owner);
        card.face_up = true;
        // A face-up card's rank is public (`game_rules.md` §5).
        card.known_to = 0b11;
        self.push(lane, owner, card)
    }

    /// Place a face-down card that was played from its owner's hand — so its owner knows
    /// what it is and the opponent does not (§4). Returns its slot index.
    pub fn face_down(&mut self, lane: usize, owner: Player, rank: Rank) -> usize {
        let id = self.next_id();
        let card = Card::played_from_hand(id, rank, owner);
        self.push(lane, owner, card)
    }

    /// Place a face-down **base** card: hidden from both players, including its owner
    /// (§3), and untouchable until every pile is empty. Returns its slot index.
    pub fn base(&mut self, lane: usize, owner: Player, rank: Rank) -> usize {
        let id = self.next_id();
        let card = Card::base_card(id, rank, owner);
        self.push(lane, owner, card)
    }

    // ---------------------------------------------------------- tweaking a card --

    /// Set a card's damage. Panics if that would already have killed it, because a dead
    /// card in play is exactly the invariant the engine asserts against.
    ///
    /// Note the ceiling depends on face-up state: a face-down Jack is a blank 2-HP card
    /// (`game_rules.md` §5), so 2 damage kills it and only a *face-up* Jack can sit on 2.
    pub fn damage(&mut self, lane: usize, owner: Player, slot: usize, damage: u8) -> &mut Self {
        let card = &mut self.state.lanes[lane].side_mut(owner)[slot];
        assert!(
            damage < card.max_hp(),
            "{damage} damage would kill a {} {} ({} HP)",
            if card.face_up { "face-up" } else { "face-down" },
            card.rank,
            card.max_hp()
        );
        card.damage = damage;
        self
    }

    /// Declare a pair between two slots, exactly as the `Pair` action would.
    pub fn pair(&mut self, lane: usize, owner: Player, slot_a: usize, slot_b: usize) -> &mut Self {
        let pid = self.state.fresh_pair_id();
        let side = self.state.lanes[lane].side_mut(owner);
        assert!(
            side[slot_a].face_up && side[slot_b].face_up,
            "both members of a pair must be face-up (game_rules.md §5)"
        );
        assert_eq!(
            side[slot_a].rank, side[slot_b].rank,
            "a pair needs matching ranks (game_rules.md §5)"
        );
        side[slot_a].pair_id = Some(pid);
        side[slot_b].pair_id = Some(pid);
        self
    }

    /// Freeze a card through the end of `until_ply`, as a 6 would.
    pub fn freeze(&mut self, lane: usize, owner: Player, slot: usize, until_ply: u32) -> &mut Self {
        self.state.lanes[lane].side_mut(owner)[slot].frozen_until_ply = Some(until_ply);
        self
    }

    /// Mark a card as having already attacked this turn.
    pub fn spent(&mut self, lane: usize, owner: Player, slot: usize) -> &mut Self {
        let card = &mut self.state.lanes[lane].side_mut(owner)[slot];
        card.attacks_used = card.attack_allowance;
        self
    }

    // ------------------------------------------------------------------ resources --

    /// Replace a hand.
    pub fn hand(&mut self, player: Player, ranks: &[Rank]) -> &mut Self {
        let mut hand = ranks.to_vec();
        hand.sort_unstable();
        self.state.hands[player.idx()] = hand;
        self
    }

    /// Replace a draw pile, top card first.
    pub fn pile(&mut self, player: Player, ranks_top_first: &[Rank]) -> &mut Self {
        let index = self.state.pile_index(player);
        self.state.piles[index] = Pile::from_ranks(ranks_top_first.to_vec());
        self
    }

    /// Replace a discard pile.
    pub fn discard(&mut self, player: Player, ranks: &[Rank]) -> &mut Self {
        self.state.discards[player.idx()] = ranks.to_vec();
        self
    }

    /// Empty every pile and unlock base cards, as if the draw phase were over
    /// (`game_rules.md` §3).
    pub fn unlock(&mut self) -> &mut Self {
        for i in 0..2 {
            self.state.piles[i] = Pile::from_ranks(Vec::new());
        }
        self.state.base_unlocked = true;
        self
    }

    // ------------------------------------------------------------------- turn state --

    pub fn to_move(&mut self, player: Player) -> &mut Self {
        self.state.to_move = player;
        self
    }

    /// Set the ply counter. Matters for freeze, which is expressed as "frozen through the
    /// end of ply N" (§8).
    pub fn ply(&mut self, ply: u32) -> &mut Self {
        self.state.ply = ply;
        self
    }

    pub fn actions(&mut self, actions: u32) -> &mut Self {
        self.state.actions_remaining = actions;
        self
    }

    pub fn quiet_plies(&mut self, plies: u32) -> &mut Self {
        self.state.quiet_plies = plies;
        self
    }

    /// Escape hatch for anything this builder does not cover.
    pub fn state_mut(&mut self) -> &mut GameState {
        &mut self.state
    }

    /// Finish. Checks the engine's own invariants, so a malformed test position fails at
    /// construction rather than as a confusing failure three actions later.
    pub fn build(self) -> GameState {
        self.state.debug_check_invariants();
        self.state
    }
}

// ============================================================== query helpers ==

/// The card at a slot. Panics with a useful message if the slot is gone, which is usually
/// what a failing test wants to hear.
pub fn card_at(state: &GameState, lane: usize, owner: Player, slot: usize) -> &Card {
    state
        .at(lane, owner, slot)
        .unwrap_or_else(|| panic!("lane {lane} {owner} #{slot} is empty"))
}

/// Damage on a slot.
pub fn damage_at(state: &GameState, lane: usize, owner: Player, slot: usize) -> u8 {
    card_at(state, lane, owner, slot).damage
}

/// How many cards are on one side of a lane.
pub fn occupancy(state: &GameState, lane: usize, owner: Player) -> usize {
    state.lanes[lane].side(owner).len()
}

/// The ranks on one side of a lane, in slot order.
pub fn ranks_in(state: &GameState, lane: usize, owner: Player) -> Vec<Rank> {
    state.lanes[lane]
        .side(owner)
        .iter()
        .map(|c| c.rank)
        .collect()
}

/// The ranks in a discard pile, sorted.
pub fn discard_ranks(state: &GameState, owner: Player) -> Vec<Rank> {
    let mut ranks = state.discards[owner.idx()].clone();
    ranks.sort_unstable();
    ranks
}
