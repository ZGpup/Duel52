//! Actions and decision phases.
//!
//! # The central encoding decision
//!
//! `DESIGN.md` §4: each of the three per-turn actions is its own decision node, and the
//! sub-choices a power opens are **separate decision nodes that cost no action**. That is
//! what keeps the branching factor in the tens instead of the thousands.
//!
//! So "flip a 5 in lane 1, which flips four cards, one of which is a King, which refires
//! three more powers" is not one enormous action — it is one [`Action::Flip`] followed by a
//! run of [`Action::ResolveNext`] choices, each made *after* seeing the previous power land
//! (`game_rules.md` §8: resolution order is adaptive).
//!
//! # Slots, not ranks
//!
//! The neural-network policy head in `DESIGN.md` §4 keys some actions by rank (`FLIP(rank,
//! lane)`, `PAIR(lane, rank)`) because same-rank cards are usually interchangeable to the
//! player choosing. The **engine** keys everything by slot, because same-rank cards are
//! *not* actually interchangeable — they can differ in damage, freeze, and attacks used.
//! Phase 3's encoder maps between the two; the engine stays exact.

use crate::rank::Rank;
use std::fmt;

/// Which side of a lane a slot index refers to. Used only by actions that can target
/// either side of the board — currently just the 4's Foresight, which may peek at your own
/// face-down cards as well as the opponent's (`game_rules.md` §6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    /// The side belonging to the player who is acting.
    Mine,
    /// The opponent's side.
    Theirs,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Mine => f.write_str("mine"),
            Side::Theirs => f.write_str("theirs"),
        }
    }
}

/// A single decision the player to move can make.
///
/// The first four variants are the *actions* of `game_rules.md` §4 and cost one action
/// each. There is no pass and no end-turn: §4 makes acting mandatory, and a turn with no
/// legal action left in it is ended by the engine rather than offered as a choice (see
/// `apply.rs`'s `settle`). Everything after them is a **sub-decision** opened by a power:
/// free, mandatory, and resolved before the turn continues.
///
/// So **every variant here is a real decision**. Nothing in this enum exists only to move
/// the game along, which is what lets a policy head treat the action space as choices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    // ---------------------------------------------------------------- costs one action --
    /// Put a card from hand **face-down** into one of your own lanes. It is inactive: it
    /// can be attacked and killed, but cannot attack and has no power (§4).
    Play { rank: Rank, lane: u8 },

    /// Turn one of your own face-down cards face-up. Its power fires immediately (one-shot)
    /// or goes live (constant). The power is **mandatory** — you do not get to decline the
    /// 2's scry or the 5's flips (§8).
    Flip { lane: u8, slot: u8 },

    /// One of your face-up cards attacks an opposing card in the same lane.
    ///
    /// If the attacker is **paired**, this is automatically a *pair attack*: one action, 2
    /// damage, both members spend their attack for the turn (§5). There is no separate
    /// "attack with a pair" action, because a paired card cannot attack alone.
    Attack { lane: u8, attacker: u8, target: u8 },

    /// Declare a pair from two face-up same-rank cards you control in one lane (§5).
    ///
    /// Both slots are given explicitly rather than just the rank: with three same-rank
    /// cards in a lane, *which two* you pair matters, because they can carry different
    /// damage and different attack budgets.
    DeclarePair { lane: u8, slot_a: u8, slot_b: u8 },

    // ------------------------------------------------------- free, mandatory sub-choices --
    /// The 4's Foresight: privately look at one face-down card anywhere on the board,
    /// including base cards, yours or the opponent's (§6).
    Peek { side: Side, lane: u8, slot: u8 },

    /// Choose the next card to resolve out of a 5's flip list or a King's reactivation list.
    ///
    /// One choice per card, made after seeing the previous one land — *adaptive* ordering
    /// (§8), which is both the correct information model and linear rather than factorial
    /// in the number of cards.
    ResolveNext { lane: u8, slot: u8 },

    /// The Queen's Move: pick the allied card, in some **other** lane, to bring into the
    /// Queen's lane (§6).
    MoveHere { lane: u8, slot: u8 },

    /// The 2's View: pick the rank to put on the bottom of your draw pile (house rule) or
    /// to discard (`two_power = discard`). See `game_rules.md` §10a.
    GiveBack { rank: Rank },

    /// The second target of a 10's twinstrike (§6).
    ///
    /// The engine asks for both targets *before* applying any damage, so that the two
    /// halves of the split land simultaneously and retaliate resolution has no ambiguous
    /// ordering.
    SplitTarget { slot: u8 },

    // ------------------------------------------------------- the encoder reserve (§7) --
    /// Pick a **lane**, on either side of the board.
    ///
    /// `MODULAR_RULES.md` §7, reserve item 2. Nothing in the canonical ruleset opens this;
    /// it exists so that a power whose target is a lane rather than a card — "empower a lane
    /// of your choice", "freeze a lane", "all cards in one enemy lane take 1" — is a Tier 2
    /// change rather than a layout break.
    ///
    /// `side` is carried even though a given power usually wants only one of the two,
    /// because the **legality mask is what restricts it**: a power that picks one of your own
    /// lanes emits only `Side::Mine` and the other three logits are never legal, and a power
    /// that picks an enemy lane emits only `Side::Theirs`. One block covers both, which is
    /// why it is `2·L` wide rather than `L`.
    ChooseLane { side: Side, lane: u8 },

    /// Pick one of up to [`crate::encode::OPTION_COUNT`] unnamed, power-defined options.
    ///
    /// `MODULAR_RULES.md` §7, reserve item 2. The meaning of an option index belongs
    /// entirely to the power that opened the node — the encoder deliberately knows nothing
    /// about it, which is what makes one block serve every modal power. A power that offers
    /// two choices makes two of the four legal and the rest are masked off.
    ChooseOption { option: u8 },
}

impl Action {
    /// True for the five §4 actions, which consume one of the turn's actions. False for the
    /// free sub-choices a power opens.
    #[inline]
    pub fn costs_an_action(&self) -> bool {
        matches!(
            self,
            Action::Play { .. }
                | Action::Flip { .. }
                | Action::Attack { .. }
                | Action::DeclarePair { .. }
        )
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Play { rank, lane } => write!(f, "play {rank} face-down in lane {lane}"),
            Action::Flip { lane, slot } => write!(f, "flip lane {lane} slot {slot}"),
            Action::Attack {
                lane,
                attacker,
                target,
            } => write!(
                f,
                "attack: lane {lane} slot {attacker} -> enemy slot {target}"
            ),
            Action::DeclarePair {
                lane,
                slot_a,
                slot_b,
            } => write!(f, "pair lane {lane} slots {slot_a} + {slot_b}"),
            Action::Peek { side, lane, slot } => {
                write!(f, "peek at {side} lane {lane} slot {slot}")
            }
            Action::ResolveNext { lane, slot } => {
                write!(f, "resolve next: lane {lane} slot {slot}")
            }
            Action::MoveHere { lane, slot } => {
                write!(f, "move lane {lane} slot {slot} into the Queen's lane")
            }
            Action::GiveBack { rank } => write!(f, "give back {rank} from hand"),
            Action::SplitTarget { slot } => write!(f, "twinstrike second target: enemy slot {slot}"),
            Action::ChooseLane { side, lane } => write!(f, "choose {side} lane {lane}"),
            Action::ChooseOption { option } => write!(f, "choose option {option}"),
        }
    }
}

/// Which kind of decision is on the table right now.
///
/// The observation carries this so the policy head knows which legality mask applies
/// (`DESIGN.md` §4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Normal turn: spend an action.
    Main,
    /// A 4 is asking which face-down card to look at.
    Foresight,
    /// A 5 or a King is asking which queued card to resolve next.
    ResolveOrder,
    /// A Queen is asking which allied card to pull into her lane.
    QueenSource,
    /// A 2 is asking which card to give back.
    GiveBack,
    /// A 10 is asking for the second half of its twinstrike.
    SplitTarget,
    /// The game is over.
    Terminal,
    /// A power is asking which lane to act on (`MODULAR_RULES.md` §7, reserve item 1+2).
    ChooseLane,
    /// A power is asking which of its options to take (§7, reserve item 1+2).
    ChooseOption,
}

impl Phase {
    /// Every phase, in [`crate::encode::phase_index`] order. The index of a phase in this
    /// list **is** its one-hot position, so a phase added in the middle would move every
    /// later one — append, never insert.
    pub const ALL: &'static [Phase] = &[
        Phase::Main,
        Phase::Foresight,
        Phase::ResolveOrder,
        Phase::QueenSource,
        Phase::GiveBack,
        Phase::SplitTarget,
        Phase::Terminal,
        Phase::ChooseLane,
        Phase::ChooseOption,
    ];

    /// True for the phases only an extended-encoder ruleset can reach.
    ///
    /// `MODULAR_RULES.md` §7: the base layout's `phase_onehot` is exactly seven wide, so a
    /// ruleset that can enter one of these needs the wider one. [`crate::powers::PowerId`]
    /// is where that is declared, and `engine/tests/reserve.rs` checks the two agree.
    pub const fn needs_extended_encoder(self) -> bool {
        matches!(self, Phase::ChooseLane | Phase::ChooseOption)
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Phase::Main => "main",
            Phase::Foresight => "foresight (4): choose a face-down card to look at",
            Phase::ResolveOrder => "resolve order: choose which card resolves next",
            Phase::QueenSource => "queen (Q): choose an allied card to move here",
            Phase::GiveBack => "view (2): choose a card to give back",
            Phase::SplitTarget => "twinstrike (10): choose the second target",
            Phase::Terminal => "terminal",
            Phase::ChooseLane => "choose a lane",
            Phase::ChooseOption => "choose an option",
        };
        f.write_str(s)
    }
}

/// Why an action was rejected. Every rejection names the rule it violates, so a wrong CLI
/// input or a buggy agent produces a diagnosable message rather than a panic.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IllegalAction {
    pub action: Action,
    pub reason: String,
}

impl fmt::Display for IllegalAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "illegal action ({}): {}", self.action, self.reason)
    }
}

impl std::error::Error for IllegalAction {}
