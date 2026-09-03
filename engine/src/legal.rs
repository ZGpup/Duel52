//! Legal-action enumeration.
//!
//! `CLAUDE.md`: "The Rust engine is the sole authority on legality." Every agent, the CLI,
//! and the Python bindings ask this module what is possible; nothing reimplements the
//! checks.
//!
//! Enumeration is written against the query helpers in `state.rs` (`legal_attack_targets`,
//! `twinstrike_split_candidates`, `queen_move_sources`, `face_down_cards`), which are the
//! same helpers `apply.rs` uses when it resolves an action — so legality and resolution
//! cannot drift apart.

use crate::action::{Action, Phase, Side};
use crate::card::CardId;
use crate::player::Player;
use crate::rank::Rank;
use crate::state::{GameState, Pending, ResolveKind};

impl GameState {
    /// Every action the acting player may take right now.
    ///
    /// Empty exactly when the game is over. In [`Phase::Main`] the list always contains at
    /// least [`Action::Pass`], and in every sub-decision phase it is non-empty because
    /// `apply.rs` only pushes a sub-decision that has an answer and prunes one that has
    /// gone stale (see `normalize_pending`).
    pub fn legal_actions(&self) -> Vec<Action> {
        if self.outcome.is_over() {
            return Vec::new();
        }
        match self.pending.last() {
            None => self.legal_main_actions(),
            Some(Pending::Foresight { .. }) => self.legal_peeks(),
            Some(Pending::ResolveOrder { remaining, .. }) => self.legal_resolutions(remaining),
            Some(Pending::QueenSource { player, lane }) => self.legal_queen_moves(*player, *lane),
            Some(Pending::GiveBack { player }) => self.legal_give_backs(*player),
            Some(Pending::SplitTarget { lane, primary, .. }) => {
                self.legal_split_targets(*lane, *primary)
            }
        }
    }

    /// Is `action` currently legal?
    ///
    /// Implemented as membership in [`GameState::legal_actions`] rather than as a second
    /// set of predicates. Duplicating the logic is how legality and resolution end up
    /// disagreeing about an edge case.
    pub fn is_legal(&self, action: Action) -> bool {
        self.legal_actions().contains(&action)
    }

    // ------------------------------------------------------------------------ main phase --

    /// The four actions of `game_rules.md` §4, plus `Pass`.
    ///
    /// "Any combination of actions is allowed, including repeats", and "A card may be
    /// played, flipped, and attack all in the same turn, actions permitting" — so there is
    /// no per-turn sequencing constraint to encode here. Everything below is a property of
    /// the position alone.
    fn legal_main_actions(&self) -> Vec<Action> {
        let me = self.to_move;
        let opponent = me.other();
        let mut out = Vec::with_capacity(64);

        // --- PLAY: a card from hand, face-down, into one of your own lanes. -------------
        // The hand is a multiset of ranks, so identical ranks collapse to one action.
        let mut last: Option<Rank> = None;
        for &rank in self.hand(me) {
            if last == Some(rank) {
                continue; // hands are kept sorted, so duplicates are adjacent
            }
            last = Some(rank);
            for lane in 0..self.lane_count() {
                out.push(Action::Play {
                    rank,
                    lane: lane as u8,
                });
            }
        }

        // --- FLIP: turn one of your own face-down cards face-up. ------------------------
        for (lane, slot, card) in self.cards_of(me) {
            if card.face_up {
                continue;
            }
            // `game_rules.md` §3: "While the draw pile is non-empty, base cards cannot be
            // attacked and cannot be flipped. They are untouchable."
            if card.is_base && !self.base_unlocked {
                continue;
            }
            // §8: "Freeze blocks exactly two things: attacking, and being flipped — by
            // anyone."
            if card.is_frozen(self.ply) {
                continue;
            }
            out.push(Action::Flip {
                lane: lane as u8,
                slot: slot as u8,
            });
        }

        // --- ATTACK: one of your face-up cards hits an opposing card in its lane. -------
        for lane in 0..self.lane_count() {
            let targets = self.legal_attack_targets(lane, opponent);
            if targets.is_empty() {
                continue;
            }
            for slot in 0..self.lanes[lane].side(me).len() {
                if !self.can_attack_from(lane, me, slot) {
                    continue;
                }
                for &target in &targets {
                    out.push(Action::Attack {
                        lane: lane as u8,
                        attacker: slot as u8,
                        target: target as u8,
                    });
                }
            }
        }

        // --- PAIR: declare a pair of face-up same-rank cards in one lane. ---------------
        //
        // §5: "A pair is two face-up cards of matching rank that you control in the same
        // lane. Both members must be face-up." A card belongs to at most one pair and
        // "cannot leave one pair to join another", so both candidates must be unpaired.
        //
        // Freeze is not a bar: it blocks attacking and being flipped, and declaring a pair
        // is neither. **[ASSUMED]**
        for lane in 0..self.lane_count() {
            let side = self.lanes[lane].side(me);
            for a in 0..side.len() {
                if !side[a].face_up || side[a].pair_id.is_some() {
                    continue;
                }
                for b in (a + 1)..side.len() {
                    if !side[b].face_up || side[b].pair_id.is_some() {
                        continue;
                    }
                    if side[a].rank != side[b].rank {
                        continue;
                    }
                    out.push(Action::DeclarePair {
                        lane: lane as u8,
                        slot_a: a as u8,
                        slot_b: b as u8,
                    });
                }
            }
        }

        // --- PASS -----------------------------------------------------------------------
        // Always available. A player can be left with nothing else — no cards in hand, no
        // face-up cards, nothing legal to attack — and the turn still has to end.
        out.push(Action::Pass);
        out
    }

    // -------------------------------------------------------------------- sub-decisions --

    /// The 4's Foresight: any face-down card on the board (`game_rules.md` §6).
    fn legal_peeks(&self) -> Vec<Action> {
        let me = self.to_move;
        self.face_down_cards()
            .into_iter()
            .map(|(lane, side_idx, slot)| Action::Peek {
                side: if Player::from_index(side_idx) == me {
                    Side::Mine
                } else {
                    Side::Theirs
                },
                lane: lane as u8,
                slot: slot as u8,
            })
            .collect()
    }

    /// The next card to resolve out of a 5's flip list or a King's reactivation list.
    ///
    /// `remaining` has already been pruned of stale entries by `normalize_pending`, so
    /// every id in it still names a card that can be resolved.
    fn legal_resolutions(&self, remaining: &[CardId]) -> Vec<Action> {
        remaining
            .iter()
            .filter_map(|&id| {
                let (lane, _, slot) = self.locate(id)?;
                Some(Action::ResolveNext {
                    lane: lane as u8,
                    slot: slot as u8,
                })
            })
            .collect()
    }

    /// The Queen's Move: an allied card in another lane (`game_rules.md` §6).
    fn legal_queen_moves(&self, player: Player, queen_lane: u8) -> Vec<Action> {
        self.queen_move_sources(player, queen_lane as usize)
            .into_iter()
            .map(|(lane, slot)| Action::MoveHere {
                lane: lane as u8,
                slot: slot as u8,
            })
            .collect()
    }

    /// The 2's View: which card to give back (`game_rules.md` §10a).
    ///
    /// "You may bottom the card you just drew" — no filter, so this is simply every
    /// distinct rank in hand. The draw has already happened by the time this node exists,
    /// so the hand is never empty here.
    fn legal_give_backs(&self, player: Player) -> Vec<Action> {
        let mut out = Vec::new();
        let mut last: Option<Rank> = None;
        for &rank in self.hand(player) {
            if last == Some(rank) {
                continue;
            }
            last = Some(rank);
            out.push(Action::GiveBack { rank });
        }
        out
    }

    /// The second half of a 10's twinstrike (`game_rules.md` §6).
    fn legal_split_targets(&self, lane: u8, primary: CardId) -> Vec<Action> {
        let defender = self.to_move.other();
        let Some((_, _, primary_slot)) = self.locate(primary) else {
            return Vec::new();
        };
        self.twinstrike_split_candidates(lane as usize, defender, primary_slot)
            .into_iter()
            .map(|slot| Action::SplitTarget { slot: slot as u8 })
            .collect()
    }

    // ------------------------------------------------------------------------ diagnostics --

    /// Human-readable description of the current decision, for the CLI.
    pub fn prompt(&self) -> String {
        match self.pending.last() {
            None => format!(
                "{} — {} action(s) remaining",
                self.to_move, self.actions_remaining
            ),
            Some(Pending::ResolveOrder { kind, lane, remaining, .. }) => format!(
                "{} — {} in lane {}: {} card(s) left to resolve",
                self.to_move,
                kind.label(),
                lane,
                remaining.len()
            ),
            Some(other) => format!("{} — {}", self.to_move, other.phase()),
        }
    }

    /// True when the current decision is a free sub-choice rather than one of the turn's
    /// actions.
    pub fn in_sub_decision(&self) -> bool {
        !matches!(self.phase(), Phase::Main | Phase::Terminal)
    }

    /// Prune sub-decisions that no longer have an answer.
    ///
    /// Only a [`Pending::ResolveOrder`] list can go stale, and only because *other*
    /// resolutions run in between its choices: a Queen fired earlier in a 5's cascade can
    /// move a queued card out of the lane, and a King's reactivation list can outlive the
    /// cards on it. Every other sub-decision is answered immediately after it is pushed,
    /// with nothing able to intervene.
    ///
    /// Entries are dropped rather than forcing a choice, because `game_rules.md` §8 is
    /// explicit that a power with no legal target simply fizzles.
    pub(crate) fn normalize_pending(&mut self) {
        loop {
            let Some(top) = self.pending.last() else { return };
            let (kind, player, lane, remaining) = match top {
                Pending::ResolveOrder {
                    kind,
                    player,
                    lane,
                    remaining,
                } => (*kind, *player, *lane, remaining.clone()),
                _ => return,
            };

            let still_valid: Vec<CardId> = remaining
                .into_iter()
                .filter(|&id| self.resolution_still_valid(kind, player, lane, id))
                .collect();

            if still_valid.is_empty() {
                self.pending.pop();
                // Popping may expose another stale ResolveOrder underneath.
                continue;
            }

            if let Some(Pending::ResolveOrder { remaining, .. }) = self.pending.last_mut() {
                *remaining = still_valid;
            }
            return;
        }
    }

    /// Can the queued card still be resolved?
    fn resolution_still_valid(
        &self,
        kind: ResolveKind,
        player: Player,
        lane: u8,
        id: CardId,
    ) -> bool {
        let Some((card_lane, side, slot)) = self.locate(id) else {
            return false; // left play
        };
        if card_lane != lane as usize || Player::from_index(side) != player {
            return false; // a Queen relocated it, so the 5/King no longer reaches it
        }
        let card = &self.lanes[card_lane].sides[side][slot];
        match kind {
            // §6: the 5 flips "all your face-down cards in its lane". §8: it "simply skips"
            // frozen cards — "they are untouchable, not merely passive".
            ResolveKind::FiveFlip => {
                !card.face_up
                    && !card.is_frozen(self.ply)
                    && (self.base_unlocked || !card.is_base)
            }
            // §6: a King reactivates "all your face-up cards in this lane", excluding other
            // Kings and constant powers. §8: freeze does *not* block reactivation.
            ResolveKind::KingEmpower => card.face_up && card.rank.is_king_reactivatable(),
        }
    }
}
