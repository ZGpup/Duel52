//! Applying actions: powers, combat, and the turn machinery.
//!
//! # How a turn actually runs
//!
//! ```text
//! apply(action)
//!   ├─ validate legality
//!   ├─ spend one action, if this action costs one
//!   ├─ dispatch  ──►  a power may push sub-decisions onto `pending`
//!   └─ settle()
//!        ├─ prune stale sub-decisions
//!        ├─ if `pending` is non-empty: stop. The action has not finished resolving,
//!        │   and `game_rules.md` §7 forbids running the terminal check mid-resolution.
//!        ├─ latch `base_unlocked`
//!        ├─ terminal check
//!        └─ if no actions remain: end the turn (quiet-ply accounting, then the next
//!            player's reset + draw + action allowance)
//! ```
//!
//! Everything a power can open — a 4's peek, a 5's flip order, a King's reactivation order,
//! a Queen's move source, a 2's give-back, a 10's second target — is a node on the
//! `pending` **stack**, so a power fired mid-cascade finishes before control returns to the
//! list underneath it. That is what makes "a 5 that flips a King which then re-empowers the
//! lane" (`game_rules.md` §8) come out in the right order.

use crate::action::{Action, IllegalAction, Side};
use crate::card::{Card, CardId};
use crate::config::TwoPower;
use crate::outcome::{DrawReason, Outcome};
use crate::player::Player;
use crate::rank::Rank;
use crate::state::{GameState, Pending, ResolveKind};

/// Bitmask meaning "both players know this card's rank".
const KNOWN_TO_BOTH: u8 = 0b11;

impl GameState {
    // =================================================================== public entry ==

    /// Apply an action, validating it first.
    ///
    /// Use this from anything whose input is not already known-good: the CLI, the Python
    /// bindings, a replay file. Validation is membership in [`GameState::legal_actions`],
    /// so it can never disagree with the legality module.
    pub fn apply(&mut self, action: Action) -> Result<(), IllegalAction> {
        if self.outcome.is_over() {
            return Err(IllegalAction {
                action,
                reason: format!("the game is already over ({})", self.outcome),
            });
        }
        if !self.is_legal(action) {
            return Err(IllegalAction {
                action,
                reason: format!(
                    "not legal in phase `{}` for {}",
                    self.phase(),
                    self.acting_player()
                ),
            });
        }
        self.dispatch(action);
        Ok(())
    }

    /// Apply an action that the caller has already taken from [`GameState::legal_actions`].
    ///
    /// Validation runs in debug builds only, which is what makes the self-play throughput
    /// target in `DESIGN.md` §8 reachable — a full legality re-enumeration per action
    /// roughly doubles the cost of a random game. Anything with an untrusted source must
    /// use [`GameState::apply`] instead.
    ///
    /// # Panics
    /// In debug builds, if the action is not legal.
    pub fn apply_trusted(&mut self, action: Action) {
        debug_assert!(
            !self.outcome.is_over(),
            "apply_trusted called on a finished game"
        );
        debug_assert!(
            self.is_legal(action),
            "apply_trusted called with an illegal action: {action} in phase {}",
            self.phase()
        );
        self.dispatch(action);
    }

    fn dispatch(&mut self, action: Action) {
        // `game_rules.md` §4: play, flip, attack and pair each cost one action. The
        // sub-decisions a power opens are free.
        if action.costs_an_action() {
            debug_assert!(self.actions_remaining > 0, "no actions left to spend");
            self.actions_remaining = self.actions_remaining.saturating_sub(1);
        }

        match action {
            Action::Play { rank, lane } => self.do_play(rank, lane as usize),
            Action::Flip { lane, slot } => self.do_flip(lane as usize, slot as usize),
            Action::Attack {
                lane,
                attacker,
                target,
            } => self.do_attack(lane as usize, attacker as usize, target as usize),
            Action::DeclarePair {
                lane,
                slot_a,
                slot_b,
            } => self.do_declare_pair(lane as usize, slot_a as usize, slot_b as usize),
            Action::Peek { side, lane, slot } => self.do_peek(side, lane as usize, slot as usize),
            Action::ResolveNext { lane, slot } => {
                self.do_resolve_next(lane as usize, slot as usize)
            }
            Action::MoveHere { lane, slot } => self.do_move_here(lane as usize, slot as usize),
            Action::GiveBack { rank } => self.do_give_back(rank),
            Action::SplitTarget { slot } => self.do_split_target(slot as usize),
        }

        self.settle();
        self.debug_check_invariants();
        self.debug_check_playable();
    }

    // ==================================================================== §4 actions ==

    /// **Play** — "Put a card from hand face-down into one of your lanes. It is inactive:
    /// it can be attacked and killed, but cannot attack and has no power." (§4)
    fn do_play(&mut self, rank: Rank, lane: usize) {
        let me = self.to_move;
        let hand = &mut self.hands[me.idx()];
        let pos = hand
            .iter()
            .position(|&r| r == rank)
            .expect("legality guaranteed this rank is in hand");
        hand.remove(pos);

        let id = self.fresh_card_id();
        self.lanes[lane]
            .side_mut(me)
            .push(Card::played_from_hand(id, rank, me));
    }

    /// **Flip** — "Turn one of your face-down cards face-up. Its power activates
    /// immediately (one-shot) or becomes live (constant)." (§4)
    fn do_flip(&mut self, lane: usize, slot: usize) {
        let me = self.to_move;
        let id = self.lanes[lane].side(me)[slot].id;
        self.flip_card(id);
    }

    /// **Pair** — declare a pair (§5). Both members get the same fresh [`PairId`], which is
    /// what makes a pair a matching rather than a group.
    fn do_declare_pair(&mut self, lane: usize, slot_a: usize, slot_b: usize) {
        let me = self.to_move;
        let pid = self.fresh_pair_id();
        let side = self.lanes[lane].side_mut(me);
        side[slot_a].pair_id = Some(pid);
        side[slot_b].pair_id = Some(pid);
    }

    // ======================================================================== combat ==

    /// **Attack** — one of your face-up cards deals damage to an opposing card in its lane.
    ///
    /// Everything rank-specific about the *amount* and the *spread* is decided here:
    ///
    /// - A **paired** attacker attacks together with its partner: one action, base 2 damage,
    ///   and both members spend their attack for the turn (§5).
    /// - A **9** deals double to a Jack — 2 alone, 4 as a pair, which one-shots a 3-HP Jack.
    /// - A **10** twinstrikes. If a split is available the engine asks for the second target
    ///   first and lands both halves together; see [`GameState::do_split_target`]. If the
    ///   split is blocked, a lone 10 deals its plain 1 while a **pair** of 10s consolidates
    ///   to the full 2, because §5 says a 10-pair never loses raw damage.
    fn do_attack(&mut self, lane: usize, attacker_slot: usize, target_slot: usize) {
        let me = self.to_move;
        let opponent = me.other();

        let attackers = self.attack_group(lane, me, attacker_slot);
        let is_pair = attackers.len() == 2;
        let attacker_rank = self.lanes[lane].side(me)[attacker_slot].rank;
        let primary = self.lanes[lane].side(opponent)[target_slot].id;

        if attacker_rank == Rank::TEN {
            let candidates = self.twinstrike_split_candidates(lane, opponent, target_slot);
            if !candidates.is_empty() {
                // Both targets are collected *before* any damage lands, so the two halves
                // of the split are simultaneous and retaliate has no ambiguous ordering.
                self.pending.push(Pending::SplitTarget {
                    player: me,
                    lane: lane as u8,
                    attackers,
                    primary,
                });
                return;
            }
            // Split blocked (a live 9 or a lone Jack) or nothing else in the lane.
            // §5: "Damage is never lost — whenever the split cannot happen ... the full 2
            // lands on that single card." That promise is about the *pair*; a lone 10's
            // second point of damage was the twinstrike bonus, so it goes away with it.
            let damage = if is_pair { 2 } else { 1 };
            self.resolve_attack(attackers, &[(primary, damage)]);
            return;
        }

        let target = &self.lanes[lane].side(opponent)[target_slot];
        let damage = self.attack_damage(attacker_rank, target, is_pair);
        self.resolve_attack(attackers, &[(primary, damage)]);
    }

    /// The second half of a 10's twinstrike: 1 damage to each of the two targets.
    ///
    /// A lone 10 therefore deals 1 + 1 (its bonus is the extra body), and a pair of 10s
    /// splits its 2 as 1 + 1 rather than doubling to 4 — §6: "A pair of 10s twinstrikes:
    /// the pair's 2 damage is split 1 + 1 across two targets, not doubled."
    fn do_split_target(&mut self, slot: usize) {
        let Some(Pending::SplitTarget {
            lane,
            attackers,
            primary,
            ..
        }) = self.pending.pop()
        else {
            unreachable!("do_split_target called outside a SplitTarget node");
        };
        let opponent = self.to_move.other();
        let secondary = self.lanes[lane as usize].side(opponent)[slot].id;
        self.resolve_attack(attackers, &[(primary, 1), (secondary, 1)]);
    }

    /// Land an attack: spend the attackers' budget, apply every hit, then resolve retaliate.
    ///
    /// `game_rules.md` §6 + §8 pin the ordering down: "Retaliate (8) resolves *after* the
    /// attacker's damage is applied, and fires even if that damage killed the 8." So the
    /// set of retaliating 8s is read **before** damage, and paid out after.
    ///
    /// - A pair attacking an 8: **both members take 1** (§5).
    /// - A **9** attacking an 8 takes nothing — Nimble, and "pairing does not forfeit
    ///   Nimble", so a 9-pair kills an 8 outright for free (§5).
    /// - A 10 whose twinstrike hits **two** 8s takes 1 from each, for 2 total, which kills
    ///   it. **[ASSUMED]** — §6 says "any card that attacks this 8 takes 1 damage" and the
    ///   10 has attacked both, so the damage adds. Nothing in the rules addresses the case
    ///   directly.
    fn resolve_attack(&mut self, attackers: Vec<CardId>, hits: &[(CardId, u8)]) {
        // Each card may attack only once per turn (§4); a pair attack is one attack for
        // both members' budget (§5).
        for &id in &attackers {
            if let Some(card) = self.card_mut(id) {
                card.attacks_used += 1;
            }
        }

        let attacker_rank = self.card(attackers[0]).map(|c| c.rank);

        // Read retaliate *before* damage: an 8 that dies to this attack still retaliates.
        let retaliations = hits
            .iter()
            .filter(|(id, _)| {
                self.card(*id)
                    .is_some_and(|c| c.has_live_power(Rank::EIGHT))
            })
            .count();

        for &(id, amount) in hits {
            self.damage_card(id, amount);
        }

        if retaliations > 0 && attacker_rank != Some(Rank::NINE) {
            for &id in &attackers {
                for _ in 0..retaliations {
                    if self.card(id).is_some() {
                        self.damage_card(id, 1);
                    }
                }
            }
        }
    }

    /// Apply damage to one card, then handle death — including the 3's Trap.
    ///
    /// `game_rules.md` §6, the 3: "If killed **while face-down**, it returns to play
    /// face-up with full 2 HP instead of dying — immediately, in the same lane. It comes
    /// back fully active with no waiting period, and it returns face-up so the Trap
    /// **cannot re-trigger**." A base 3 killed post-unlock triggers this too and returns as
    /// a normal, non-base card (§3).
    ///
    /// The card keeps its freeze, if it had one: the Trap is not a flip and nothing in §8
    /// clears a freeze early. **[ASSUMED]**
    pub(crate) fn damage_card(&mut self, id: CardId, amount: u8) {
        let Some((lane, side, slot)) = self.locate(id) else {
            return; // already left play
        };

        {
            let card = &mut self.lanes[lane].sides[side][slot];
            card.damage = card.damage.saturating_add(amount);
        }
        // §7: the quiet-ply counter "resets on damage or a kill and on nothing else".
        self.damage_this_ply = true;

        if !self.lanes[lane].sides[side][slot].is_dead() {
            return;
        }

        let is_face_down_three = {
            let card = &self.lanes[lane].sides[side][slot];
            card.rank == Rank::THREE && !card.face_up
        };

        if is_face_down_three {
            let card = &mut self.lanes[lane].sides[side][slot];
            card.damage = 0;
            card.face_up = true;
            card.known_to = KNOWN_TO_BOTH;
            card.is_base = false;
            card.attacks_used = 0;
            card.attack_allowance = 1;
        } else {
            self.kill_card(lane, side, slot);
        }
    }

    /// Remove a dead card from play.
    ///
    /// §5: "A card that has taken damage equal to its hit points is killed and goes to the
    /// discard pile." §5 also: a pair is "broken if a member dies", so the survivor is
    /// unpaired and free to attack alone again.
    fn kill_card(&mut self, lane: usize, side: usize, slot: usize) {
        let owner = Player::from_index(side);
        self.unpair(lane, owner, slot);
        let card = self.lanes[lane].sides[side].remove(slot);
        self.discards[side].push(card.rank);
    }

    /// Dissolve the pair the card at `slot` belongs to, if any, clearing both members.
    ///
    /// §5: a pair breaks only when a Queen moves a member out or a member dies. It "cannot
    /// be dissolved voluntarily", which is why nothing else calls this.
    fn unpair(&mut self, lane: usize, owner: Player, slot: usize) {
        let side = self.lanes[lane].side_mut(owner);
        let Some(pid) = side[slot].pair_id else {
            return;
        };
        for card in side.iter_mut() {
            if card.pair_id == Some(pid) {
                card.pair_id = None;
            }
        }
    }

    // ======================================================================== powers ==

    /// Turn a card face-up and fire its power.
    ///
    /// The card becomes known to both players first, then the power resolves — so a 5 that
    /// flips a 4 has already revealed the 4 before the peek happens.
    fn flip_card(&mut self, id: CardId) {
        {
            let Some(card) = self.card_mut(id) else { return };
            card.face_up = true;
            card.known_to = KNOWN_TO_BOTH;
        }
        self.fire_power(id);
    }

    /// Fire a card's power.
    ///
    /// Called on a flip, and again for each King reactivation. `game_rules.md` §8: "A
    /// one-shot power is **mandatory** on flip — you do not get to decline the 2's scry or
    /// the 5's flips", and "A power with **no legal target simply fizzles**, and the flip
    /// remains a legal action". Both are visible here as: push a sub-decision when there is
    /// something to choose, and do nothing at all when there is not.
    fn fire_power(&mut self, id: CardId) {
        let Some((lane, side, slot)) = self.locate(id) else {
            return;
        };
        let owner = Player::from_index(side);
        let rank = self.lanes[lane].sides[side][slot].rank;
        debug_assert_eq!(
            owner, self.to_move,
            "a power fired for a player who is not to move"
        );

        match rank {
            // ---- A · Action ---------------------------------------------------------
            // "Gain 1 action this turn, usable however you like. On the turn it is
            // flipped, the Ace itself may attack twice." (§6)
            //
            // A King reactivating an Ace grants the action again — once — and *resets* the
            // attack counter rather than stacking it, so an Ace that attacked once and was
            // then Kinged tops out at three attacks that turn, not four (§6).
            Rank::ACE => {
                self.actions_remaining += 1;
                let card = &mut self.lanes[lane].sides[side][slot];
                card.attacks_used = 0;
                card.attack_allowance = 2;
            }

            // ---- 2 · View -----------------------------------------------------------
            // "Draw a card, then put a card from your hand on the bottom of your draw
            // pile — a scry, not a discard." (§6, house rule §10a)
            //
            // Gated on the pile **you** draw from, not the global `base_unlocked` flag
            // (§3, §9): "if that pile is empty the power does nothing at all — no draw,
            // and no bottoming either, so it cannot be used to refill an empty pile." So a
            // 2 can go dead a turn before the global unlock.
            Rank::TWO => {
                if self.pile(owner).is_empty() {
                    return;
                }
                self.draw_one(owner);
                // The draw guarantees a non-empty hand, so this node always has an answer.
                self.pending.push(Pending::GiveBack { player: owner });
            }

            // ---- 3 · Trap -----------------------------------------------------------
            // Conditional, and only while face-down. Nothing happens on the flip; see
            // `damage_card`.
            Rank::THREE => {}

            // ---- 4 · Foresight ------------------------------------------------------
            // "Look at any one face-down card on the board — including base cards, yours
            // or your opponent's. Private information." (§6)
            Rank::FOUR => {
                if !self.face_down_cards().is_empty() {
                    self.pending.push(Pending::Foresight { player: owner });
                }
            }

            // ---- 5 · Flip -----------------------------------------------------------
            // "Flip all your face-down cards in its lane. You choose the order in which
            // their powers resolve — one at a time, seeing each result before choosing the
            // next." (§6) Includes your base card in that lane once the pile is empty.
            //
            // All-or-nothing: §8 stresses that post-unlock it flips your base card
            // "whether you want it flipped or not", which is what makes a held 5 committal
            // in the endgame. The sole exception is frozen cards, which it "simply skips".
            //
            // The list is snapshotted here, so a face-down card that a Queen brings into
            // the lane *during* the cascade is not caught by it. **[ASSUMED]** — §8 says
            // the player picks from a queue, which reads as a fixed set.
            Rank::FIVE => {
                let queue = self.five_flip_targets(owner, lane, id);
                if !queue.is_empty() {
                    self.pending.push(Pending::ResolveOrder {
                        kind: ResolveKind::FiveFlip,
                        player: owner,
                        lane: lane as u8,
                        remaining: queue,
                    });
                }
            }

            // ---- 6 · Freeze ---------------------------------------------------------
            // "All enemy cards in the lane are frozen: they may not attack, and cannot be
            // flipped at all. Cannot freeze a 9, ever." (§6)
            //
            // Per card, not per lane: "Cards that enter the lane *after* the 6 resolves are
            // not frozen", and a frozen card a Queen relocates stays frozen (§8).
            //
            // The 9's immunity is Nimble, and powers are inert while face-down (§6), so a
            // **face-down** 9 can be frozen. **[ASSUMED]** — the rulebook's "ever" is about
            // timing (a 9 already in the lane is still immune), not about face-up-ness.
            Rank::SIX => {
                let ply = self.ply;
                let enemy = owner.other();
                for card in self.lanes[lane].side_mut(enemy) {
                    if card.has_live_power(Rank::NINE) {
                        continue;
                    }
                    // §8: unfrozen "at the end of the frozen player's next turn — so
                    // exactly one of their turns is lost". Plies strictly alternate, so the
                    // victim's next turn is `ply + 1`.
                    card.frozen_until_ply = Some(ply + 1);
                }
            }

            // ---- 7 · Heal All -------------------------------------------------------
            // "Heal all your damaged cards 2 HP, in all lanes, face-up and face-down."
            // (§6) Includes base cards once the pile is empty. "Healing is capped at the
            // card's maximum HP — a Jack on 1 HP heals to 3, not 5", which is what
            // `saturating_sub` on the damage counter expresses.
            Rank::SEVEN => {
                let unlocked = self.base_unlocked;
                for lane_ref in self.lanes.iter_mut() {
                    for card in lane_ref.side_mut(owner) {
                        if card.is_base && !unlocked {
                            continue;
                        }
                        card.damage = card.damage.saturating_sub(2);
                    }
                }
            }

            // ---- 8, 9, 10, J · constant powers --------------------------------------
            // Nothing fires on the flip; these are read live during combat and targeting.
            Rank::EIGHT | Rank::NINE | Rank::TEN | Rank::JACK => {}

            // ---- Q · Move -----------------------------------------------------------
            // "Move one allied card from another lane into the Queen's lane, face-down or
            // face-up." (§6) Fizzles with no allied card elsewhere — and §8 notes that is
            // often exactly why you flip her: "a Queen with no move available is still a
            // body that can attack".
            Rank::QUEEN => {
                if !self.queen_move_sources(owner, lane).is_empty() {
                    self.pending.push(Pending::QueenSource {
                        player: owner,
                        lane: lane as u8,
                    });
                }
            }

            // ---- K · Empower --------------------------------------------------------
            // "All your face-up cards in this lane reactivate their powers. Does not affect
            // other Kings. Does not affect constant powers." (§6)
            //
            // Because Kings cannot activate Kings, no infinite loop is possible — §6 says
            // so explicitly, and it is worth noting the engine relies on it rather than on
            // a depth limit.
            Rank::KING => {
                let queue = self.king_reactivation_targets(owner, lane, id);
                if !queue.is_empty() {
                    self.pending.push(Pending::ResolveOrder {
                        kind: ResolveKind::KingEmpower,
                        player: owner,
                        lane: lane as u8,
                        remaining: queue,
                    });
                }
            }

            _ => unreachable!("rank {rank} has no power branch"),
        }
    }

    /// Your face-down cards in `lane` that a 5 would flip.
    fn five_flip_targets(&self, owner: Player, lane: usize, five_id: CardId) -> Vec<CardId> {
        self.lanes[lane]
            .side(owner)
            .iter()
            .filter(|c| c.id != five_id)
            .filter(|c| !c.face_up)
            // §8: a 5 "simply skips" frozen cards — "they are untouchable, not merely
            // passive".
            .filter(|c| !c.is_frozen(self.ply))
            // §3: base cards cannot be flipped while any pile is non-empty.
            .filter(|c| self.base_unlocked || !c.is_base)
            .map(|c| c.id)
            .collect()
    }

    /// Your face-up cards in `lane` that a King would refire.
    fn king_reactivation_targets(
        &self,
        owner: Player,
        lane: usize,
        king_id: CardId,
    ) -> Vec<CardId> {
        self.lanes[lane]
            .side(owner)
            .iter()
            .filter(|c| c.id != king_id)
            .filter(|c| c.face_up)
            // Excludes 8/9/10/J (constant), K (excluded by rule) and 3 (conditional).
            .filter(|c| c.rank.is_king_reactivatable())
            .map(|c| c.id)
            .collect()
    }

    // ================================================================ sub-decisions ==

    /// The 4's Foresight. Private, persistent knowledge: only the peeker's bit is set.
    fn do_peek(&mut self, side: Side, lane: usize, slot: usize) {
        let me = self.to_move;
        let target_owner = match side {
            Side::Mine => me,
            Side::Theirs => me.other(),
        };
        self.lanes[lane].side_mut(target_owner)[slot].known_to |= me.bit();
        self.pending.pop();
    }

    /// Resolve the next card in a 5's flip list or a King's reactivation list.
    ///
    /// The chosen card is struck off the list first, and the node popped if that empties
    /// it, *before* the power fires — so anything the power pushes lands on top of the
    /// stack and resolves before the rest of the list. That ordering is what `game_rules.md`
    /// §8 requires: "Each resolution completes before the next begins."
    fn do_resolve_next(&mut self, lane: usize, slot: usize) {
        let me = self.to_move;
        let id = self.lanes[lane].side(me)[slot].id;

        let (kind, list_now_empty) = match self.pending.last_mut() {
            Some(Pending::ResolveOrder {
                kind, remaining, ..
            }) => {
                remaining.retain(|&other| other != id);
                (*kind, remaining.is_empty())
            }
            _ => unreachable!("do_resolve_next called outside a ResolveOrder node"),
        };
        if list_now_empty {
            self.pending.pop();
        }

        match kind {
            // A 5 flips the card, which fires its power.
            ResolveKind::FiveFlip => self.flip_card(id),
            // A King refires an already face-up power. §8: "Freeze does not block
            // reactivation: a face-up frozen card that a King empowers fires its power
            // normally. It still cannot attack."
            ResolveKind::KingEmpower => self.fire_power(id),
        }
    }

    /// The Queen's Move (§6).
    ///
    /// The moved card "keeps its damage, does not reactivate its one-shot power, keeps
    /// constant powers, and may attack after the move if it has not already attacked this
    /// turn" — so everything on the card is carried over untouched except:
    ///
    /// - `is_base` is cleared: §3, "a base card that a Queen moves to another lane stops
    ///   being a base card". `entered_as_base` is *not* cleared, so its owner still may not
    ///   look at it — moving a base card is not a back-door Foresight on your own base.
    /// - the pair breaks: §5, "A pair is broken if a Queen moves one member to another
    ///   lane."
    /// - the freeze does **not** clear: §8, "a frozen card a Queen moves to another lane
    ///   stays frozen for the remaining duration. A Queen is therefore not an escape hatch
    ///   from a 6 — she relocates the problem."
    fn do_move_here(&mut self, from_lane: usize, slot: usize) {
        let Some(Pending::QueenSource { player, lane }) = self.pending.pop() else {
            unreachable!("do_move_here called outside a QueenSource node");
        };
        let to_lane = lane as usize;

        self.unpair(from_lane, player, slot);
        let mut card = self.lanes[from_lane].side_mut(player).remove(slot);
        card.is_base = false;
        card.pair_id = None;
        self.lanes[to_lane].side_mut(player).push(card);
    }

    /// The 2's View, second half: give a card back (§10a).
    ///
    /// Under the house rule it goes on the **bottom of your own pile**, known to you and to
    /// nobody else — "You know both its identity and its position — the bottom of that
    /// pile" (§5). Under `two_power = discard` it goes to the public discard instead, which
    /// shrinks the pile and is exactly the parity lever §10a objects to.
    fn do_give_back(&mut self, rank: Rank) {
        let Some(Pending::GiveBack { player }) = self.pending.pop() else {
            unreachable!("do_give_back called outside a GiveBack node");
        };
        let hand = &mut self.hands[player.idx()];
        let pos = hand
            .iter()
            .position(|&r| r == rank)
            .expect("legality guaranteed this rank is in hand");
        hand.remove(pos);

        match self.config.two_power {
            TwoPower::Bottom => self.pile_mut(player).put_on_bottom(rank, player),
            TwoPower::Discard => self.discards[player.idx()].push(rank),
        }
    }

    // ================================================================ turn machinery ==

    /// Draw one card into `player`'s hand, if their pile is non-empty.
    ///
    /// The pile records who knows each card's rank, so a card the drawer bottomed earlier
    /// comes back as known. The mask is then dropped, because the engine models a hand as a
    /// multiset of ranks.
    ///
    /// That drop loses one thing the physical game has, and only in the **base** variant: if
    /// your opponent bottomed a card into the shared pile and *you* drew it, they know a
    /// rank you hold. In the split variants you only ever draw from your own pile, so a
    /// bottomed card can only return to the player who put it there and nothing is lost.
    /// Recorded as a Phase 3 observation-encoding gap in `DESIGN.md` §5; it affects no
    /// legality or outcome.
    fn draw_one(&mut self, player: Player) -> Option<Rank> {
        let (rank, _known_to) = self.pile_mut(player).draw()?;
        self.hands[player.idx()].push(rank);
        self.hands[player.idx()].sort_unstable();
        self.draws_taken[player.idx()] += 1;
        Some(rank)
    }

    /// Actions the player to move gets this turn.
    ///
    /// §2: "The first player takes only two actions on their opening turn. Every turn
    /// thereafter is three actions."
    fn actions_for_ply(&self, ply: u32) -> u32 {
        if ply == 0 {
            self.config.first_turn_actions
        } else {
            self.config.actions_per_turn
        }
    }

    /// Start of turn for `self.to_move`: reset attack budgets, thaw, draw, set the action
    /// allowance (§4).
    pub(crate) fn begin_turn(&mut self) {
        let ply = self.ply;
        let me = self.to_move;

        // Clear expired freezes everywhere. `Card::is_frozen` already compares against the
        // ply, so this is bookkeeping rather than rules — it keeps the rendered board and
        // any future observation encoding honest.
        for lane in self.lanes.iter_mut() {
            for side in lane.sides.iter_mut() {
                for card in side.iter_mut() {
                    if matches!(card.frozen_until_ply, Some(last) if ply > last) {
                        card.frozen_until_ply = None;
                    }
                }
            }
        }

        // "Each card may attack only once per turn" (§4) — and a freshly flipped Ace's
        // allowance of 2 belongs to that turn only.
        for lane in self.lanes.iter_mut() {
            for card in lane.side_mut(me) {
                card.reset_turn_attacks();
            }
        }

        // "Draw one card from the draw pile, if it is non-empty. The draw happens at the
        // start of the turn, including the first player's opening turn." (§4)
        for _ in 0..self.config.draws_per_turn {
            self.draw_one(me);
        }

        self.actions_remaining = self.actions_for_ply(ply);
        self.damage_this_ply = false;

        self.refresh_base_unlocked();
        self.check_terminal();
    }

    /// End the current turn and begin the next.
    ///
    /// `pub(crate)` for `testkit::end_turn`, which lets a hand-built position skip a turn
    /// without having to find three legal actions to burn.
    pub(crate) fn end_turn(&mut self) {
        // §7: the quiet-ply counter counts "individual player turns (plies)" with "no damage
        // dealt and no kill", and "resets on damage or a kill and on nothing else".
        if self.damage_this_ply {
            self.quiet_plies = 0;
        } else {
            self.quiet_plies += 1;
        }

        self.ply += 1;
        self.to_move = self.to_move.other();
        self.begin_turn();
    }

    /// Latch `base_unlocked` once every pile is empty.
    ///
    /// Never cleared, and only ever evaluated at an action boundary. Both properties are
    /// required by §10a: firing the house 2 on a one-card pile draws that card and puts one
    /// back, dipping the pile to zero *inside* the resolution — "There is no last-card
    /// stall ... The engine needs no special case for an empty-after-draw pile."
    fn refresh_base_unlocked(&mut self) {
        if !self.base_unlocked && self.all_piles_empty() {
            self.base_unlocked = true;
        }
    }

    /// Finish resolving an action: prune, unlock, evaluate, and end the turn if it is over.
    fn settle(&mut self) {
        self.normalize_pending();
        if !self.pending.is_empty() {
            // §7: "The terminal check runs after each action fully resolves, including
            // every sub-decision that action opened. It never runs mid-resolution."
            return;
        }
        self.refresh_base_unlocked();
        self.check_terminal();
        if self.outcome.is_over() {
            return;
        }
        if self.actions_remaining == 0 {
            self.end_turn();
        }
        self.skip_turns_with_nothing_to_do();
    }

    /// End any turn that has no legal action in it, and keep going while the next one is
    /// equally empty.
    ///
    /// §4 makes actions mandatory and offers no pass, so "I cannot act" is not a decision a
    /// player makes — it is a fact about the position, and the engine acts on it here. This
    /// is what keeps [`GameState::legal_actions`] empty *only* when the game is over, so no
    /// caller ever has to ask what to do with a position that allows nothing.
    ///
    /// **It terminates.** Every iteration calls `end_turn`, which advances the ply and runs
    /// the terminal check; `check_terminal` ends the game at `config.max_plies` and at the
    /// quiet-ply threshold, and a turn with no action in it is by definition quiet. Two
    /// permanently stuck players therefore draw rather than loop.
    pub(crate) fn skip_turns_with_nothing_to_do(&mut self) {
        // The hand check is the cheap half of the test and almost always settles it: a card
        // in hand can always be played into any of your own lanes, so a player holding one
        // is never stuck. Enumerating the main phase on every action would roughly double
        // the cost of a game (see `apply_trusted`), and this keeps it off the hot path.
        //
        // That shortcut is a second statement of a rule, which is how legality drifts. The
        // guard is `rule_4_a_player_holding_a_card_is_never_stuck`, which asserts across
        // full games in every variant that a non-empty hand really does imply a legal
        // `Play`. If a lane capacity ever becomes a *rule* rather than an encoding cap,
        // that test fails here rather than somewhere far away.
        while !self.outcome.is_over()
            && self.pending.is_empty()
            && self.hands[self.to_move.idx()].is_empty()
            && self.legal_main_actions().is_empty()
        {
            self.end_turn();
        }
    }

    // ==================================================================== termination ==

    /// How many lanes `player` has won.
    ///
    /// §7 — **all three** conditions must hold:
    ///
    /// 1. the opponent has no cards left in that lane, base card included;
    /// 2. every draw pile is empty (`base_unlocked`);
    /// 3. the opponent's hand is empty.
    ///
    /// "So long as the opponent holds any card in hand, they can defend the lane, and it
    /// cannot be won. Lane wins are therefore strictly an endgame event."
    pub fn lanes_won_by(&self, player: Player) -> usize {
        if !self.base_unlocked {
            return 0;
        }
        let opponent = player.other();
        if !self.hands[opponent.idx()].is_empty() {
            return 0;
        }
        self.lanes
            .iter()
            .filter(|lane| lane.side(opponent).is_empty())
            .count()
    }

    /// Set `self.outcome` if the game is over.
    ///
    /// §7 notes that a lane win cannot be undone, so live evaluation and latched evaluation
    /// are equivalent and the engine simply re-checks live state: refilling an empty side
    /// needs a card from hand (empty by condition 3), a draw (no pile by condition 2), or a
    /// Queen — "and a Queen only moves cards into the lane she is already in, so an empty
    /// side has no Queen to pull anything back."
    fn check_terminal(&mut self) {
        if self.outcome.is_over() {
            return;
        }

        // Safety cap first, so a rules bug shows up as a logged draw rather than a hang.
        if self.ply >= self.config.max_plies {
            self.outcome = Outcome::Draw(DrawReason::PlyLimit);
            return;
        }

        let need = self.config.lanes_to_win;
        let won_by_p0 = self.lanes_won_by(Player::P0);
        let won_by_p1 = self.lanes_won_by(Player::P1);

        // §7: "A single action may complete the second and third lanes at once. This is a
        // plain win." And if *both* players reach the threshold on the same check — the
        // retaliate double-kill — "it is a draw (0.5/0.5): a symmetric outcome gets a
        // symmetric result, with no arbitrary tiebreak."
        match (won_by_p0 >= need, won_by_p1 >= need) {
            (true, true) => {
                self.outcome = Outcome::Draw(DrawReason::MutualLaneWin);
                return;
            }
            (true, false) => {
                self.outcome = Outcome::Win(Player::P0);
                return;
            }
            (false, true) => {
                self.outcome = Outcome::Win(Player::P1);
                return;
            }
            (false, false) => {}
        }

        // A decisive result always beats the stalemate rule, so this comes last.
        if self.quiet_plies >= self.config.stalemate_quiet_plies {
            self.outcome = Outcome::Draw(DrawReason::Stalemate);
        }
    }
}
