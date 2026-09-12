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
use crate::card::{Card, CardId, KNOWN_TO_BOTH};
use crate::config::TwoPower;
use crate::damage::{DamageQueue, DamageSource, Hit};
use crate::outcome::{DrawReason, Outcome};
use crate::player::Player;
use crate::powers::{self, LethalOutcome, PowerCtx, PowerId};
use crate::rank::Rank;
use crate::state::{GameState, LaneChoice, OptionChoice, Pending, ResolveKind};

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
            Action::ChooseLane { side, lane } => self.do_choose_lane(side, lane as usize),
            Action::ChooseOption { option } => self.do_choose_option(option),
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
        let attacker_power = self.lanes[lane].side(me)[attacker_slot].live_power(&self.config);
        let primary = self.lanes[lane].side(opponent)[target_slot].id;
        let source = DamageSource::Attack {
            attacker: attackers[0],
            partner: attackers.get(1).copied(),
        };

        if attacker_power.is_some_and(|p| p.twinstrikes()) {
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
            let damage = if is_pair {
                self.config.pair_attack_damage
            } else {
                self.config.single_attack_damage
            };
            self.resolve_attack(attackers, &[(primary, damage)], source);
            return;
        }

        let target = &self.lanes[lane].side(opponent)[target_slot];
        let damage = self.attack_damage(attacker_power, target, is_pair);
        self.resolve_attack(attackers, &[(primary, damage)], source);
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
        let half = self.config.twinstrike_split_damage;
        let source = DamageSource::Attack {
            attacker: attackers[0],
            partner: attackers.get(1).copied(),
        };
        self.resolve_attack(attackers, &[(primary, half), (secondary, half)], source);
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
    fn resolve_attack(
        &mut self,
        attackers: Vec<CardId>,
        hits: &[(CardId, u8)],
        source: DamageSource,
    ) {
        // Each card may attack only once per turn (§4); a pair attack is one attack for
        // both members' budget (§5).
        for &id in &attackers {
            if let Some(card) = self.card_mut(id) {
                card.attacks_used += 1;
            }
        }

        let attacker_power = self.card(attackers[0]).and_then(|c| c.live_power(&self.config));

        // Read retaliate *before* damage: under the rules as written an 8 that dies to this
        // attack still retaliates. `retaliate_on_survival` asks again afterwards instead;
        // both readings are taken here so the attack has one place that owns the ordering.
        let retaliation = powers::eight::read(self, hits);

        // The attack's own damage, and everything it triggers, resolves in full first.
        for &(id, amount) in hits {
            self.enqueue_damage(Hit {
                target: id,
                amount,
                source,
            });
        }
        self.drain_damage();

        // Nimble takes no retaliate damage, and "pairing does not forfeit Nimble", so a
        // 9-pair kills an 8 outright for free (§5).
        if attacker_power.is_some_and(|p| p.is_nimble()) {
            return;
        }
        let owed = powers::eight::settle(self, &retaliation);
        if owed.is_empty() {
            return;
        }
        let per_hit = powers::eight::damage_per_hit(self);
        // A pair attacking an 8: **both members take 1** (§5). A 10 whose twinstrike hits
        // two 8s takes one from each, for 2 total, which kills it — **[ASSUMED]**, §6 says
        // "any card that attacks this 8 takes 1 damage" and the 10 has attacked both.
        for &attacker in &attackers {
            for &eight in &owed {
                self.enqueue_damage(Hit {
                    target: attacker,
                    amount: per_hit,
                    source: DamageSource::Retaliate { from: eight },
                });
            }
        }
        self.drain_damage();
    }

    // ======================================================================== damage ==

    /// Add a hit to the queue. Lands when the current [`GameState::drain_damage`] reaches
    /// it, or immediately after if none is running.
    pub(crate) fn enqueue_damage(&mut self, hit: Hit) {
        self.damage_queue.push(hit);
    }

    /// Apply every queued hit, including any a death trigger adds while we are draining.
    ///
    /// Re-entrant by design: a power body that enqueues damage calls this indirectly through
    /// `apply_one_hit`, and the nested call returns immediately rather than starting a
    /// second drain. That is what keeps the order a single FIFO — `MODULAR_RULES.md` §3a
    /// describes the bug the other way round, where a 10 twinstrikes two face-down 3s and
    /// the second 3's vengeance is dropped because the loop that spawned it moved on.
    pub(crate) fn drain_damage(&mut self) {
        if self.damage_queue.is_draining() {
            return;
        }
        self.damage_queue.set_draining(true);
        let mut applied = 0usize;
        while let Some(hit) = self.damage_queue.pop() {
            applied += 1;
            assert!(
                applied <= DamageQueue::MAX_CASCADE,
                "damage cascade exceeded {} hits — a ruleset has a cycle of death triggers. \
                 `game_rules.md` §7's finiteness argument does not cover it.",
                DamageQueue::MAX_CASCADE
            );
            self.apply_one_hit(hit);
        }
        self.damage_queue.set_draining(false);
    }

    /// Apply damage to one card, then ask its power what happens if that was lethal.
    fn apply_one_hit(&mut self, hit: Hit) {
        let Some((lane, side, slot)) = self.locate(hit.target) else {
            return; // already left play
        };

        // A shield absorbs the whole hit and is spent doing it (`MODULAR_RULES.md` §7,
        // reserve item 3 — set by `PowerId::SevenShieldAll`). Under the canonical ruleset no
        // card ever carries a status bit, so this is one predictable-false test per hit.
        //
        // It returns **before** `damage_this_ply`, so a shielded hit does not reset §7's
        // quiet-ply counter: nothing was damaged and nothing was killed, and treating it as
        // action would let two shielded boards stall the stalemate detector forever.
        {
            let card = &mut self.lanes[lane].sides[side][slot];
            if card.has_status(crate::card::STATUS_SHIELDED) {
                card.clear_status(crate::card::STATUS_SHIELDED);
                return;
            }
            card.damage = card.damage.saturating_add(hit.amount);
        }
        // §7: the quiet-ply counter "resets on damage or a kill and on nothing else".
        self.damage_this_ply = true;

        if !self.lanes[lane].sides[side][slot].is_dead(&self.config) {
            return;
        }

        // The power is read whether or not it is *live*: the 3's Trap fires only while the
        // card is face-down, which is the opposite of a constant power's condition, so the
        // face-up test belongs to the power rather than to this dispatch.
        let power = self.lanes[lane].sides[side][slot].power(&self.config);
        let ctx = PowerCtx {
            id: hit.target,
            owner: Player::from_index(side),
            lane,
            side,
            slot,
        };
        match powers::on_lethal_damage(self, power, ctx, hit.source) {
            LethalOutcome::Restored => {}
            LethalOutcome::Die => self.kill_card(lane, side, slot),
        }
    }

    /// Apply one hit with no attribution, immediately. For `testkit` positions and anything
    /// the engine applies as bookkeeping rather than as a rule.
    #[allow(dead_code)]
    pub(crate) fn damage_card(&mut self, id: CardId, amount: u8) {
        self.enqueue_damage(Hit {
            target: id,
            amount,
            source: DamageSource::Unattributed,
        });
        self.drain_damage();
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

        // `MODULAR_RULES.md` §5b: the thirteen-arm match on `Rank` that used to live here
        // moved into `powers/`, one module per rank. The turn machinery no longer names a
        // card, which is the property that makes changing the 3 touch `powers/three.rs` and
        // nothing else.
        let power = self.config.power(rank);
        powers::on_flip(
            self,
            power,
            PowerCtx {
                id,
                owner,
                lane,
                side,
                slot,
            },
        );
    }

    /// Your face-down cards in `lane` that a 5 would flip.
    pub(crate) fn five_flip_targets(
        &self,
        owner: Player,
        lane: usize,
        five_id: CardId,
    ) -> Vec<CardId> {
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
    pub(crate) fn king_reactivation_targets(
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
            // Excludes 8/9/10/J (constant), K (excluded by rule) and 3 (conditional) — but
            // reads it off the *power*, so an ablated card is excluded too rather than
            // being refired into a no-op.
            .filter(|c| self.config.power(c.rank).is_king_reactivatable())
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

        // `view_choose` puts the destination to the player instead of reading it off the
        // config. The card is already out of hand either way, so the option node cannot
        // fizzle and leave a card in limbo.
        if self.config.power(Rank::TWO) == PowerId::TwoViewChoose {
            self.pending.push(Pending::ChooseOption {
                player,
                kind: OptionChoice::GiveBackDestination { rank },
            });
            return;
        }
        match self.config.two_power {
            TwoPower::Bottom => self.pile_mut(player).put_on_bottom(rank, player),
            TwoPower::Discard => self.discards[player.idx()].push(rank),
        }
    }

    // ==================================================== the encoder reserve (§7) ==

    /// Answer a [`Pending::ChooseLane`] node. `MODULAR_RULES.md` §7, reserve item 2.
    ///
    /// `side` is not read: `legal_lane_choices` is what decides which side a given power may
    /// name, and it emits only the one that power is entitled to. Checking it again here
    /// would be a second copy of that rule, which is exactly how legality and resolution
    /// drift apart.
    fn do_choose_lane(&mut self, _side: Side, lane: usize) {
        let Some(Pending::ChooseLane { player, kind }) = self.pending.pop() else {
            unreachable!("do_choose_lane called outside a ChooseLane node");
        };
        match kind {
            LaneChoice::KingEmpower { king } => {
                let queue = self.king_reactivation_targets(player, lane, king);
                if !queue.is_empty() {
                    self.pending.push(Pending::ResolveOrder {
                        kind: ResolveKind::KingEmpower,
                        player,
                        lane: lane as u8,
                        remaining: queue,
                    });
                }
            }
        }
    }

    /// Answer a [`Pending::ChooseOption`] node. §7, reserve item 2.
    fn do_choose_option(&mut self, option: u8) {
        let Some(Pending::ChooseOption { player, kind }) = self.pending.pop() else {
            unreachable!("do_choose_option called outside a ChooseOption node");
        };
        match kind {
            OptionChoice::GiveBackDestination { rank } => match option {
                0 => self.pile_mut(player).put_on_bottom(rank, player),
                1 => self.discards[player.idx()].push(rank),
                other => unreachable!("legality offers only 0..{}, got {other}", kind.count()),
            },
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
    pub(crate) fn draw_one(&mut self, player: Player) -> Option<Rank> {
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
