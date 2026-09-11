//! Instrumented play — the Phase 2 deliverable.
//!
//! `PLAN.md` Phase 2: "**Deliverable:** first real strategic observations logged to
//! `FINDINGS.md`", and the note under it: "A competent ISMCTS bot with zero training will
//! expose lane-allocation patterns, flip timing, and the first-player edge. Do not rush past
//! it to get to the neural net."
//!
//! So this module watches games rather than just counting them. It records what
//! `FINDINGS.md`'s hypotheses actually ask about:
//!
//! | Recorded | Answers |
//! |---|---|
//! | hand size when the last pile empties | **H2** — is hand size the resource, and where is the crossover |
//! | share of plays and attacks in a player's busiest two lanes | **H3** — does strong play concentrate |
//! | how many played cards are ever flipped, and how late | **H4** — is concealment worth its tempo |
//! | flips per rank, and the mean ply of each | **H5, H6, H7** — which cards get held, which get fired on sight |
//! | quiet-ply draws | **F1.2** — the stalemate rule, which random play could not reach |
//! | maximum cards on one side of one lane | **F1.7** — the Phase 3 encoding bound |
//!
//! # Instrumentation reads ground truth, and that is fine
//!
//! [`GameStats`] records the rank of every flipped card by looking at the engine's state.
//! That is measurement, not play: the number never re-enters a decision. The rule that
//! matters — agents see only their own information set — is enforced where decisions are
//! made, in [`crate::agents`] and [`GameState::determinize`].

use std::sync::Arc;

use crate::action::Action;
use crate::agents::{Agent, AgentSpec, SearchInProgress, SearchStep};
use crate::card::CardId;
use crate::config::GameConfig;
use crate::encode::{action_dim, obs_dim};
use crate::nn::{MlpEvaluator, Scratch};
use crate::outcome::{DrawReason, Outcome};
use crate::player::Player;
use crate::rank::Rank;
use crate::state::GameState;

/// RNG stream tags for a match. Distinct from every `setup.rs` stream, so how many random
/// numbers an agent consumes cannot perturb the deal.
pub const AGENT_STREAM: [u64; 2] = [0x4147_454E_5430_0002, 0x4147_454E_5431_0002];

/// How a card first came to be face-up. `game_rules.md` §6 has three ways, and they are not
/// the same event: only the first is a decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaceUpKind {
    /// Still face-down. Either on the board when the game ended, or killed while hidden.
    Never,
    /// Its owner spent an action on [`Action::Flip`].
    Chose,
    /// A 5 or a King turned it up from inside a resolution. Its owner was acting, but chose
    /// some other card.
    Cascade,
    /// A death trigger — canonically the 3's Trap — turned it up as it was killed.
    Trap,
}

impl FaceUpKind {
    /// The token written into `cards.csv`. Stable: the analysis script matches on it.
    pub fn label(self) -> &'static str {
        match self {
            FaceUpKind::Never => "never",
            FaceUpKind::Chose => "chose",
            FaceUpKind::Cascade => "cascade",
            FaceUpKind::Trap => "trap",
        }
    }
}

/// The whole life of one card that entered play, from the outside.
///
/// One row per card is what makes a question like "how long does a 7 stay face-down" or "how
/// often is a card killed before it is ever flipped" answerable *after* the games have been
/// played, without new engine code and without a replay. The aggregate counters below are
/// derived from these rows rather than kept alongside them, so the two cannot disagree.
///
/// Every ply here is [`GameState::ply`] — one player's turn. A card's own tenure is a
/// difference of two of them, and both events belong to its owner's turns whenever the owner
/// caused them, so `(face_up_ply - entered_ply) / 2` is a whole number of *the owner's* turns
/// and a card flipped on the turn it was played measures 0. A [`FaceUpKind::Trap`] lands on
/// the opponent's turn instead, so that one is a half-integer, which is correct rather than
/// an off-by-one.
#[derive(Clone, Copy, Debug)]
pub struct CardRecord {
    pub id: CardId,
    pub owner: Player,
    pub rank: Rank,
    /// It entered play as a base card, so it was never played from hand and could not be
    /// turned face-up before the unlock. Base cards are a different population and every
    /// table that mixes them says so.
    pub entered_as_base: bool,
    /// The ply it entered play on. `0` for a base card.
    pub entered_ply: u32,
    /// The ply it first turned face-up on, if it ever did.
    pub face_up_ply: Option<u32>,
    pub face_up_kind: FaceUpKind,
    /// The ply it was killed on. `None` means it was still on the board at the end.
    pub death_ply: Option<u32>,
    /// Whether it was face-up when it was killed. Only meaningful with `death_ply` set.
    pub died_face_up: bool,
    /// It was a member of a declared pair at some point (§5).
    pub ever_paired: bool,
}

/// The live index behind [`GameStats::cards`]: one entry per card on the board, carrying its
/// row in that vector and its face-up state as of the last reconcile.
///
/// `row` rather than a reference because the vector grows, and an index survives that.
#[derive(Clone, Copy, Debug)]
struct LiveCard {
    id: CardId,
    row: usize,
    face_up: bool,
    /// Reconcile scratch: cleared before each board walk, set by whatever the walk finds.
    /// An entry still clear afterwards names a card that left the board, which is the only
    /// way [`GameStats::note_cards`] learns about a death.
    seen: bool,
}

/// Everything one game produced.
///
/// Per-player arrays are indexed by [`Player::idx`], **not** by seat: `by_player[0]` is P0's,
/// whichever agent that was. [`MatchStats`] does the seat-to-agent mapping.
#[derive(Clone, Debug)]
pub struct GameStats {
    pub seed: u64,
    pub outcome: Outcome,
    /// Total player turns played, counting from 1.
    pub plies: u32,
    /// Total decisions made, including the free sub-choices powers open.
    pub decisions: u32,

    /// The ply on which every draw pile first became empty, unlocking base cards.
    pub ply_at_unlock: Option<u32>,
    /// Hand sizes at that moment — the H2 measurement.
    pub hand_at_unlock: [u32; 2],
    /// Hand sizes when the game ended.
    pub hand_at_end: [u32; 2],
    /// Largest number of cards seen on one side of one lane.
    pub max_side_occupancy: usize,
    pub draws_taken: [u32; 2],

    /// Cards played from hand, by player and lane.
    pub plays_by_lane: [Vec<u32>; 2],
    /// Attack actions, by player and lane.
    pub attacks_by_lane: [Vec<u32>; 2],
    /// Cards played from hand, by player and rank.
    pub plays_by_rank: [[u32; Rank::COUNT]; 2],
    /// Flips, by player and rank. Includes base cards, which were never "played".
    pub flips_by_rank: [[u32; Rank::COUNT]; 2],
    /// Of those, the ones that were base cards, by player and rank. Counted separately so
    /// the flip *rate* has a clean denominator: a base card was never played from hand, so
    /// it belongs to neither side of "how much of what I played did I turn face-up".
    pub base_flips_by_rank: [[u32; Rank::COUNT]; 2],
    /// Summed ply of every flip, by player and rank. Divide by `flips_by_rank` for the mean
    /// ply at which that rank goes face-up.
    pub flip_ply_sum: [[u64; Rank::COUNT]; 2],
    /// Cards still face-down on the board when the game ended.
    pub unflipped_at_end: [u32; 2],
    /// Turns that ended with actions unspent because **nothing was legal**.
    ///
    /// §4 makes acting mandatory and there is no pass, so this is not an action anyone
    /// takes — the engine ends such a turn itself and the agent never sees the position.
    /// It is counted here from the outside, by watching the ply advance while the acting
    /// player still had an allowance. A rising count means play is running out of material,
    /// not that the agent is being passive; there is no longer any way to be passive.
    pub stuck_turns: [u32; 2],
    pub pairs_declared: [u32; 2],
    /// Pairs declared, by player and the rank that was paired. Two cards per entry.
    pub pairs_by_rank: [[u32; Rank::COUNT]; 2],

    /// Rank counts of the hand each player holds **at the start of their own first turn**,
    /// indexed by [`Rank::index`].
    ///
    /// ⚠️ Not the dealt hand, and the difference is the whole reason this is worded so
    /// carefully. The draw happens at the start of a turn *including the first player's*
    /// (`game_rules.md` §2), and `GameState::new` performs P0's, so P0 is holding six cards
    /// before anyone has acted while P1 is still holding five. Recording "the hand at
    /// setup" would therefore give P0 an extra card and P1 none, and every per-rank win
    /// rate would carry that asymmetry as if it were a fact about the cards.
    pub start_hand: [[u8; Rank::COUNT]; 2],
    /// Whether `start_hand` has been taken for each player yet. P1's comes one turn later
    /// than P0's, so this cannot be inferred from the counts.
    start_hand_taken: [bool; 2],
    /// Rank counts of each player's hand at the moment the last pile emptied. All zero if
    /// the game ended before the unlock — read [`GameStats::ply_at_unlock`] first.
    pub unlock_hand: [[u8; Rank::COUNT]; 2],

    /// One row per card that ever entered play — every base card, and every card played
    /// from hand. Cards that stayed in a hand or a pile all game are not here, because they
    /// were never on the board to have a life.
    ///
    /// In game order: base cards first, then plays as they happened.
    pub cards: Vec<CardRecord>,
    /// The live index into `cards`. Reconcile scratch, not a result.
    live: Vec<LiveCard>,

    /// Face-down cards that were killed and survived it — a death trigger firing — by
    /// player and rank.
    ///
    /// Canonically this is only the 3, the one rank whose power is conditioned on staying
    /// hidden: the flip rate alone cannot say whether holding it paid, because a 3 held to
    /// the end of the game and a 3 held until it sprang look identical in `flips_by_rank`.
    /// This counts the payoff.
    ///
    /// Keyed by **rank** rather than hardcoded to the 3 (`MODULAR_RULES.md` §11 step 7): a
    /// card module that owns its rules should own its measurement too, so that a new death
    /// trigger arrives instrumented rather than invisible to the probe.
    ///
    /// Derived from [`GameStats::cards`] at the end of the game, not counted alongside it.
    pub triggers_sprung_by_rank: [[u32; Rank::COUNT]; 2],
    /// Cards turned face-up by a **cascade** — a 5 or a King — rather than by their owner
    /// choosing to, by player and rank.
    ///
    /// Split out because `flips_by_rank` counts only [`Action::Flip`]; a 5 flips the lane
    /// from inside `apply`, so those never reach [`GameStats::note_action`]. Without this
    /// the four outcomes of a played card do not sum.
    ///
    /// ⚠️ **This counts every rank as of 2026-09-10.** It used to be built by watching only
    /// the cards whose power has a death trigger, so a cascade-flipped 7 was counted
    /// nowhere and [`AgentBehaviour::card_fates`] silently reported 0 in that column for
    /// every rank but the 3. It is now derived from [`GameStats::cards`], which sees all of
    /// them.
    pub flipped_by_cascade_by_rank: [[u32; Rank::COUNT]; 2],
    /// Cards still face-down on the board when the game ended, by player and rank. Includes
    /// base cards.
    pub face_down_at_end_by_rank: [[u32; Rank::COUNT]; 2],
    /// Cards killed while still face-down, by player and rank. Includes base cards.
    ///
    /// The other end of the same question as `face_down_at_end_by_rank`: a card that is
    /// never flipped either survived to the end hidden or was killed hidden, and those are
    /// very different things to have happened to it.
    pub died_face_down_by_rank: [[u32; Rank::COUNT]; 2],
    /// Cards killed while face-up, by player and rank.
    pub died_face_up_by_rank: [[u32; Rank::COUNT]; 2],
}

impl GameStats {
    fn new(config: &GameConfig, seed: u64) -> GameStats {
        GameStats {
            seed,
            outcome: Outcome::Ongoing,
            plies: 0,
            decisions: 0,
            ply_at_unlock: None,
            hand_at_unlock: [0, 0],
            hand_at_end: [0, 0],
            max_side_occupancy: 0,
            draws_taken: [0, 0],
            plays_by_lane: [vec![0; config.lanes], vec![0; config.lanes]],
            attacks_by_lane: [vec![0; config.lanes], vec![0; config.lanes]],
            plays_by_rank: [[0; Rank::COUNT]; 2],
            flips_by_rank: [[0; Rank::COUNT]; 2],
            base_flips_by_rank: [[0; Rank::COUNT]; 2],
            flip_ply_sum: [[0; Rank::COUNT]; 2],
            unflipped_at_end: [0, 0],
            stuck_turns: [0, 0],
            pairs_declared: [0, 0],
            pairs_by_rank: [[0; Rank::COUNT]; 2],
            start_hand: [[0; Rank::COUNT]; 2],
            start_hand_taken: [false; 2],
            unlock_hand: [[0; Rank::COUNT]; 2],
            // 5 base cards + up to ~26 played cards in the split variants. One allocation
            // per game either way; the reserve just stops it doubling four times.
            cards: Vec::with_capacity(40),
            live: Vec::with_capacity(24),
            triggers_sprung_by_rank: [[0; Rank::COUNT]; 2],
            flipped_by_cascade_by_rank: [[0; Rank::COUNT]; 2],
            face_down_at_end_by_rank: [[0; Rank::COUNT]; 2],
            died_face_down_by_rank: [[0; Rank::COUNT]; 2],
            died_face_up_by_rank: [[0; Rank::COUNT]; 2],
        }
    }

    /// Record the deal: the base cards, which are on the board before anyone has acted.
    ///
    /// The hands are **not** taken here — see [`GameStats::start_hand`]. They are taken by
    /// [`GameStats::note_start_hands`], which this calls for P0 and which catches P1 one
    /// turn later.
    fn note_setup(&mut self, state: &GameState) {
        // `acting` and `flipped` cannot matter: nothing is face-up yet and nothing has died.
        self.note_cards(0, Player::P0, None, state);
        self.note_start_hands(state);
    }

    /// Take a player's opening hand the first time the game stands at the start of their
    /// first turn — ply 0 for P0, ply 1 for P1, after each has drawn.
    ///
    /// Called after every action, so the guard has to be a flag: `note_state` runs again
    /// mid-turn, by which point the player may have spent a card.
    fn note_start_hands(&mut self, state: &GameState) {
        for p in Player::BOTH {
            if state.ply as usize == p.idx() && !self.start_hand_taken[p.idx()] {
                self.start_hand_taken[p.idx()] = true;
                for rank in &state.hands[p.idx()] {
                    self.start_hand[p.idx()][rank.index()] += 1;
                }
            }
        }
    }

    /// Reconcile [`GameStats::cards`] against the board after one `apply`, at `at_ply`.
    ///
    /// Three transitions are read off by comparing the board to the live index, which is
    /// what lets one walk cover arrivals, flips and deaths at once:
    ///
    /// - **A card the index does not know** just entered play — a card played from hand, or
    ///   at setup, a base card.
    /// - **A card the index has face-down that is now face-up.** Which of §6's three ways
    ///   it was is decided by who was acting and what they chose:
    ///   1. **Its owner was not acting.** Then it was killed, and a death trigger fired —
    ///      [`FaceUpKind::Trap`]. Nothing else can turn an opponent's card face-up: the 4
    ///      only *looks*, privately, and the 5 flips its own side. And the owner cannot
    ///      have sprung their own 3 on their own turn, because damage only ever originates
    ///      from an attack — including the 8's Retaliate and the 10's Twinstrike — and a
    ///      face-down card cannot attack (§4).
    ///   2. **Its owner was acting and named it.** An ordinary [`FaceUpKind::Chose`] flip.
    ///   3. **Its owner was acting and named something else.** A 5 or a King turned it up
    ///      from inside the resolution — [`FaceUpKind::Cascade`].
    /// - **A card the index knows that is no longer on the board.** It was killed, and it
    ///   died in whatever state the index last saw it in. Nothing else removes a card from
    ///   play: a Queen moves one between lanes and it keeps its id, and a face-down 3 that
    ///   is killed springs rather than dying (§6), so it is still there afterwards.
    ///
    /// ⚠️ Case 1's reasoning is a property of the **canonical** ruleset, not of the engine.
    /// A variant that let a player damage their own cards would misfile a self-inflicted
    /// spring as a cascade flip. Nothing here can detect that; it is noted so the next
    /// person to add such a power knows this classification is one of the things it breaks.
    ///
    /// `flipped` is the card [`Action::Flip`] named, looked up before the apply while it was
    /// still in the slot the action gave.
    fn note_cards(
        &mut self,
        at_ply: u32,
        acting: Player,
        flipped: Option<CardId>,
        state: &GameState,
    ) {
        for live in &mut self.live {
            live.seen = false;
        }
        for p in Player::BOTH {
            for (_, _, card) in state.cards_of(p) {
                match self.live.iter().position(|l| l.id == card.id) {
                    Some(i) => {
                        self.live[i].seen = true;
                        if card.face_up && !self.live[i].face_up {
                            self.live[i].face_up = true;
                            let kind = if flipped == Some(card.id) {
                                FaceUpKind::Chose
                            } else if p != acting {
                                FaceUpKind::Trap
                            } else {
                                FaceUpKind::Cascade
                            };
                            let row = self.live[i].row;
                            self.cards[row].face_up_ply = Some(at_ply);
                            self.cards[row].face_up_kind = kind;
                        }
                    }
                    None => {
                        self.cards.push(CardRecord {
                            id: card.id,
                            owner: p,
                            rank: card.rank,
                            entered_as_base: card.entered_as_base,
                            entered_ply: at_ply,
                            face_up_ply: None,
                            face_up_kind: FaceUpKind::Never,
                            death_ply: None,
                            died_face_up: false,
                            ever_paired: false,
                        });
                        self.live.push(LiveCard {
                            id: card.id,
                            row: self.cards.len() - 1,
                            face_up: card.face_up,
                            seen: true,
                        });
                    }
                }
            }
        }
        let mut i = 0;
        while i < self.live.len() {
            if self.live[i].seen {
                i += 1;
                continue;
            }
            // `swap_remove` is safe here precisely because the vector holds row *indices*
            // into `cards` rather than positions in itself.
            let gone = self.live.swap_remove(i);
            self.cards[gone.row].death_ply = Some(at_ply);
            self.cards[gone.row].died_face_up = gone.face_up;
        }
    }

    /// Mark both members of a declared pair, looked up before the apply.
    fn note_paired(&mut self, ids: [CardId; 2]) {
        for id in ids {
            if let Some(live) = self.live.iter().find(|l| l.id == id) {
                self.cards[live.row].ever_paired = true;
            }
        }
    }

    /// Record what an action was, reading the board *before* it is applied — a `Flip` names
    /// a slot, and the card's rank has to be looked up while it is still there.
    fn note_action(&mut self, state: &GameState, action: Action) {
        let me = state.to_move.idx();
        match action {
            Action::Play { rank, lane } => {
                self.plays_by_lane[me][lane as usize] += 1;
                self.plays_by_rank[me][rank.index()] += 1;
            }
            Action::Attack { lane, .. } => self.attacks_by_lane[me][lane as usize] += 1,
            Action::DeclarePair { lane, slot_a, .. } => {
                self.pairs_declared[me] += 1;
                // Both members are the same rank by §5, so either slot names the pair.
                if let Some(card) = state.at(lane as usize, state.to_move, slot_a as usize) {
                    self.pairs_by_rank[me][card.rank.index()] += 1;
                }
            }
            Action::Flip { lane, slot } => {
                if let Some(card) = state.at(lane as usize, state.to_move, slot as usize) {
                    self.flips_by_rank[me][card.rank.index()] += 1;
                    self.flip_ply_sum[me][card.rank.index()] += state.ply as u64;
                    if card.entered_as_base {
                        self.base_flips_by_rank[me][card.rank.index()] += 1;
                    }
                }
            }
            _ => {}
        }
    }

    /// Count the turns this action ended that still had actions left in them.
    ///
    /// §4 has no pass, so a turn ending early is the engine's doing, not the player's —
    /// there is no action to intercept in [`GameStats::note_action`]. It has to be read off
    /// the ply counter instead, comparing the position before and after one `apply`.
    ///
    /// Two ways a turn can end early here, and both are counted:
    ///
    /// 1. **The acting player's own turn.** They spent an action but the ply still advanced,
    ///    which means what remained of their allowance had nothing to spend itself on.
    /// 2. **Turns nobody was offered.** If the ply advanced by more than one, the players in
    ///    between were handed a turn with no legal action in it and the engine passed
    ///    straight through. They alternate, starting with the opponent.
    fn note_turn_ends(
        &mut self,
        acting: Player,
        ply_before: u32,
        allowance_before: u32,
        costs_an_action: bool,
        state: &GameState,
    ) {
        let plies_advanced = state.ply - ply_before;
        if plies_advanced == 0 {
            return; // mid-turn, or a free sub-decision
        }
        if allowance_before > u32::from(costs_an_action) {
            self.stuck_turns[acting.idx()] += 1;
        }
        let mut skipped = acting.other();
        for _ in 1..plies_advanced {
            self.stuck_turns[skipped.idx()] += 1;
            skipped = skipped.other();
        }
    }

    /// Record board facts that have to be sampled continuously rather than at the end.
    fn note_state(&mut self, state: &GameState) {
        self.note_start_hands(state);
        for lane in &state.lanes {
            for side in &lane.sides {
                self.max_side_occupancy = self.max_side_occupancy.max(side.len());
            }
        }
        if self.ply_at_unlock.is_none() && state.base_unlocked {
            self.ply_at_unlock = Some(state.ply);
            self.hand_at_unlock = [state.hands[0].len() as u32, state.hands[1].len() as u32];
            for p in Player::BOTH {
                for rank in &state.hands[p.idx()] {
                    self.unlock_hand[p.idx()][rank.index()] += 1;
                }
            }
        }
    }

    fn finish(&mut self, state: &GameState) {
        self.outcome = state.outcome;
        self.plies = state.ply + 1;
        self.draws_taken = state.draws_taken;
        self.hand_at_end = [state.hands[0].len() as u32, state.hands[1].len() as u32];
        for p in Player::BOTH {
            self.unflipped_at_end[p.idx()] = state
                .cards_of(p)
                .filter(|(_, _, card)| !card.face_up)
                .count() as u32;
            for (_, _, card) in state.cards_of(p).filter(|(_, _, c)| !c.face_up) {
                self.face_down_at_end_by_rank[p.idx()][card.rank.index()] += 1;
            }
        }
        // The per-rank fate counters are a fold over the card log rather than a second tally
        // kept beside it, so there is one mechanism to be wrong rather than two that can
        // disagree.
        for card in &self.cards {
            let (p, r) = (card.owner.idx(), card.rank.index());
            match card.face_up_kind {
                FaceUpKind::Trap => self.triggers_sprung_by_rank[p][r] += 1,
                FaceUpKind::Cascade => self.flipped_by_cascade_by_rank[p][r] += 1,
                FaceUpKind::Chose | FaceUpKind::Never => {}
            }
            if card.death_ply.is_some() {
                if card.died_face_up {
                    self.died_face_up_by_rank[p][r] += 1;
                } else {
                    self.died_face_down_by_rank[p][r] += 1;
                }
            }
        }
        self.live.clear();
        self.live.shrink_to_fit();
    }

    /// Share of this player's plays that landed in their busiest `lanes_to_win` lanes.
    ///
    /// The H3 measurement. It has no meaningful absolute scale — picking the busiest lanes
    /// after the fact inflates it even for a uniform player — so read it only against the
    /// random rung's value in the same table. `None` when the player played no cards.
    pub fn lane_concentration(&self, p: Player, lanes_to_win: usize) -> Option<f64> {
        top_share(&self.plays_by_lane[p.idx()], lanes_to_win)
    }

    /// The same measure over attacks rather than plays.
    pub fn attack_concentration(&self, p: Player, lanes_to_win: usize) -> Option<f64> {
        top_share(&self.attacks_by_lane[p.idx()], lanes_to_win)
    }
}

/// Fraction of the total held by the `k` largest entries.
fn top_share(counts: &[u32], k: usize) -> Option<f64> {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return None;
    }
    let mut sorted: Vec<u32> = counts.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let top: u32 = sorted.iter().take(k).sum();
    Some(top as f64 / total as f64)
}

/// One instrumented game, driven a network evaluation at a time.
///
/// The same loop [`play_instrumented`] always ran, turned inside out so a caller can keep
/// several games in flight and evaluate their positions in one batch (`PLAN.md` §4.2d). Only
/// `netmcts` suspends — [`Agent::begin_decision`] returns `None` for everything else, and
/// those agents decide inline exactly as before.
///
/// There is one implementation: [`play_instrumented`] is this type driven one row at a time.
pub struct MatchGame {
    state: GameState,
    stats: GameStats,
    /// Indexed by [`Player::idx`], so the agent to ask is `agents[state.to_move.idx()]`.
    agents: [Box<dyn Agent>; 2],
    /// The legal actions of the decision being searched — what `end_decision` indexes.
    legal: Vec<Action>,
    search: Option<SearchInProgress>,
    finished: bool,
}

impl MatchGame {
    pub fn new(
        config: GameConfig,
        seed: u64,
        p0: Box<dyn Agent>,
        p1: Box<dyn Agent>,
    ) -> MatchGame {
        let state = GameState::new(config, seed);
        let mut stats = GameStats::new(&config, seed);
        stats.note_setup(&state);
        stats.note_state(&state);
        MatchGame {
            state,
            stats,
            agents: [p0, p1],
            legal: Vec::new(),
            search: None,
            finished: false,
        }
    }

    /// Play on until the game needs a network evaluation, or it is over.
    pub fn advance(&mut self, obs: &mut [f32], mask: &mut [bool]) -> SearchStep {
        loop {
            if let Some(search) = &mut self.search {
                match search.advance(obs, mask) {
                    SearchStep::NeedsEval => return SearchStep::NeedsEval,
                    SearchStep::Done => {
                        let search = self.search.take().expect("just matched");
                        let seat = self.state.to_move.idx();
                        let action = self.agents[seat].end_decision(search, &self.legal);
                        self.play(action);
                    }
                }
            } else if self.finished {
                return SearchStep::Done;
            } else if self.state.outcome.is_over() {
                self.stats.finish(&self.state);
                self.finished = true;
                return SearchStep::Done;
            } else {
                self.legal = self.state.legal_actions();
                let seat = self.state.to_move.idx();
                match self.agents[seat].begin_decision(&self.state, &self.legal) {
                    Some(search) => self.search = Some(search),
                    None => {
                        let action = self.agents[seat].choose(&self.state, &self.legal);
                        self.play(action);
                    }
                }
            }
        }
    }

    /// Answer the evaluation [`Self::advance`] suspended on. `mask` must be the buffer it
    /// filled.
    pub fn supply(&mut self, logits: &[f32], value: f32, mask: &mut [bool]) {
        self.search
            .as_mut()
            .expect("supply without a search in progress")
            .supply(logits, value, mask);
    }

    /// The network the suspended search is waiting on — how a driver groups games into
    /// batches when the two agents hold different checkpoints, which in a gate they do.
    pub fn pending_evaluator(&self) -> Option<&Arc<MlpEvaluator>> {
        self.search.as_ref().map(|s| s.evaluator())
    }

    pub fn finish(self) -> GameStats {
        debug_assert!(self.finished, "finish before the game ended");
        self.stats
    }

    fn play(&mut self, action: Action) {
        let acting = self.state.to_move;
        let ply_before = self.state.ply;
        let allowance_before = self.state.actions_remaining;
        let costs = action.costs_an_action();

        // Both of these name cards by slot, and slots compact on death — so the ids have to
        // be read out while the board still matches the action.
        let flipped = match action {
            Action::Flip { lane, slot } => self
                .state
                .at(lane as usize, acting, slot as usize)
                .map(|c| c.id),
            _ => None,
        };
        let paired = match action {
            Action::DeclarePair {
                lane,
                slot_a,
                slot_b,
            } => {
                let a = self.state.at(lane as usize, acting, slot_a as usize).map(|c| c.id);
                let b = self.state.at(lane as usize, acting, slot_b as usize).map(|c| c.id);
                a.zip(b)
            }
            _ => None,
        };

        self.stats.note_action(&self.state, action);
        self.state.apply_trusted(action);
        self.stats.decisions += 1;
        self.stats.note_state(&self.state);
        self.stats
            .note_turn_ends(acting, ply_before, allowance_before, costs, &self.state);
        self.stats
            .note_cards(ply_before, acting, flipped, &self.state);
        if let Some((a, b)) = paired {
            self.stats.note_paired([a, b]);
        }
    }
}

/// Per-evaluator working buffers, so a driver holding two checkpoints does not allocate one
/// per evaluation. Keyed by pointer, which `nn::evaluator_for`'s cache makes checkpoint
/// identity.
#[derive(Default)]
pub struct ScratchPool {
    entries: Vec<(usize, Scratch)>,
}

impl ScratchPool {
    pub fn get(&mut self, evaluator: &Arc<MlpEvaluator>) -> &mut Scratch {
        let key = Arc::as_ptr(evaluator) as usize;
        let at = match self.entries.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                self.entries.push((key, evaluator.scratch()));
                self.entries.len() - 1
            }
        };
        &mut self.entries[at].1
    }
}

/// Play one instrumented game between two agents.
pub fn play_instrumented(
    config: GameConfig,
    seed: u64,
    p0: Box<dyn Agent>,
    p1: Box<dyn Agent>,
) -> GameStats {
    let (od, ad) = (obs_dim(&config), action_dim(&config));
    let mut obs = vec![0.0f32; od];
    let mut mask = vec![false; ad];
    let mut logits = vec![0.0f32; ad];
    let mut pool = ScratchPool::default();

    let mut game = MatchGame::new(config, seed, p0, p1);
    while game.advance(&mut obs, &mut mask) == SearchStep::NeedsEval {
        let evaluator = game
            .pending_evaluator()
            .expect("a suspended search names its network")
            .clone();
        let value = evaluator.eval_masked_with(&obs, &mask, &mut logits, pool.get(&evaluator));
        game.supply(&logits, value, &mut mask);
    }
    game.finish()
}

/// Play one instrumented game between two [`AgentSpec`]s, building both from `seed`.
pub fn play_spec_game(
    config: GameConfig,
    seed: u64,
    p0: AgentSpec,
    p1: AgentSpec,
) -> GameStats {
    let a = p0.build(seed, AGENT_STREAM[0]);
    let b = p1.build(seed, AGENT_STREAM[1]);
    play_instrumented(config, seed, a, b)
}

// ============================================================== aggregation ==

/// Aggregated statistics for one agent playing one role in a set of games.
///
/// Kept per *agent*, not per seat, so the numbers are about how that agent plays rather than
/// about which side of the table it sat on.
#[derive(Clone, Debug, Default)]
pub struct AgentBehaviour {
    pub games: usize,
    pub hand_at_unlock: Vec<u32>,
    /// Hand size at the unlock, split by how the game then ended.
    ///
    /// The sharp test of `FINDINGS.md` H2. Comparing hand size *across* agents confounds the
    /// hypothesis with agent strength; comparing winners against losers **within one agent's
    /// self-play** holds strength fixed and asks the question directly: among games this
    /// agent played against itself, did the side holding more cards at pile-empty win?
    pub hand_at_unlock_won: Vec<u32>,
    pub hand_at_unlock_lost: Vec<u32>,
    pub hand_at_end: Vec<u32>,
    pub plays: u64,
    pub flips: u64,
    pub base_flips: u64,
    pub base_flips_by_rank: [u64; Rank::COUNT],
    pub stuck_turns: u64,
    pub pairs: u64,
    pub unflipped_at_end: u64,
    pub plays_by_rank: [u64; Rank::COUNT],
    pub flips_by_rank: [u64; Rank::COUNT],
    pub flip_ply_sum: [u64; Rank::COUNT],
    pub lane_concentration: Vec<f64>,
    pub attack_concentration: Vec<f64>,
    /// The fate of every card with a death trigger — canonically just the 3, the one rank
    /// whose power needs darkness. See [`GameStats`].
    pub triggers_sprung_by_rank: [u64; Rank::COUNT],
    pub flipped_by_cascade_by_rank: [u64; Rank::COUNT],
    pub face_down_at_end_by_rank: [u64; Rank::COUNT],
}

impl AgentBehaviour {
    fn absorb(&mut self, stats: &GameStats, p: Player, lanes_to_win: usize) {
        let i = p.idx();
        self.games += 1;
        if stats.ply_at_unlock.is_some() {
            self.hand_at_unlock.push(stats.hand_at_unlock[i]);
            match stats.outcome {
                Outcome::Win(w) if w == p => self.hand_at_unlock_won.push(stats.hand_at_unlock[i]),
                Outcome::Win(_) => self.hand_at_unlock_lost.push(stats.hand_at_unlock[i]),
                // A draw is evidence for neither side, so it is left out rather than
                // counted half in each bucket, which would blur the very difference the
                // split exists to show.
                _ => {}
            }
        }
        self.hand_at_end.push(stats.hand_at_end[i]);
        self.plays += stats.plays_by_lane[i].iter().map(|&v| v as u64).sum::<u64>();
        self.flips += stats.flips_by_rank[i].iter().map(|&v| v as u64).sum::<u64>();
        self.base_flips += stats.base_flips_by_rank[i].iter().map(|&v| v as u64).sum::<u64>();
        self.stuck_turns += stats.stuck_turns[i] as u64;
        self.pairs += stats.pairs_declared[i] as u64;
        self.unflipped_at_end += stats.unflipped_at_end[i] as u64;
        for r in 0..Rank::COUNT {
            self.plays_by_rank[r] += stats.plays_by_rank[i][r] as u64;
            self.flips_by_rank[r] += stats.flips_by_rank[i][r] as u64;
            self.base_flips_by_rank[r] += stats.base_flips_by_rank[i][r] as u64;
            self.flip_ply_sum[r] += stats.flip_ply_sum[i][r];
            self.triggers_sprung_by_rank[r] += stats.triggers_sprung_by_rank[i][r] as u64;
            self.flipped_by_cascade_by_rank[r] += stats.flipped_by_cascade_by_rank[i][r] as u64;
            self.face_down_at_end_by_rank[r] += stats.face_down_at_end_by_rank[i][r] as u64;
        }
        if let Some(v) = stats.lane_concentration(p, lanes_to_win) {
            self.lane_concentration.push(v);
        }
        if let Some(v) = stats.attack_concentration(p, lanes_to_win) {
            self.attack_concentration.push(v);
        }
    }

    /// Merge another shard's counts. Used to combine per-thread results.
    pub fn merge(&mut self, other: &AgentBehaviour) {
        self.games += other.games;
        self.hand_at_unlock.extend_from_slice(&other.hand_at_unlock);
        self.hand_at_unlock_won
            .extend_from_slice(&other.hand_at_unlock_won);
        self.hand_at_unlock_lost
            .extend_from_slice(&other.hand_at_unlock_lost);
        self.hand_at_end.extend_from_slice(&other.hand_at_end);
        self.plays += other.plays;
        self.flips += other.flips;
        self.base_flips += other.base_flips;
        self.stuck_turns += other.stuck_turns;
        self.pairs += other.pairs;
        self.unflipped_at_end += other.unflipped_at_end;
        for r in 0..Rank::COUNT {
            self.plays_by_rank[r] += other.plays_by_rank[r];
            self.flips_by_rank[r] += other.flips_by_rank[r];
            self.base_flips_by_rank[r] += other.base_flips_by_rank[r];
            self.flip_ply_sum[r] += other.flip_ply_sum[r];
            self.triggers_sprung_by_rank[r] += other.triggers_sprung_by_rank[r];
            self.flipped_by_cascade_by_rank[r] += other.flipped_by_cascade_by_rank[r];
            self.face_down_at_end_by_rank[r] += other.face_down_at_end_by_rank[r];
        }
        self.lane_concentration
            .extend_from_slice(&other.lane_concentration);
        self.attack_concentration
            .extend_from_slice(&other.attack_concentration);
    }

    pub fn mean_hand_at_unlock(&self) -> f64 {
        mean_u32(&self.hand_at_unlock)
    }

    /// Mean hand size at the unlock in games this agent went on to win, and in games it went
    /// on to lose. The H2 test — see [`AgentBehaviour::hand_at_unlock_won`].
    pub fn hand_at_unlock_by_result(&self) -> (f64, f64) {
        (
            mean_u32(&self.hand_at_unlock_won),
            mean_u32(&self.hand_at_unlock_lost),
        )
    }

    /// Half-width of a 95% interval on the winner-minus-loser difference in hand size at the
    /// unlock. Two independent means, so the variances add.
    pub fn hand_at_unlock_gap_ci95(&self) -> f64 {
        let se = |v: &Vec<u32>| {
            if v.len() < 2 {
                return f64::INFINITY;
            }
            let m = mean_u32(v);
            let var = v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
            var / v.len() as f64
        };
        1.96 * (se(&self.hand_at_unlock_won) + se(&self.hand_at_unlock_lost)).sqrt()
    }
    pub fn mean_lane_concentration(&self) -> f64 {
        mean_f64(&self.lane_concentration)
    }
    pub fn mean_attack_concentration(&self) -> f64 {
        mean_f64(&self.attack_concentration)
    }
    pub fn plays_per_game(&self) -> f64 {
        self.plays as f64 / self.games.max(1) as f64
    }
    pub fn stuck_turns_per_game(&self) -> f64 {
        self.stuck_turns as f64 / self.games.max(1) as f64
    }

    /// Fraction of cards this agent played from hand that it ever turned face-up.
    ///
    /// The H4 measurement. Base-card flips are subtracted out: a base card was never played
    /// from hand, and counting it would push the ratio above 1 in an endgame where the base
    /// cards come up.
    pub fn flip_rate(&self) -> f64 {
        if self.plays == 0 {
            return f64::NAN;
        }
        self.flips.saturating_sub(self.base_flips) as f64 / self.plays as f64
    }

    /// Fraction of the cards of one rank this agent played from hand that it then turned
    /// face-up. `None` if it never played that rank.
    ///
    /// Base-card flips are subtracted from the numerator, so the ratio is exactly "of the
    /// `rank`s I chose to play, how many did I choose to flip". The sharper form of the H4
    /// question than mean flip ply: *whether* a rank is ever turned up, not just when.
    ///
    /// One rank needs a caveat, and it is the interesting one. A face-down 3 that is killed
    /// returns **face-up** through its Trap (`game_rules.md` §6), and that is not a `Flip`
    /// action — so it is counted nowhere here. This ratio therefore measures *voluntary*
    /// flips of a 3, which is the right quantity for the question but not the same as "how
    /// often a 3 ends up face-up".
    pub fn flip_rate_for(&self, rank: Rank) -> Option<f64> {
        let r = rank.index();
        if self.plays_by_rank[r] == 0 {
            return None;
        }
        let voluntary = self.flips_by_rank[r].saturating_sub(self.base_flips_by_rank[r]);
        Some(voluntary as f64 / self.plays_by_rank[r] as f64)
    }

    /// What became of every 3 this agent played from hand, as fractions that sum to 1:
    /// `(flipped by choice, flipped by a 5, Trap sprang, still face-down at the end)`.
    ///
    /// `None` if it never played a 3. The four are exhaustive because a face-down 3 cannot
    /// die — it springs instead — so a played 3 is always still on the board.
    ///
    /// The last two are the point. `flip_rate_for` says how often the agent *declines* to
    /// spend a 3's concealment, but declining is only worth something if the Trap then
    /// fires; a 3 held all game and never killed bought nothing, and is indistinguishable
    /// from a blank card the agent happened not to flip.
    ///
    /// The face-down-at-end count includes **base** 3s, which were never played from hand,
    /// so it is the one term that can push the sum past 1. At the default config each
    /// player has two 3s and three base cards, so this is a small and infrequent bias.
    pub fn three_fates(&self) -> Option<(f64, f64, f64, f64)> {
        self.card_fates(Rank::THREE)
    }

    /// The four fates of one played rank: flipped by choice, flipped by a cascade, sprang a
    /// death trigger, still face-down at the end.
    ///
    /// Generalised from `three_fates` (`MODULAR_RULES.md` §11 step 7) so a new death-trigger
    /// power is measurable the day it exists. For a rank with no death trigger the third
    /// term is always zero, which is correct rather than missing — it says the card never
    /// came back.
    pub fn card_fates(&self, rank: Rank) -> Option<(f64, f64, f64, f64)> {
        let r = rank.index();
        let played = self.plays_by_rank[r];
        if played == 0 {
            return None;
        }
        let n = played as f64;
        let voluntary = self.flips_by_rank[r].saturating_sub(self.base_flips_by_rank[r]);
        Some((
            voluntary as f64 / n,
            self.flipped_by_cascade_by_rank[r] as f64 / n,
            self.triggers_sprung_by_rank[r] as f64 / n,
            self.face_down_at_end_by_rank[r] as f64 / n,
        ))
    }

    /// Threes played from hand per game — the denominator `three_fates` divides by.
    pub fn threes_played_per_game(&self) -> f64 {
        self.played_per_game(Rank::THREE)
    }

    /// Cards of one rank played from hand per game.
    pub fn played_per_game(&self, rank: Rank) -> f64 {
        if self.games == 0 {
            return f64::NAN;
        }
        self.plays_by_rank[rank.index()] as f64 / self.games as f64
    }

    /// Traps sprung per game. The absolute frequency behind `three_fates`, because a rate
    /// out of a rarely played card can look large while describing almost nothing.
    pub fn traps_per_game(&self) -> f64 {
        self.triggers_per_game(Rank::THREE)
    }

    /// Death triggers of one rank fired per game.
    pub fn triggers_per_game(&self, rank: Rank) -> f64 {
        if self.games == 0 {
            return f64::NAN;
        }
        self.triggers_sprung_by_rank[rank.index()] as f64 / self.games as f64
    }

    /// Every rank whose power has a death trigger under `config`, with its fates. The
    /// per-card instrumentation a modded ruleset needs, in one call.
    pub fn death_trigger_fates(
        &self,
        config: &GameConfig,
    ) -> Vec<(Rank, (f64, f64, f64, f64))> {
        Rank::ALL
            .into_iter()
            .filter(|r| r.index() <= config.max_rank_index)
            .filter(|r| config.power(*r).has_death_trigger())
            .filter_map(|r| self.card_fates(r).map(|f| (r, f)))
            .collect()
    }

    /// Mean ply at which this agent turns a given rank face-up. `None` if it never did.
    pub fn mean_flip_ply(&self, rank: Rank) -> Option<f64> {
        let n = self.flips_by_rank[rank.index()];
        if n == 0 {
            None
        } else {
            Some(self.flip_ply_sum[rank.index()] as f64 / n as f64)
        }
    }
}

pub(crate) fn mean_u32(values: &[u32]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64
}

pub(crate) fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Everything a set of games between two agents produced.
#[derive(Clone, Debug)]
pub struct MatchStats {
    pub config: GameConfig,
    pub agents: [AgentSpec; 2],
    pub games: usize,

    /// Wins by agent index, then draws.
    pub wins: [usize; 2],
    pub draws: usize,
    pub draws_stalemate: usize,
    pub draws_mutual_lane_win: usize,
    pub draws_ply_limit: usize,

    /// Score for the agent that sat as P0, over the games in which it did.
    pub p0_seat_score: f64,
    /// Sum of squared per-game P0 scores, for the interval on [`MatchStats::first_player_score`].
    pub p0_seat_score_sq: f64,
    pub p0_seat_games: usize,

    pub lengths: Vec<u32>,
    pub unlock_plies: Vec<u32>,
    pub max_side_occupancy: usize,
    pub behaviour: [AgentBehaviour; 2],
    pub elapsed_secs: f64,
}

impl MatchStats {
    pub(crate) fn empty(config: GameConfig, agents: [AgentSpec; 2]) -> MatchStats {
        MatchStats {
            config,
            agents,
            games: 0,
            wins: [0, 0],
            draws: 0,
            draws_stalemate: 0,
            draws_mutual_lane_win: 0,
            draws_ply_limit: 0,
            p0_seat_score: 0.0,
            p0_seat_score_sq: 0.0,
            p0_seat_games: 0,
            lengths: Vec::new(),
            unlock_plies: Vec::new(),
            max_side_occupancy: 0,
            behaviour: [AgentBehaviour::default(), AgentBehaviour::default()],
            elapsed_secs: 0.0,
        }
    }

    /// Fold in one finished game. `seats[i]` is the agent index that played as `Player::i`.
    pub(crate) fn absorb(&mut self, stats: &GameStats, seats: [usize; 2]) {
        self.games += 1;
        match stats.outcome {
            Outcome::Win(p) => self.wins[seats[p.idx()]] += 1,
            Outcome::Draw(reason) => {
                self.draws += 1;
                match reason {
                    DrawReason::Stalemate => self.draws_stalemate += 1,
                    DrawReason::MutualLaneWin => self.draws_mutual_lane_win += 1,
                    DrawReason::PlyLimit => self.draws_ply_limit += 1,
                }
            }
            Outcome::Ongoing => unreachable!("an unfinished game reached the aggregator"),
        }

        self.p0_seat_games += 1;
        let p0_score = stats.outcome.value_for(Player::P0) as f64;
        self.p0_seat_score += p0_score;
        self.p0_seat_score_sq += p0_score * p0_score;

        self.lengths.push(stats.plies);
        if let Some(ply) = stats.ply_at_unlock {
            self.unlock_plies.push(ply);
        }
        self.max_side_occupancy = self.max_side_occupancy.max(stats.max_side_occupancy);
        for p in Player::BOTH {
            self.behaviour[seats[p.idx()]].absorb(stats, p, self.config.lanes_to_win);
        }
    }

    /// Merge another shard. Used to combine per-thread results.
    pub(crate) fn merge(&mut self, other: &MatchStats) {
        self.games += other.games;
        self.wins[0] += other.wins[0];
        self.wins[1] += other.wins[1];
        self.draws += other.draws;
        self.draws_stalemate += other.draws_stalemate;
        self.draws_mutual_lane_win += other.draws_mutual_lane_win;
        self.draws_ply_limit += other.draws_ply_limit;
        self.p0_seat_score += other.p0_seat_score;
        self.p0_seat_score_sq += other.p0_seat_score_sq;
        self.p0_seat_games += other.p0_seat_games;
        self.lengths.extend_from_slice(&other.lengths);
        self.unlock_plies.extend_from_slice(&other.unlock_plies);
        self.max_side_occupancy = self.max_side_occupancy.max(other.max_side_occupancy);
        self.behaviour[0].merge(&other.behaviour[0]);
        self.behaviour[1].merge(&other.behaviour[1]);
    }

    /// Score for agent 0: 1 per win, 0.5 per draw.
    pub fn score(&self) -> f64 {
        if self.games == 0 {
            return 0.5;
        }
        (self.wins[0] as f64 + 0.5 * self.draws as f64) / self.games as f64
    }

    /// Half-width of a 95% confidence interval on [`MatchStats::score`].
    ///
    /// Draws contribute 0.25 to a single game's variance rather than 0, so counting them as
    /// half-wins without adjusting the variance would understate the interval.
    pub fn score_ci95(&self) -> f64 {
        if self.games < 2 {
            return f64::NAN;
        }
        let n = self.games as f64;
        let mean = self.score();
        let sum_sq = self.wins[0] as f64 + 0.25 * self.draws as f64;
        1.96 * ((sum_sq / n - mean * mean).max(0.0) / n).sqrt()
    }

    /// Score of whoever sat first, pooled over both colour assignments — the first-player
    /// advantage at this level of play (`FINDINGS.md` H8).
    pub fn first_player_score(&self) -> f64 {
        if self.p0_seat_games == 0 {
            return 0.5;
        }
        self.p0_seat_score / self.p0_seat_games as f64
    }

    /// Half-width of a 95% interval on [`MatchStats::first_player_score`].
    ///
    /// **Deliberately conservative.** Colour-paired deals mean each deal is played twice
    /// with the seats swapped, which cancels deal luck out of the *agent* score but not out
    /// of the first-player score: both games of a pair deal P0 the identical cards. So the
    /// effective sample is the number of distinct deals, half the game count, and the
    /// interval is computed on that. Treating 400 paired games as 400 independent
    /// observations would understate the interval by about √2 — enough to turn a
    /// null result into a spurious one at these effect sizes.
    pub fn first_player_score_ci95(&self) -> f64 {
        let deals = self.p0_seat_games / 2;
        if deals < 2 {
            return f64::NAN;
        }
        let n = self.p0_seat_games as f64;
        let mean = self.first_player_score();
        let variance = (self.p0_seat_score_sq / n - mean * mean).max(0.0);
        1.96 * (variance / deals as f64).sqrt()
    }

    pub fn draw_rate(&self) -> f64 {
        self.draws as f64 / self.games.max(1) as f64
    }

    /// How often the quiet-ply rule ended the game. `FINDINGS.md` F1.2 asks for exactly this
    /// number against non-random agents, because random play could never produce it.
    pub fn stalemate_rate(&self) -> f64 {
        self.draws_stalemate as f64 / self.games.max(1) as f64
    }

    pub fn mean_plies(&self) -> f64 {
        mean_u32(&self.lengths)
    }

    pub fn games_per_sec(&self) -> f64 {
        if self.elapsed_secs <= 0.0 {
            f64::INFINITY
        } else {
            self.games as f64 / self.elapsed_secs
        }
    }

    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{} vs {} — {} games\n",
            self.agents[0], self.agents[1], self.games
        ));
        out.push_str(&format!("  config: {}\n", self.config.summary()));
        out.push_str(&format!(
            "  score for {}: {:.4} +/- {:.4} (95% CI) — W{} L{} D{}\n",
            self.agents[0],
            self.score(),
            self.score_ci95(),
            self.wins[0],
            self.wins[1],
            self.draws,
        ));
        out.push_str(&format!(
            "  first-player score: {:.4} +/- {:.4}  ·  draws {:.1}% (stalemate {:.1}%, mutual lane win {}, ply cap {})\n",
            self.first_player_score(),
            self.first_player_score_ci95(),
            100.0 * self.draw_rate(),
            100.0 * self.stalemate_rate(),
            self.draws_mutual_lane_win,
            self.draws_ply_limit,
        ));
        out.push_str(&format!(
            "  mean plies {:.1} · max cards on one side of one lane {} · {:.1} games/sec\n",
            self.mean_plies(),
            self.max_side_occupancy,
            self.games_per_sec(),
        ));
        for i in 0..2 {
            let b = &self.behaviour[i];
            let (won, lost) = b.hand_at_unlock_by_result();
            out.push_str(&format!(
                "  {:<14} hand@unlock {:.2} (won {:.2} vs lost {:.2}, gap {:+.2} +/- {:.2}) · \
                 plays/game {:.1} · flip rate {:.2} · lane conc {:.3} · attack conc {:.3} · \
                 stuck/game {:.2}\n",
                self.agents[i].name(),
                b.mean_hand_at_unlock(),
                won,
                lost,
                won - lost,
                b.hand_at_unlock_gap_ci95(),
                b.plays_per_game(),
                b.flip_rate(),
                b.mean_lane_concentration(),
                b.mean_attack_concentration(),
                b.stuck_turns_per_game(),
            ));
        }
        out
    }
}
