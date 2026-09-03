//! Observation and action encoding — the bridge between a [`GameState`] and a tensor.
//!
//! `CLAUDE.md`: "The Rust engine is the sole authority on legality. Never reimplement rules
//! logic in Python." That applies to encoding too, and more sharply: if Python owned a
//! second copy of the feature layout, the trained function and the evaluated function could
//! drift apart silently, which is the worst failure mode available to this phase. So the
//! encoder lives here, Python reaches it through PyO3, and both sides stamp
//! [`obs_layout_hash`] / [`action_layout_hash`] into every checkpoint so a mismatch is a
//! load error rather than a mysteriously bad agent.
//!
//! # The observation is a function of the information set
//!
//! Every feature below is something the observer is entitled to know. That is not a comment
//! — `engine/tests/encoding.rs::phase3_observation_is_a_function_of_the_information_set`
//! asserts it as exact f32 equality between the real state and a determinized world, which
//! is in the same information set by construction. The field list is taken from
//! `bindings/src/lib.rs::observation`, which was already the authoritative per-observer
//! projection.
//!
//! # Layout
//!
//! Two blocks, in this order. `L` is `config.lanes`, `S` is `config.encoding_slots`, `R` is
//! `config.rank_count()`.
//!
//! ```text
//! offset 0                     board:   L × 2 × S × SLOT_FEATURES   (3·2·16·33 = 3168)
//! offset board_len             scalars: SCALAR_FEATURES             (132)
//! total                                                             (3300)
//! ```
//!
//! The board's side axis is ordered **`[observer, opponent]`**, so the tensor is always
//! from the observer's point of view and the network never has to learn a seat convention.
//! Per-slot features, in order, are listed in [`SLOT_FEATURE_NAMES`]; the scalar block's
//! fields and widths are listed in [`scalar_fields`].
//!
//! # Action encoding
//!
//! `DESIGN.md` §4's original table keyed `FLIP` and `PAIR` by *rank*. That is lossy against
//! this engine in two places, and `action.rs`'s own module header says so: two same-rank
//! face-down cards can carry different damage, and with three same-rank cards in a lane
//! *which two* you pair changes the pair's damage and attack budget. In an AlphaZero loop
//! the policy target is a visit distribution over engine actions, so two engine actions
//! sharing a logit would force an invented rule for folding their visits and another for
//! which one to play — both arbitrary, both distorting the policy Phase 4 reads. This
//! encoding is exact and slot-keyed instead. See [`action_blocks`] for the table.

use crate::action::{Action, Phase, Side};
use crate::card::Card;
use crate::config::GameConfig;
use crate::player::Player;
use crate::rank::Rank;
use crate::state::{GameState, Pending};

// ===================================================================== the board block ==

/// Per-slot features, in encoding order. The length of this list is [`SLOT_FEATURES`].
///
/// Kept as names rather than as a bare count because [`obs_layout_hash`] hashes them: a
/// reordering that left the width unchanged would otherwise be invisible to the checkpoint
/// check, and it is exactly the kind of edit that silently breaks a trained network.
pub const SLOT_FEATURE_NAMES: &[&str] = &[
    "occupied",
    "rank_onehot",       // R wide; all zero when the observer may not know the rank
    "rank_unknown",      //
    "face_up",           //
    "is_base",           //
    "entered_as_base",   // DESIGN.md §3 — distinct from is_base
    "damage_onehot",     // DAMAGE_BUCKETS wide
    "max_hp_onehot",     // MAX_HP_BUCKETS wide: 2 HP or 3 HP
    "frozen",            //
    "allowance_onehot",  // ALLOWANCE_BUCKETS wide; a fresh Ace has 2
    "attacks_used_frac", //
    "can_attack_now",    //
    "paired",            //
    "is_mine",           //
];

/// Damage one-hot width. A card in play always satisfies `damage < max_hp`, and the largest
/// max HP is a face-up Jack's 3 (`game_rules.md` §5), so 0–2 is the whole live range. The
/// fourth bucket is a clamped overflow that the engine's invariants say cannot fire; it
/// exists so a future rule that raised hit points would degrade into a saturated feature
/// rather than an out-of-bounds write.
pub const DAMAGE_BUCKETS: usize = 4;

/// Max-HP one-hot width: 2 HP, or a face-up Jack's 3.
pub const MAX_HP_BUCKETS: usize = 2;

/// Attack-allowance one-hot width. Normally 1; a freshly flipped Ace has 2 (§6). The last
/// bucket is a clamped "3 or more" — a King reactivating an Ace *resets* the allowance to 2
/// rather than stacking, so nothing in the current rules reaches it.
pub const ALLOWANCE_BUCKETS: usize = 4;

/// Actions-remaining one-hot width, clamped at the top bucket. `actions_per_turn` is 3, but
/// an Ace grants `+1` and a King can reactivate several Aces in one lane, so the value is
/// not bounded by config — hence "3 or more" rather than an assertion.
pub const ACTIONS_BUCKETS: usize = 4;

/// Phase one-hot width — every variant of [`Phase`], `Terminal` included.
pub const PHASE_COUNT: usize = 7;

/// Floats per slot, for a given rank count.
#[inline]
pub const fn slot_features(config: &GameConfig) -> usize {
    // occupied + rank one-hot + rank_unknown + face_up + is_base + entered_as_base
    1 + config.rank_count() + 1 + 1 + 1 + 1
        // damage + max HP + frozen + allowance + attacks_used_frac + can_attack_now
        + DAMAGE_BUCKETS + MAX_HP_BUCKETS + 1 + ALLOWANCE_BUCKETS + 1 + 1
        // paired + is_mine
        + 1 + 1
}

/// Floats in the board block: `lanes × 2 sides × encoding_slots × slot_features`.
#[inline]
pub const fn board_len(config: &GameConfig) -> usize {
    config.lanes * 2 * config.encoding_slots * slot_features(config)
}

// ==================================================================== the scalar block ==

/// The scalar block, as `(name, width)` pairs in encoding order.
///
/// Everything here is normalised to roughly `[0, 1]`. Widths that depend on the rank count
/// are computed from `config`, so Duel52-mini (`DESIGN.md` §7) shrinks the tensor rather
/// than misaligning it.
pub fn scalar_fields(config: &GameConfig) -> Vec<(&'static str, usize)> {
    let r = config.rank_count();
    vec![
        ("phase_onehot", PHASE_COUNT),
        ("actions_remaining_onehot", ACTIONS_BUCKETS),
        ("is_mine_to_move", 1),
        ("ply_frac", 1),
        ("quiet_frac", 1),
        ("base_unlocked", 1),
        ("observer_is_first_player", 1),
        ("lanes_won", 2),
        ("my_hand_counts", r),
        ("my_hand_size", 1),
        ("opponent_hand_size", 1),
        ("my_pile_size", 1),
        ("opponent_pile_size", 1),
        ("shared_pile", 1),
        ("my_discard_counts", r),
        ("opponent_discard_counts", r),
        ("unseen_counts", r),
        ("removed_size", 1),
        ("removed_revealed", 1),
        ("removed_counts", r),
        // Per pile, ordered [mine, theirs]: do I know a bottomed card, how many, and what
        // is the bottom-most one. `DESIGN.md` §5 — without this the net cannot value a 2.
        ("my_pile_bottom_known_any", 1),
        ("my_pile_bottom_known_count", 1),
        ("my_pile_bottom_rank", r),
        ("their_pile_bottom_known_any", 1),
        ("their_pile_bottom_known_count", 1),
        ("their_pile_bottom_rank", r),
        // Lane aggregates. Cheap, and it saves a dense MLP from rediscovering that slot
        // indices within one lane belong together.
        ("lane_counts", config.lanes * 4),
    ]
}

/// Floats in the scalar block.
pub fn scalar_len(config: &GameConfig) -> usize {
    scalar_fields(config).iter().map(|(_, w)| w).sum()
}

/// Total observation length. ~3300 at the default configuration.
///
/// Note this is ~2.5× `DESIGN.md` §5's original "~1300 floats", which silently assumed the
/// 8-slot board §3 has since abandoned. See `FINDINGS.md` F2.7.
pub fn obs_dim(config: &GameConfig) -> usize {
    board_len(config) + scalar_len(config)
}

// =========================================================== the observation encoder ==

/// Write the observation `observer` is entitled to into `out`.
///
/// `out` must be exactly [`obs_dim`] long. It is fully overwritten, so a caller may reuse
/// one buffer across a batch without clearing it.
///
/// # Panics
///
/// If any side of any lane holds more than `config.encoding_slots` cards. The encoder
/// asserts rather than truncating: a dropped card is a different position, and a network
/// trained on quietly-truncated boards would be wrong in exactly the situations where the
/// board matters most.
pub fn encode_observation(state: &GameState, observer: Player, out: &mut [f32]) {
    let config = &state.config;
    let s = config.encoding_slots;
    let f = slot_features(config);
    assert_eq!(
        out.len(),
        obs_dim(config),
        "observation buffer is {} floats, expected {}",
        out.len(),
        obs_dim(config)
    );
    out.fill(0.0);

    let opponent = observer.other();
    let r = config.rank_count();

    // ------------------------------------------------------------------- the board --
    for lane in 0..config.lanes {
        for (side_idx, owner) in [observer, opponent].into_iter().enumerate() {
            let side = state.lanes[lane].side(owner);
            assert!(
                side.len() <= s,
                "lane {lane} side {owner} holds {} cards, over the encoder's bound of {s}. \
                 Raise `encoding_slots` in the config (see FINDINGS.md F2.7); the encoder \
                 refuses to truncate, because a dropped card is a different position.",
                side.len(),
            );
            for (slot, card) in side.iter().enumerate() {
                let base = ((lane * 2 + side_idx) * s + slot) * f;
                encode_slot(state, observer, card, owner == observer, &mut out[base..base + f]);
            }
        }
    }

    // ----------------------------------------------------------------- the scalars --
    let mut w = Writer::new(&mut out[board_len(config)..]);

    w.one_hot(phase_index(state.phase()), PHASE_COUNT);
    w.one_hot(
        (state.actions_remaining as usize).min(ACTIONS_BUCKETS - 1),
        ACTIONS_BUCKETS,
    );
    w.bit(state.to_move == observer);
    w.push(state.ply as f32 / config.max_plies.max(1) as f32);
    w.push(state.quiet_plies as f32 / config.stalemate_quiet_plies.max(1) as f32);
    w.bit(state.base_unlocked);
    // Which *seat* the observer holds. P0 moves first and takes only two actions on the
    // opening turn (`game_rules.md` §2), so the seat is a real asymmetry — and unlike
    // "P0 is to move" it is not already implied by `is_mine_to_move`.
    w.bit(observer == Player::P0);
    w.push(state.lanes_won_by(observer) as f32 / config.lanes as f32);
    w.push(state.lanes_won_by(opponent) as f32 / config.lanes as f32);

    let deck = config.copies_per_rank as f32;
    w.counts(&state.hand_counts(observer), r, deck);
    w.push(state.hand(observer).len() as f32 / hand_scale(config));
    w.push(state.hand(opponent).len() as f32 / hand_scale(config));
    w.push(state.pile(observer).len() as f32 / pile_scale(config));
    w.push(state.pile(opponent).len() as f32 / pile_scale(config));
    w.bit(state.shared_pile());

    w.counts(&crate::rank_counts(&state.discards[observer.idx()]), r, deck);
    w.counts(&crate::rank_counts(&state.discards[opponent.idx()]), r, deck);

    // Belief. `game_rules.md` §2: this never resolves to certainty, because the removed
    // cards stay indistinguishable from cards in a hand or a base slot.
    w.counts(&state.unseen_counts(observer), r, deck);
    w.push(state.all_removed().count() as f32 / removed_scale(config));
    w.bit(state.removed_revealed);
    if state.removed_revealed {
        // §9b only. The two players' removed multisets are rank-identical there, so
        // reading the observer's own is symmetric.
        w.counts(&crate::rank_counts(&state.removed[observer.idx()]), r, deck);
    } else {
        w.skip(r);
    }

    // Bottomed cards: private, persistent, and the whole value of a 2 (§10a).
    for owner in [observer, opponent] {
        let known = state.piles[state.pile_index(owner)].known_from_bottom(observer);
        let count = known.iter().filter(|k| k.is_some()).count();
        w.bit(count > 0);
        w.push(count as f32 / pile_scale(config));
        match known.iter().flatten().next() {
            Some(rank) => w.one_hot(rank.index(), r),
            None => w.skip(r),
        }
    }

    // Lane aggregates: (mine total, mine face-up, theirs total, theirs face-up).
    for lane in 0..config.lanes {
        for owner in [observer, opponent] {
            let side = state.lanes[lane].side(owner);
            w.push(side.len() as f32 / s as f32);
            w.push(side.iter().filter(|c| c.face_up).count() as f32 / s as f32);
        }
    }

    debug_assert_eq!(
        w.written(),
        scalar_len(config),
        "the scalar block wrote a different number of floats than `scalar_fields` declares"
    );
}

/// One slot's features. `out` is exactly [`slot_features`] long and starts zeroed.
fn encode_slot(state: &GameState, observer: Player, card: &Card, mine: bool, out: &mut [f32]) {
    let r = state.config.rank_count();
    let mut w = Writer::new(out);

    w.push(1.0); // occupied
    // The load-bearing line. A rank the observer may not read contributes *nothing* —
    // not a smoothed prior, not a placeholder index — so there is no channel through
    // which a hidden rank could reach the network.
    let known = card.rank_known_to(observer);
    if known {
        w.one_hot(card.rank.index(), r);
    } else {
        w.skip(r);
    }
    w.bit(!known);
    w.bit(card.face_up);
    w.bit(card.is_base);
    w.bit(card.entered_as_base);

    w.one_hot((card.damage as usize).min(DAMAGE_BUCKETS - 1), DAMAGE_BUCKETS);
    // Max HP is public even on a face-down card, and leaks nothing: §5 makes every
    // face-down card a blank 2-HP card whatever its rank, so a Jack cannot be identified
    // by watching it survive.
    w.one_hot(if card.max_hp() >= 3 { 1 } else { 0 }, MAX_HP_BUCKETS);
    w.bit(card.is_frozen(state.ply));
    w.one_hot(
        (card.attack_allowance as usize).min(ALLOWANCE_BUCKETS - 1),
        ALLOWANCE_BUCKETS,
    );
    w.push(card.attacks_used as f32 / card.attack_allowance.max(1) as f32);
    w.bit(card.can_attack(state.ply));
    w.bit(card.pair_id.is_some());
    w.bit(mine);

    debug_assert_eq!(w.written(), slot_features(&state.config));
}

/// Cursor that writes floats in order and remembers how many it wrote.
///
/// The alternative — computing an offset per field — puts the layout in two places and
/// makes an off-by-one in the middle of the scalar block silently shift everything after
/// it. Here the order in the code *is* the layout, and `written()` is checked against
/// [`scalar_fields`].
struct Writer<'a> {
    out: &'a mut [f32],
    at: usize,
}

impl<'a> Writer<'a> {
    fn new(out: &'a mut [f32]) -> Writer<'a> {
        Writer { out, at: 0 }
    }
    #[inline]
    fn push(&mut self, v: f32) {
        self.out[self.at] = v;
        self.at += 1;
    }
    #[inline]
    fn bit(&mut self, v: bool) {
        self.push(if v { 1.0 } else { 0.0 });
    }
    /// Leave `n` floats at their zeroed value — an unknown one-hot, encoded as all zeros.
    #[inline]
    fn skip(&mut self, n: usize) {
        self.at += n;
    }
    #[inline]
    fn one_hot(&mut self, index: usize, width: usize) {
        debug_assert!(index < width, "one-hot index {index} out of width {width}");
        self.out[self.at + index] = 1.0;
        self.at += width;
    }
    /// Per-rank counts, normalised by the copies of each rank the deck holds.
    fn counts(&mut self, counts: &[u8], width: usize, scale: f32) {
        for i in 0..width {
            self.push(counts[i] as f32 / scale);
        }
    }
    fn written(&self) -> usize {
        self.at
    }
}

/// Normalisers. Each is a config-derived quantity the feature can plausibly reach, so the
/// feature lands in roughly `[0, 1]` without clipping.
fn hand_scale(config: &GameConfig) -> f32 {
    (config.hand_size.max(1) * 2) as f32
}
fn pile_scale(config: &GameConfig) -> f32 {
    config.expected_pile_size().max(1) as f32
}
fn removed_scale(config: &GameConfig) -> f32 {
    let total = if config.variant.is_split() {
        config.removal_count * 2
    } else {
        config.removal_count
    };
    total.max(1) as f32
}

const fn phase_index(phase: Phase) -> usize {
    match phase {
        Phase::Main => 0,
        Phase::Foresight => 1,
        Phase::ResolveOrder => 2,
        Phase::QueenSource => 3,
        Phase::GiveBack => 4,
        Phase::SplitTarget => 5,
        Phase::Terminal => 6,
    }
}

// ===================================================================== action encoding ==

/// One block of the policy head.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ActionBlock {
    pub name: &'static str,
    /// First index of the block in the flat policy vector.
    pub offset: usize,
    pub len: usize,
}

/// The policy head's blocks, in order, with their offsets.
///
/// At the default configuration (`L = 3`, `S = 16`, `R = 13`):
///
/// | block | formula | size | engine `Action` |
/// |---|---|---:|---|
/// | `PLAY(rank, lane)` | `R·L` | 39 | [`Action::Play`] |
/// | `FLIP(lane, slot)` | `L·S` | 48 | [`Action::Flip`] |
/// | `ATTACK(lane, atk, tgt)` | `L·S·S` | 768 | [`Action::Attack`] |
/// | `PAIR(lane, a<b)` | `L·S(S−1)/2` | 360 | [`Action::DeclarePair`] |
/// | `PASS` | 1 | 1 | [`Action::Pass`] |
/// | `CHOOSE_SLOT(side, lane, slot)` | `2·L·S` | 96 | [`Action::Peek`] / [`Action::ResolveNext`] / [`Action::MoveHere`] / [`Action::SplitTarget`] |
/// | `CHOOSE_RANK(rank)` | `R` | 13 | [`Action::GiveBack`] |
/// | **total** | | **1325** | |
///
/// `CHOOSE_SLOT` is shared by four sub-decisions **because their phases are mutually
/// exclusive** — [`Phase::Foresight`], [`Phase::ResolveOrder`], [`Phase::QueenSource`] and
/// [`Phase::SplitTarget`] are never simultaneously active, so the legality mask
/// disambiguates and no two of them can collide on one logit. Sharing `FLIP` or `PAIR`
/// across same-rank cards would *not* have been safe, because those collide inside a single
/// phase.
pub fn action_blocks(config: &GameConfig) -> Vec<ActionBlock> {
    let l = config.lanes;
    let s = config.encoding_slots;
    let r = config.rank_count();
    let sizes = [
        ("PLAY", r * l),
        ("FLIP", l * s),
        ("ATTACK", l * s * s),
        ("PAIR", l * pairs_per_lane(s)),
        ("PASS", 1),
        ("CHOOSE_SLOT", 2 * l * s),
        ("CHOOSE_RANK", r),
    ];
    let mut offset = 0;
    sizes
        .into_iter()
        .map(|(name, len)| {
            let block = ActionBlock { name, offset, len };
            offset += len;
            block
        })
        .collect()
}

/// Unordered pairs of distinct slots: `S(S−1)/2`.
#[inline]
pub const fn pairs_per_lane(slots: usize) -> usize {
    slots * (slots - 1) / 2
}

/// Total policy-head width. 1325 at the default configuration.
pub fn action_dim(config: &GameConfig) -> usize {
    action_blocks(config).iter().map(|b| b.len).sum()
}

/// Offsets of the seven blocks, resolved once so encode and decode share them.
struct Offsets {
    lanes: usize,
    slots: usize,
    play: usize,
    flip: usize,
    attack: usize,
    pair: usize,
    pass: usize,
    choose_slot: usize,
    choose_rank: usize,
    total: usize,
}

impl Offsets {
    fn new(config: &GameConfig) -> Offsets {
        let b = action_blocks(config);
        Offsets {
            lanes: config.lanes,
            slots: config.encoding_slots,
            play: b[0].offset,
            flip: b[1].offset,
            attack: b[2].offset,
            pair: b[3].offset,
            pass: b[4].offset,
            choose_slot: b[5].offset,
            choose_rank: b[6].offset,
            total: b[6].offset + b[6].len,
        }
    }

    /// Index of the unordered pair `{a, b}` within one lane, for `a < b`.
    ///
    /// Row-major over the strict upper triangle: `{0,1}, {0,2}, …, {0,S−1}, {1,2}, …`.
    fn pair_index(&self, a: usize, b: usize) -> usize {
        debug_assert!(a < b && b < self.slots);
        // Slots skipped by the rows above `a`, plus the offset within row `a`.
        a * self.slots - a * (a + 1) / 2 + (b - a - 1)
    }

    fn unpair_index(&self, mut i: usize) -> (usize, usize) {
        let mut a = 0;
        loop {
            let row = self.slots - a - 1;
            if i < row {
                return (a, a + 1 + i);
            }
            i -= row;
            a += 1;
        }
    }

    fn choose_slot(&self, side: Side, lane: usize, slot: usize) -> usize {
        let s = match side {
            Side::Mine => 0,
            Side::Theirs => 1,
        };
        self.choose_slot + (s * self.lanes + lane) * self.slots + slot
    }
}

/// Bounds-check a lane/slot pair coming out of an [`Action`], with a message that names the
/// config key a training run would have to change.
fn checked(config: &GameConfig, lane: usize, slot: usize, what: &str) -> (usize, usize) {
    assert!(
        lane < config.lanes,
        "{what}: lane {lane} is outside the configured {} lanes",
        config.lanes
    );
    assert!(
        slot < config.encoding_slots,
        "{what}: slot {slot} is at or over the encoder's bound of {} — raise \
         `encoding_slots` in the config (see FINDINGS.md F2.7)",
        config.encoding_slots
    );
    (lane, slot)
}

/// The policy-head index for `action` in `state`.
///
/// `state` is a parameter rather than an oversight: [`Action::SplitTarget`] carries no lane,
/// because the lane comes from the attack already in flight. It is read out of the pending
/// decision here, so a split target encodes as `CHOOSE_SLOT(Theirs, pending_lane, slot)`
/// and cannot be confused with a peek at some other lane.
///
/// # Panics
///
/// If a lane or slot is outside the encoder's configured bounds, or if a `SplitTarget` is
/// encoded outside [`Phase::SplitTarget`] (which would mean the action did not come from
/// [`GameState::legal_actions`]).
pub fn encode_action(action: &Action, state: &GameState) -> usize {
    let config = &state.config;
    let o = Offsets::new(config);
    match *action {
        Action::Play { rank, lane } => {
            let lane = lane as usize;
            assert!(lane < o.lanes, "play: lane {lane} is out of range");
            o.play + rank.index() * o.lanes + lane
        }
        Action::Flip { lane, slot } => {
            let (lane, slot) = checked(config, lane as usize, slot as usize, "flip");
            o.flip + lane * o.slots + slot
        }
        Action::Attack {
            lane,
            attacker,
            target,
        } => {
            let (lane, attacker) = checked(config, lane as usize, attacker as usize, "attack");
            let (_, target) = checked(config, lane, target as usize, "attack target");
            o.attack + (lane * o.slots + attacker) * o.slots + target
        }
        Action::DeclarePair {
            lane,
            slot_a,
            slot_b,
        } => {
            // A pair is unordered (`game_rules.md` §5), so the index is canonicalised to
            // `a < b`. `legal.rs` already emits only that order — see
            // `phase3_legal_pairs_are_already_canonical` — so this never actually reorders
            // an action the engine produced; it is here for actions built by hand.
            let (lane, a) = checked(config, lane as usize, slot_a as usize, "pair");
            let (_, b) = checked(config, lane, slot_b as usize, "pair");
            assert_ne!(a, b, "a pair needs two distinct slots");
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            o.pair + lane * pairs_per_lane(o.slots) + o.pair_index(lo, hi)
        }
        Action::Pass => o.pass,
        Action::Peek { side, lane, slot } => {
            let (lane, slot) = checked(config, lane as usize, slot as usize, "peek");
            o.choose_slot(side, lane, slot)
        }
        // Both of these are always a card on the acting player's own side: a 5/King
        // resolution list is confined to one lane and one side (`legal.rs`
        // `resolution_still_valid`), and a Queen only pulls allied cards (§6).
        Action::ResolveNext { lane, slot } | Action::MoveHere { lane, slot } => {
            let (lane, slot) = checked(config, lane as usize, slot as usize, "choose slot");
            o.choose_slot(Side::Mine, lane, slot)
        }
        Action::GiveBack { rank } => o.choose_rank + rank.index(),
        Action::SplitTarget { slot } => {
            let lane = pending_split_lane(state)
                .expect("a split target only exists while a 10's twinstrike is pending");
            let (lane, slot) = checked(config, lane, slot as usize, "split target");
            o.choose_slot(Side::Theirs, lane, slot)
        }
    }
}

/// The action `index` names in `state`, or `None` if the index cannot be an action here.
///
/// The phase is what disambiguates the shared `CHOOSE_SLOT` block, so this is a partial
/// inverse of [`encode_action`] *at a position*, not a global one. A `None` result is not an
/// error — most of the 1325 indices are meaningless in any given position, which is what the
/// legality mask is for.
pub fn decode_action(index: usize, state: &GameState) -> Option<Action> {
    let config = &state.config;
    let o = Offsets::new(config);
    if index >= o.total {
        return None;
    }

    if index < o.flip {
        let i = index - o.play;
        return Some(Action::Play {
            rank: Rank::try_from_index(i / o.lanes)?,
            lane: (i % o.lanes) as u8,
        });
    }
    if index < o.attack {
        let i = index - o.flip;
        return Some(Action::Flip {
            lane: (i / o.slots) as u8,
            slot: (i % o.slots) as u8,
        });
    }
    if index < o.pair {
        let i = index - o.attack;
        let target = i % o.slots;
        let attacker = (i / o.slots) % o.slots;
        let lane = i / (o.slots * o.slots);
        return Some(Action::Attack {
            lane: lane as u8,
            attacker: attacker as u8,
            target: target as u8,
        });
    }
    if index < o.pass {
        let i = index - o.pair;
        let per_lane = pairs_per_lane(o.slots);
        let (a, b) = o.unpair_index(i % per_lane);
        return Some(Action::DeclarePair {
            lane: (i / per_lane) as u8,
            slot_a: a as u8,
            slot_b: b as u8,
        });
    }
    if index == o.pass {
        return Some(Action::Pass);
    }
    if index < o.choose_rank {
        let i = index - o.choose_slot;
        let slot = i % o.slots;
        let lane = (i / o.slots) % o.lanes;
        let side = if i / (o.slots * o.lanes) == 0 {
            Side::Mine
        } else {
            Side::Theirs
        };
        return match state.phase() {
            Phase::Foresight => Some(Action::Peek {
                side,
                lane: lane as u8,
                slot: slot as u8,
            }),
            // A resolution list and a Queen's source are both on the acting player's own
            // side, so `Theirs` names nothing.
            Phase::ResolveOrder if side == Side::Mine => Some(Action::ResolveNext {
                lane: lane as u8,
                slot: slot as u8,
            }),
            Phase::QueenSource if side == Side::Mine => Some(Action::MoveHere {
                lane: lane as u8,
                slot: slot as u8,
            }),
            // The lane is fixed by the attack in flight, so any other lane decodes to
            // nothing rather than to a split in the wrong lane.
            Phase::SplitTarget if side == Side::Theirs && pending_split_lane(state) == Some(lane) => {
                Some(Action::SplitTarget { slot: slot as u8 })
            }
            _ => None,
        };
    }
    let i = index - o.choose_rank;
    Some(Action::GiveBack {
        rank: Rank::try_from_index(i)?,
    })
}

/// The lane of the twinstrike currently waiting for its second target.
fn pending_split_lane(state: &GameState) -> Option<usize> {
    match state.pending.last() {
        Some(Pending::SplitTarget { lane, .. }) => Some(*lane as usize),
        _ => None,
    }
}

/// Write the legality mask for `state` into `out`, which must be [`action_dim`] long.
///
/// Built **from [`GameState::legal_actions`]** and nothing else. `CLAUDE.md`: the engine is
/// the sole authority on legality, and a second copy of the rules living in the mask is
/// exactly the bug that would be hardest to find — it would present as a policy that
/// occasionally proposes an illegal move, at which point the natural suspect is the network.
pub fn legal_mask(state: &GameState, out: &mut [bool]) {
    assert_eq!(
        out.len(),
        action_dim(&state.config),
        "mask buffer is {} entries, expected {}",
        out.len(),
        action_dim(&state.config)
    );
    out.fill(false);
    for action in state.legal_actions() {
        out[encode_action(&action, state)] = true;
    }
}

// ======================================================================= layout hashes ==

/// FNV-1a, 64-bit (Fowler–Noll–Vo, 1991). Chosen for the same reason [`crate::rng`] carries
/// its own generator: the value has to be stable forever, and the standard library's hasher
/// explicitly does not promise that across versions.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A stable, human-readable description of the observation layout.
///
/// Everything that could move a feature is in here: the board's shape, every per-slot
/// feature name in order, and every scalar field with its width. Two builds that agree on
/// this string encode the same function.
pub fn obs_layout_string(config: &GameConfig) -> String {
    let mut s = String::new();
    s.push_str("duel52.obs.v1\n");
    s.push_str(&format!(
        "board lanes={} sides=2 slots={} features={}\n",
        config.lanes,
        config.encoding_slots,
        slot_features(config)
    ));
    for name in SLOT_FEATURE_NAMES {
        s.push_str(&format!("slot {name}\n"));
    }
    s.push_str(&format!(
        "widths rank={} damage={} maxhp={} allowance={} actions={} phase={}\n",
        config.rank_count(),
        DAMAGE_BUCKETS,
        MAX_HP_BUCKETS,
        ALLOWANCE_BUCKETS,
        ACTIONS_BUCKETS,
        PHASE_COUNT
    ));
    for (name, width) in scalar_fields(config) {
        s.push_str(&format!("scalar {name} {width}\n"));
    }
    s.push_str(&format!("total {}\n", obs_dim(config)));
    s
}

/// A stable description of the policy-head layout: block names, offsets and widths.
pub fn action_layout_string(config: &GameConfig) -> String {
    let mut s = String::new();
    s.push_str("duel52.action.v1\n");
    for b in action_blocks(config) {
        s.push_str(&format!("block {} {} {}\n", b.name, b.offset, b.len));
    }
    s.push_str(&format!("total {}\n", action_dim(config)));
    s
}

/// Hash of [`obs_layout_string`]. Stamped into every checkpoint; Rust recomputes it from its
/// own constants at load and refuses a mismatch.
pub fn obs_layout_hash(config: &GameConfig) -> u64 {
    fnv1a64(obs_layout_string(config).as_bytes())
}

/// Hash of [`action_layout_string`].
pub fn action_layout_hash(config: &GameConfig) -> u64 {
    fnv1a64(action_layout_string(config).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_layout_matches_the_plan() {
        let cfg = GameConfig::default();
        assert_eq!(slot_features(&cfg), 33);
        assert_eq!(board_len(&cfg), 3 * 2 * 16 * 33);
        assert_eq!(action_dim(&cfg), 1325);
        assert_eq!(
            action_blocks(&cfg)
                .iter()
                .map(|b| b.len)
                .collect::<Vec<_>>(),
            vec![39, 48, 768, 360, 1, 96, 13]
        );
    }

    /// The slot-feature name list is what the layout hash commits to, so it has to stay in
    /// step with the width the encoder actually writes.
    #[test]
    fn slot_feature_names_cover_every_written_feature() {
        // 14 names, 33 floats: five of the names are one-hots wider than one float.
        assert_eq!(SLOT_FEATURE_NAMES.len(), 14);
        let cfg = GameConfig::default();
        let widths = 1
            + cfg.rank_count()
            + 1
            + 1
            + 1
            + 1
            + DAMAGE_BUCKETS
            + MAX_HP_BUCKETS
            + 1
            + ALLOWANCE_BUCKETS
            + 1
            + 1
            + 1
            + 1;
        assert_eq!(widths, slot_features(&cfg));
    }

    /// The pair index has to be a bijection onto `0..S(S−1)/2`, or two different pairs share
    /// a logit — the exact failure this encoding exists to avoid.
    #[test]
    fn pair_indices_are_a_bijection() {
        let cfg = GameConfig::default();
        let o = Offsets::new(&cfg);
        let mut seen = vec![false; pairs_per_lane(cfg.encoding_slots)];
        for a in 0..cfg.encoding_slots {
            for b in (a + 1)..cfg.encoding_slots {
                let i = o.pair_index(a, b);
                assert!(!seen[i], "pair ({a},{b}) collided at index {i}");
                seen[i] = true;
                assert_eq!(o.unpair_index(i), (a, b));
            }
        }
        assert!(seen.into_iter().all(|s| s));
    }

    /// The hash has to move when the layout does, or it is not protecting anything.
    #[test]
    fn layout_hashes_are_sensitive_to_the_shape() {
        let a = GameConfig::default();
        let mut b = a;
        b.encoding_slots = 12;
        assert_ne!(obs_layout_hash(&a), obs_layout_hash(&b));
        assert_ne!(action_layout_hash(&a), action_layout_hash(&b));
    }

    /// Every variant shares one layout, so one checkpoint plays all three.
    #[test]
    fn the_three_variants_share_a_layout() {
        let hashes: Vec<u64> = crate::config::Variant::ALL
            .into_iter()
            .map(|v| obs_layout_hash(&GameConfig::preset(v)))
            .collect();
        assert!(hashes.windows(2).all(|w| w[0] == w[1]));
    }
}
