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
//! which one to play — both arbitrary, both distorting the policy the insight phase reads. This
//! encoding is exact and slot-keyed instead. See [`action_blocks`] for the table.

use crate::action::{Action, Phase, Side};
use crate::card::Card;
use crate::config::GameConfig;
use crate::player::Player;
use crate::rank::Rank;
use crate::state::{GameState, Pending};

// ===================================================================== the board block ==

/// Per-slot features of the **base** layout, in encoding order.
///
/// Kept as names rather than as a bare count because [`obs_layout_hash`] hashes them: a
/// reordering that left the width unchanged would otherwise be invisible to the checkpoint
/// check, and it is exactly the kind of edit that silently breaks a trained network.
///
/// ⚠️ The reserve's status flags are **appended** to this list rather than inserted into it
/// (see [`RESERVE_SLOT_FEATURE_NAMES`]). That is what makes a base slot a prefix of an
/// extended one, which is in turn what makes [`reserve_embedding`] a monotonic map and the
/// checkpoint bridge obviously correct. Do not insert a feature in the middle.
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

/// The reserve's per-slot status flags, appended to [`SLOT_FEATURE_NAMES`] in the extended
/// layout. `MODULAR_RULES.md` §7, reserve item 3.
///
/// Unnamed on purpose. A flag's *meaning* belongs to the power that claims it
/// ([`crate::card::STATUS_SHIELDED`] is flag 0), and naming them here would make every new
/// status a change to the layout string — which is a hash move, which is the break the
/// reserve exists to avoid. So the encoder knows only that there are eight bits.
pub const RESERVE_SLOT_FEATURE_NAMES: &[&str] = &[
    "status_0", "status_1", "status_2", "status_3", "status_4", "status_5", "status_6",
    "status_7",
];

/// Per-slot feature names for this ruleset, base or extended.
pub fn slot_feature_names(config: &GameConfig) -> Vec<&'static str> {
    let mut names = SLOT_FEATURE_NAMES.to_vec();
    if config.extended_encoder() {
        names.extend_from_slice(RESERVE_SLOT_FEATURE_NAMES);
    }
    names
}

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

/// Phase one-hot width in the **base** layout — the seven phases the canonical ruleset can
/// reach, `Terminal` included.
pub const BASE_PHASE_COUNT: usize = 7;

/// Phase one-hot width in the **extended** layout: `MODULAR_RULES.md` §7's reserve item 1.
///
/// Twelve rather than nine. Two of the spare five are used — [`Phase::ChooseLane`] and
/// [`Phase::ChooseOption`] — and three are unclaimed, because §1c found that `PHASE_COUNT`
/// is the constraint that actually binds when a new *kind* of sub-decision is added, and
/// five floats of 4,290 is the cheapest structural option in the codebase. A phase that
/// nothing reaches is simply a one-hot position never written.
pub const EXTENDED_PHASE_COUNT: usize = 12;

/// Number of unnamed options the `CHOOSE_OPTION` policy block offers a modal power.
///
/// Four. The meaning of an option index is the power's business, not the encoder's — see
/// [`Action::ChooseOption`].
pub const OPTION_COUNT: usize = 4;

/// Phase one-hot width for this ruleset.
#[inline]
pub const fn phase_count(config: &GameConfig) -> usize {
    if config.extended_encoder() {
        EXTENDED_PHASE_COUNT
    } else {
        BASE_PHASE_COUNT
    }
}

/// Floats per slot in the **base** layout, for a given rank count. 33 at 13 ranks.
#[inline]
pub const fn base_slot_features(config: &GameConfig) -> usize {
    // occupied + rank one-hot + rank_unknown + face_up + is_base + entered_as_base
    1 + config.rank_count() + 1 + 1 + 1 + 1
        // damage + max HP + frozen + allowance + attacks_used_frac + can_attack_now
        + DAMAGE_BUCKETS + MAX_HP_BUCKETS + 1 + ALLOWANCE_BUCKETS + 1 + 1
        // paired + is_mine
        + 1 + 1
}

/// Floats per slot for this ruleset: 33 at 13 ranks, or 41 with the reserve's status flags.
#[inline]
pub const fn slot_features(config: &GameConfig) -> usize {
    if config.extended_encoder() {
        base_slot_features(config) + crate::card::STATUS_FLAG_COUNT
    } else {
        base_slot_features(config)
    }
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
        ("phase_onehot", phase_count(config)),
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
        ("lane_counts", config.lanes * LANE_COUNT_FEATURES),
    ]
}

/// Floats per lane in the `lane_counts` aggregate: `(mine total, mine face-up, theirs
/// total, theirs face-up)`, lane-major.
///
/// Named rather than written as a literal because [`lane_permutations`] has to know the
/// stride, and a permutation table that disagreed with the encoder about it would be
/// silent — see that function's warning.
pub const LANE_COUNT_FEATURES: usize = 4;

/// First float of the named scalar field, as an offset into the **scalar block**.
///
/// Derived from [`scalar_fields`] rather than written down a second time, so a field that
/// moves takes its offset with it.
fn scalar_offset(config: &GameConfig, name: &str) -> usize {
    let mut offset = 0;
    for (field, width) in scalar_fields(config) {
        if field == name {
            return offset;
        }
        offset += width;
    }
    panic!("the scalar block has no field named {name:?}");
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
            note_occupancy(side.len());
            for (slot, card) in side.iter().enumerate() {
                let base = ((lane * 2 + side_idx) * s + slot) * f;
                encode_slot(state, observer, card, owner == observer, &mut out[base..base + f]);
            }
        }
    }

    // ----------------------------------------------------------------- the scalars --
    let mut w = Writer::new(&mut out[board_len(config)..]);

    w.one_hot(phase_index(state.phase()), phase_count(config));
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

// ============================================================ the occupancy high-water ==

/// The widest lane side any call to [`encode_observation`] has seen in this process.
///
/// `FINDINGS.md` F2.7 asks for the encoding bound to be re-measured against the *trained*
/// agent — and for the **distribution**, not the maximum, because "a maximum is not a
/// statistic: it grows with the sample". This counter is the cheap half of that: a
/// high-water mark that says immediately whether a run is anywhere near the bound, without
/// having to instrument the caller.
///
/// It is process-wide and monotonic, so it says nothing about *which* run produced the
/// maximum. Call [`reset_observed_max_slots`] between runs if that matters.
static OBSERVED_MAX_SLOTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[inline]
fn note_occupancy(n: usize) {
    // Relaxed: this is instrumentation, and the only thing that reads it is a report at the
    // end of a run. Next to a five-million-parameter forward pass the cost is not
    // measurable, and an exact ordering would buy nothing.
    OBSERVED_MAX_SLOTS.fetch_max(n, std::sync::atomic::Ordering::Relaxed);
}

/// The widest lane side the encoder has seen since this process started, or since the last
/// [`reset_observed_max_slots`].
pub fn observed_max_slots() -> usize {
    OBSERVED_MAX_SLOTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Reset the high-water mark, so a measurement can be scoped to one run.
pub fn reset_observed_max_slots() {
    OBSERVED_MAX_SLOTS.store(0, std::sync::atomic::Ordering::Relaxed);
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
    w.one_hot(
        if card.max_hp(&state.config) >= 3 { 1 } else { 0 },
        MAX_HP_BUCKETS,
    );
    w.bit(card.is_frozen(state.ply));
    w.one_hot(
        (card.attack_allowance as usize).min(ALLOWANCE_BUCKETS - 1),
        ALLOWANCE_BUCKETS,
    );
    w.push(card.attacks_used as f32 / card.attack_allowance.max(1) as f32);
    w.bit(card.can_attack(state.ply));
    w.bit(card.pair_id.is_some());
    w.bit(mine);

    // The reserve's status flags (`MODULAR_RULES.md` §7). Appended after every base feature,
    // so a base slot is a prefix of an extended one — see [`reserve_embedding`].
    //
    // A status is **public**, like damage: both players see the token on the card. That is
    // what lets it be written here rather than per-observer, and it is asserted by
    // `phase3_observation_is_a_function_of_the_information_set` like everything else.
    if state.config.extended_encoder() {
        for flag in 0..crate::card::STATUS_FLAG_COUNT {
            w.bit(card.has_status(flag as u8));
        }
    }

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
        for &n in &counts[..width] {
            self.push(n as f32 / scale);
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
        // The reserve's phases (`MODULAR_RULES.md` §7). These indices exist only in the
        // extended layout — `phase_count` is 7 under canonical rules, and no canonical
        // ruleset can reach either phase, which `reserve_phases_need_the_extended_encoder`
        // asserts by walking every power.
        Phase::ChooseLane => 7,
        Phase::ChooseOption => 8,
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
/// | `CHOOSE_SLOT(side, lane, slot)` | `2·L·S` | 96 | [`Action::Peek`] / [`Action::ResolveNext`] / [`Action::MoveHere`] / [`Action::SplitTarget`] |
/// | `CHOOSE_RANK(rank)` | `R` | 13 | [`Action::GiveBack`] |
/// | **total** | | **1324** | |
///
/// There is no `PASS` block. §4 makes actions mandatory, so a turn with nothing legal in it
/// is ended by the engine rather than chosen — every logit here is a decision somebody
/// actually makes.
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
    let mut sizes = vec![
        ("PLAY", r * l),
        ("FLIP", l * s),
        ("ATTACK", l * s * s),
        ("PAIR", l * pairs_per_lane(s)),
        ("CHOOSE_SLOT", 2 * l * s),
        ("CHOOSE_RANK", r),
    ];
    // `MODULAR_RULES.md` §7's reserve, item 2. **Appended**, not inserted: that keeps the
    // base policy head an exact prefix of the extended one, so [`reserve_embedding`]'s
    // action map is the identity and a widened checkpoint's existing logits do not move.
    // The price is that `global_action` in [`lane_structure`] stops being one contiguous
    // range — `CHOOSE_LANE` is lane-owned and sits past `CHOOSE_RANK`, which is global — and
    // that is paid there, once, behind `assert_partitions`.
    if config.extended_encoder() {
        sizes.push(("CHOOSE_LANE", 2 * l));
        sizes.push(("CHOOSE_OPTION", OPTION_COUNT));
    }
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

/// Total policy-head width. 1324 at the default configuration.
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
    choose_slot: usize,
    choose_rank: usize,
    /// First index of `CHOOSE_LANE`, or `None` in the base layout. `MODULAR_RULES.md` §7.
    choose_lane: Option<usize>,
    /// First index of `CHOOSE_OPTION`, or `None` in the base layout.
    choose_option: Option<usize>,
    total: usize,
}

impl Offsets {
    fn new(config: &GameConfig) -> Offsets {
        let b = action_blocks(config);
        let last = b.last().expect("the policy head always has blocks");
        Offsets {
            lanes: config.lanes,
            slots: config.encoding_slots,
            play: b[0].offset,
            flip: b[1].offset,
            attack: b[2].offset,
            pair: b[3].offset,
            choose_slot: b[4].offset,
            choose_rank: b[5].offset,
            choose_lane: b.get(6).map(|x| x.offset),
            choose_option: b.get(7).map(|x| x.offset),
            total: last.offset + last.len,
        }
    }

    /// Index of `CHOOSE_LANE(side, lane)`. Lane-major within each side, so a lane
    /// relabelling is [`relabel_lane_major`] with a stride of one.
    fn choose_lane(&self, side: Side, lane: usize) -> usize {
        let base = self.choose_lane.expect(
            "a ChooseLane action needs the extended encoder — the ruleset installed a power \
             that opens Phase::ChooseLane but PowerId::needs_extended_encoder said false",
        );
        let s = match side {
            Side::Mine => 0,
            Side::Theirs => 1,
        };
        base + s * self.lanes + lane
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

/// Bounds-check a rank against the configured deck.
///
/// `Rank` is always `0..13`, but a block is only `config.rank_count()` wide — so in a
/// reduced-deck configuration (Duel52-mini, `DESIGN.md` §7) an out-of-range rank would not
/// be caught by `Rank`'s own invariant and would index into the *next* block instead. A real
/// deal cannot produce one; `testkit` can.
fn checked_rank(config: &GameConfig, rank: Rank, what: &str) -> usize {
    assert!(
        rank.index() < config.rank_count(),
        "{what}: rank {rank} is outside the configured deck of {} ranks",
        config.rank_count()
    );
    rank.index()
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
            o.play + checked_rank(config, rank, "play") * o.lanes + lane
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
        Action::GiveBack { rank } => o.choose_rank + checked_rank(config, rank, "give back"),
        Action::SplitTarget { slot } => {
            let lane = pending_split_lane(state)
                .expect("a split target only exists while a 10's twinstrike is pending");
            let (lane, slot) = checked(config, lane, slot as usize, "split target");
            o.choose_slot(Side::Theirs, lane, slot)
        }
        Action::ChooseLane { side, lane } => {
            let lane = lane as usize;
            assert!(lane < o.lanes, "choose lane: lane {lane} is out of range");
            o.choose_lane(side, lane)
        }
        Action::ChooseOption { option } => {
            let base = o.choose_option.expect(
                "a ChooseOption action needs the extended encoder — the ruleset installed a \
                 power that opens Phase::ChooseOption but PowerId::needs_extended_encoder \
                 said false",
            );
            assert!(
                (option as usize) < OPTION_COUNT,
                "choose option: option {option} is at or over the reserve's {OPTION_COUNT}"
            );
            base + option as usize
        }
    }
}

/// The action `index` names in `state`, or `None` if the index cannot be an action here.
///
/// The phase is what disambiguates the shared `CHOOSE_SLOT` block, so this is a partial
/// inverse of [`encode_action`] *at a position*, not a global one. A `None` result is not an
/// error — most of the 1324 indices are meaningless in any given position, which is what the
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
    if index < o.choose_slot {
        let i = index - o.pair;
        let per_lane = pairs_per_lane(o.slots);
        let (a, b) = o.unpair_index(i % per_lane);
        return Some(Action::DeclarePair {
            lane: (i / per_lane) as u8,
            slot_a: a as u8,
            slot_b: b as u8,
        });
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
    // `CHOOSE_RANK` is the last base block, so in the base layout everything from here on is
    // a give-back. In the extended layout the two reserve blocks follow it.
    let choose_lane = o.choose_lane.unwrap_or(o.total);
    if index < choose_lane {
        let i = index - o.choose_rank;
        return Some(Action::GiveBack {
            rank: Rank::try_from_index(i)?,
        });
    }

    // Like `CHOOSE_SLOT`, the reserve blocks are disambiguated by the phase, and decode to
    // nothing outside it rather than to an action nobody can take.
    let choose_option = o.choose_option.unwrap_or(o.total);
    if index < choose_option {
        if state.phase() != Phase::ChooseLane {
            return None;
        }
        let i = index - choose_lane;
        return Some(Action::ChooseLane {
            side: if i / o.lanes == 0 { Side::Mine } else { Side::Theirs },
            lane: (i % o.lanes) as u8,
        });
    }
    if state.phase() != Phase::ChooseOption {
        return None;
    }
    Some(Action::ChooseOption {
        option: (index - choose_option) as u8,
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

// =================================================================== lane permutations ==

/// One element of S(`lanes`), as index maps on the observation and on the policy head.
///
/// `PLAN.md` §4.2a: Duel 52 is invariant under any permutation of its three lanes. No rule
/// in `game_rules.md` names a lane, orders them or tells one from another, so for every
/// position `s` and every `σ` there is a position `σ(s)` with the same value, whose legal
/// actions are `σ` applied to `s`'s and whose optimal policy is `σ` applied to `s`'s. That
/// makes these tables an **exact** relabelling rather than an approximation, and it holds at
/// every position in the game rather than only on the symmetric opening.
///
/// Both maps run **old index → new index**, which is the direction a sparse sample is
/// augmented in: a training row holds indices into `s`'s tensors and wants indices into
/// `σ(s)`'s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LanePermutation {
    /// `lanes[l]` is the lane that lane `l` becomes. The identity is first in
    /// [`lane_permutations`]'s list.
    pub lanes: Vec<usize>,
    /// `encode_observation(σ(s))[obs[i]] == encode_observation(s)[i]`, for every `i`.
    /// [`obs_dim`] long.
    pub obs: Vec<u32>,
    /// `encode_action(σ(a), σ(s)) == action[encode_action(a, s)]`. [`action_dim`] long.
    pub action: Vec<u32>,
}

/// Every permutation of the lanes, with its observation and policy index maps. Six of them
/// at `lanes = 3`, the identity first.
///
/// # Why this is in Rust
///
/// `CLAUDE.md`: there is exactly one encoder and it owns the feature layout. A permutation
/// table is a *reading* of that layout, so a copy of it derived in Python would be a second
/// encoder in the only sense that matters — it would drift silently. ⚠️ **A wrong table
/// does not crash anything.** The network trains on observations paired with somebody
/// else's targets and comes out merely bad, with the training run as the natural suspect.
/// `engine/tests/encoding.rs::phase4_lane_permutation_commutes_with_the_encoder` is the
/// guard: it checks these tables against [`encode_observation`] and [`encode_action`]
/// themselves, not against a second reading of the layout.
///
/// # What moves
///
/// Everything lane-indexed, and nothing else:
///
/// - **The board**, `((lane * 2 + side) * slots + slot) * features`, is lane-outermost, so
///   it is three contiguous `2 × slots × features` chunks changing places.
/// - **`lane_counts`**, the only lane-indexed scalar, [`LANE_COUNT_FEATURES`] per lane,
///   lane-major.
/// - **`FLIP`, `ATTACK`, `PAIR` and `CHOOSE_SLOT`** are lane-major blocks (`CHOOSE_SLOT`
///   twice over, once per side); the index *within* a lane is built from slots, which a
///   relabelling does not touch.
/// - **`PLAY`** is `rank * lanes + lane` and so strided rather than blocked.
/// - **`CHOOSE_RANK`** and every other scalar are fixed points.
pub fn lane_permutations(config: &GameConfig) -> Vec<LanePermutation> {
    permutations(config.lanes)
        .into_iter()
        .map(|sigma| LanePermutation {
            obs: obs_permutation(config, &sigma),
            action: action_permutation(config, &sigma),
            lanes: sigma,
        })
        .collect()
}

/// Every permutation of `0..n`, in lexicographic order — so the identity is first.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current: Vec<usize> = (0..n).collect();
    loop {
        out.push(current.clone());
        // Narayana's next-permutation: the standard in-place lexicographic successor.
        let Some(i) = (0..current.len().saturating_sub(1)).rev().find(|&i| current[i] < current[i + 1])
        else {
            return out;
        };
        let j = (i + 1..current.len()).rev().find(|&j| current[j] > current[i]).expect("i qualifies");
        current.swap(i, j);
        current[i + 1..].reverse();
    }
}

/// Relabel a run of `lanes` contiguous, equal-sized, lane-major blocks.
fn relabel_lane_major(map: &mut [u32], base: usize, stride: usize, lanes: usize, sigma: &[usize]) {
    for lane in 0..lanes {
        for k in 0..stride {
            map[base + lane * stride + k] = (base + sigma[lane] * stride + k) as u32;
        }
    }
}

fn obs_permutation(config: &GameConfig, sigma: &[usize]) -> Vec<u32> {
    // Identity everywhere, then overwrite the lane-indexed parts: a feature nobody names
    // below is a feature a relabelling cannot move.
    let mut map: Vec<u32> = (0..obs_dim(config) as u32).collect();
    let per_lane = 2 * config.encoding_slots * slot_features(config);
    relabel_lane_major(&mut map, 0, per_lane, config.lanes, sigma);
    let lane_counts = board_len(config) + scalar_offset(config, "lane_counts");
    relabel_lane_major(&mut map, lane_counts, LANE_COUNT_FEATURES, config.lanes, sigma);
    map
}

fn action_permutation(config: &GameConfig, sigma: &[usize]) -> Vec<u32> {
    let mut map: Vec<u32> = (0..action_dim(config) as u32).collect();
    let o = Offsets::new(config);
    let (l, s) = (o.lanes, o.slots);

    // PLAY is the one strided block: `rank * lanes + lane`.
    for rank in 0..config.rank_count() {
        for lane in 0..l {
            map[o.play + rank * l + lane] = (o.play + rank * l + sigma[lane]) as u32;
        }
    }
    relabel_lane_major(&mut map, o.flip, s, l, sigma);
    relabel_lane_major(&mut map, o.attack, s * s, l, sigma);
    relabel_lane_major(&mut map, o.pair, pairs_per_lane(s), l, sigma);
    // `CHOOSE_SLOT` is `(side * lanes + lane) * slots + slot`: lane-major within each side.
    for side in 0..2 {
        relabel_lane_major(&mut map, o.choose_slot + side * l * s, s, l, sigma);
    }
    // `CHOOSE_LANE` is `side * lanes + lane` — lane-major per side with a stride of one.
    //
    // ⚠️ Omitting this block would be **silent**. A permutation that left it fixed is still a
    // bijection, so `lane_permutations_are_bijections_with_the_identity_first` would pass and
    // the tables would still compose as S₃; the only thing that would break is the meaning,
    // and the symptom would be an agent that is merely bad. The guard is
    // `phase4_lane_permutation_commutes_with_the_encoder`, which checks against
    // `encode_action` itself rather than against a second reading of the layout.
    if let Some(base) = o.choose_lane {
        for side in 0..2 {
            relabel_lane_major(&mut map, base + side * l, 1, l, sigma);
        }
    }
    // `CHOOSE_RANK` and `CHOOSE_OPTION` name no lane.
    map
}

// ===================================================================== lane structure ==

/// Floats of the observation that belong to one lane: its board chunk plus its slice of
/// `lane_counts`.
pub fn lane_obs_len(config: &GameConfig) -> usize {
    2 * config.encoding_slots * slot_features(config) + LANE_COUNT_FEATURES
}

/// Floats of the observation that belong to no lane: the scalar block minus `lane_counts`.
pub fn global_obs_len(config: &GameConfig) -> usize {
    obs_dim(config) - config.lanes * lane_obs_len(config)
}

/// Logits of the policy head that belong to one lane.
pub fn lane_action_len(config: &GameConfig) -> usize {
    let s = config.encoding_slots;
    // PLAY(·, lane) + FLIP + ATTACK + PAIR + CHOOSE_SLOT on both sides.
    let base = config.rank_count() + s + s * s + pairs_per_lane(s) + 2 * s;
    // …and `CHOOSE_LANE(side, lane)`, two per lane, when the reserve is on.
    if config.extended_encoder() {
        base + 2
    } else {
        base
    }
}

/// Logits of the policy head that name no lane: `CHOOSE_RANK`, and `CHOOSE_OPTION` when the
/// reserve is on.
pub fn global_action_len(config: &GameConfig) -> usize {
    if config.extended_encoder() {
        config.rank_count() + OPTION_COUNT
    } else {
        config.rank_count()
    }
}

/// The lane-structured partition of the observation and policy vectors.
///
/// `PLAN.md` §4.2b. [`lane_permutations`] says *how* a relabelling moves an index; this says
/// **which lane owns it**, which is what an architecture needs in order to share one weight
/// matrix across the three lanes instead of learning the symmetry from data. The two are the
/// same reading of the same layout, and `phase4_lane_structure_agrees_with_the_permutations`
/// checks them against each other rather than against a second transcription.
///
/// # The order within a lane is the contract
///
/// `lane_obs[l]` and `lane_action[l]` list lane `l`'s indices in an order that is *identical*
/// for every `l` — position `k` of every list is the same feature of a different lane. That
/// is what makes one shared matrix meaningful, and it is why these are index lists rather
/// than ranges: the board is contiguous per lane but `PLAY` is strided (`rank * lanes + lane`)
/// and `CHOOSE_SLOT` is two chunks, so no single range covers a lane.
///
/// # Why it is in Rust
///
/// `CLAUDE.md`: there is exactly one encoder and it owns the feature layout. This is a
/// reading of that layout, so a copy derived in Python would be a second encoder in the only
/// sense that matters. ⚠️ A wrong table does not crash: the network reads lane 2's board
/// through lane 1's weights and comes out merely bad.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneStructure {
    /// `lane_obs[l]` — observation indices owned by lane `l`, [`lane_obs_len`] of them.
    pub lane_obs: Vec<Vec<u32>>,
    /// Observation indices owned by no lane, [`global_obs_len`] of them.
    pub global_obs: Vec<u32>,
    /// `lane_action[l]` — policy indices owned by lane `l`, [`lane_action_len`] of them.
    pub lane_action: Vec<Vec<u32>>,
    /// Policy indices owned by no lane, [`global_action_len`] of them.
    pub global_action: Vec<u32>,
}

/// Which lane owns each observation float and each policy logit.
///
/// Panics if the four lists do not partition `0..obs_dim` and `0..action_dim` exactly. That
/// is a build-time property of the layout rather than a runtime condition, and the failure it
/// prevents — a feature reaching no weight, or a logit written twice — is silent.
pub fn lane_structure(config: &GameConfig) -> LaneStructure {
    let l = config.lanes;
    let s = config.encoding_slots;
    let r = config.rank_count();
    let board_per_lane = 2 * s * slot_features(config);
    let lane_counts = board_len(config) + scalar_offset(config, "lane_counts");
    let o = Offsets::new(config);

    let lane_obs: Vec<Vec<u32>> = (0..l)
        .map(|lane| {
            let mut idx = Vec::with_capacity(lane_obs_len(config));
            let base = lane * board_per_lane;
            idx.extend((0..board_per_lane).map(|k| (base + k) as u32));
            let counts = lane_counts + lane * LANE_COUNT_FEATURES;
            idx.extend((0..LANE_COUNT_FEATURES).map(|k| (counts + k) as u32));
            idx
        })
        .collect();

    // The scalar block, minus the `lane_counts` window the lists above claimed.
    let scalars = board_len(config)..obs_dim(config);
    let claimed = lane_counts..lane_counts + l * LANE_COUNT_FEATURES;
    let global_obs: Vec<u32> = scalars.filter(|i| !claimed.contains(i)).map(|i| i as u32).collect();

    let lane_action: Vec<Vec<u32>> = (0..l)
        .map(|lane| {
            let mut idx = Vec::with_capacity(lane_action_len(config));
            // PLAY is strided: `rank * lanes + lane`.
            idx.extend((0..r).map(|rank| (o.play + rank * l + lane) as u32));
            idx.extend((0..s).map(|slot| (o.flip + lane * s + slot) as u32));
            idx.extend((0..s * s).map(|k| (o.attack + lane * s * s + k) as u32));
            let p = pairs_per_lane(s);
            idx.extend((0..p).map(|k| (o.pair + lane * p + k) as u32));
            // CHOOSE_SLOT is `(side * lanes + lane) * slots + slot` — lane-major per side.
            for side in 0..2 {
                let base = o.choose_slot + (side * l + lane) * s;
                idx.extend((0..s).map(|slot| (base + slot) as u32));
            }
            // CHOOSE_LANE is `side * lanes + lane`: one logit per side per lane, and the
            // logit that *names* lane `l` is owned by lane `l`.
            if let Some(base) = o.choose_lane {
                idx.extend((0..2).map(|side| (base + side * l + lane) as u32));
            }
            idx
        })
        .collect();

    // `CHOOSE_RANK` plus, in the extended layout, `CHOOSE_OPTION`. Not one range: the
    // reserve appends `CHOOSE_LANE` between them and that block is lane-owned, so this is a
    // filter over the tail rather than the `o.choose_rank..o.total` it is in the base
    // layout. `assert_partitions` below is what makes getting this wrong a build failure.
    let claimed_lanes: Vec<usize> = match o.choose_lane {
        Some(base) => (base..base + 2 * l).collect(),
        None => Vec::new(),
    };
    let global_action: Vec<u32> = (o.choose_rank..o.total)
        .filter(|i| !claimed_lanes.contains(i))
        .map(|i| i as u32)
        .collect();

    let structure = LaneStructure { lane_obs, global_obs, lane_action, global_action };
    structure.assert_partitions(config);
    structure
}

impl LaneStructure {
    /// Every index of both vectors is claimed exactly once.
    fn assert_partitions(&self, config: &GameConfig) {
        let check = |lists: &[&[u32]], total: usize, what: &str| {
            let mut seen = vec![false; total];
            for list in lists {
                for &i in *list {
                    let i = i as usize;
                    assert!(i < total, "{what} index {i} is outside 0..{total}");
                    assert!(!seen[i], "{what} index {i} is claimed twice");
                    seen[i] = true;
                }
            }
            if let Some(missing) = seen.iter().position(|&s| !s) {
                panic!("{what} index {missing} is claimed by no lane and by no global list");
            }
        };
        let mut obs: Vec<&[u32]> = self.lane_obs.iter().map(|v| v.as_slice()).collect();
        obs.push(&self.global_obs);
        check(&obs, obs_dim(config), "observation");
        let mut action: Vec<&[u32]> = self.lane_action.iter().map(|v| v.as_slice()).collect();
        action.push(&self.global_action);
        check(&action, action_dim(config), "action");
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
    for name in slot_feature_names(config) {
        s.push_str(&format!("slot {name}\n"));
    }
    s.push_str(&format!(
        "widths rank={} damage={} maxhp={} allowance={} actions={} phase={}\n",
        config.rank_count(),
        DAMAGE_BUCKETS,
        MAX_HP_BUCKETS,
        ALLOWANCE_BUCKETS,
        ACTIONS_BUCKETS,
        phase_count(config)
    ));
    // Written only in the extended layout, so the base string stays **byte-identical** to
    // the pre-reserve build and every checkpoint in `models/` still loads. It is here so
    // that `duel52 config` on a reserve ruleset says so in words rather than leaving a
    // reader to infer it from the widths.
    if config.extended_encoder() {
        s.push_str(&format!(
            "reserve status_flags={} phases={}\n",
            crate::card::STATUS_FLAG_COUNT,
            EXTENDED_PHASE_COUNT - BASE_PHASE_COUNT
        ));
    }
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

// ==================================================================== the reserve bridge ==

/// Where each **base**-layout index lands in the **extended** layout.
///
/// `MODULAR_RULES.md` §7. Both maps run base index → extended index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReserveEmbedding {
    /// `obs[i]` is the extended-layout index of base observation float `i`. [`obs_dim`] of
    /// the *base* layout long.
    pub obs: Vec<u32>,
    /// `action[i]` is the extended-layout index of base policy logit `i`.
    pub action: Vec<u32>,
    /// Width of the base layout these maps come from.
    pub base_obs_dim: usize,
    pub base_action_dim: usize,
    /// Width of the extended layout they map into.
    pub extended_obs_dim: usize,
    pub extended_action_dim: usize,

    // ---- the same embedding, expressed per lane-partition ----
    //
    // The lane-equivariant network (`PLAN.md` §4.2b) does not hold a `[width, obs_dim]`
    // matrix: it holds `[width, lane_obs_len]` and `[width, global_obs_len]`, and gathers
    // with [`lane_structure`]. So widening it needs the embedding stated in *those*
    // coordinates — base position within a lane → extended position within a lane.
    //
    // Derived from `lane_structure` on both layouts rather than recomputed, so there is no
    // second reading of the layout to drift.
    /// Base position within a lane → extended position within a lane, for the observation.
    /// Identical for every lane, which [`reserve_embedding`] asserts.
    pub lane_obs: Vec<u32>,
    /// Base position → extended position among the observation floats no lane owns.
    pub global_obs: Vec<u32>,
    /// Base position within a lane → extended position, for the policy head.
    pub lane_action: Vec<u32>,
    /// Base position → extended position among the logits no lane owns.
    pub global_action: Vec<u32>,
}

/// Re-express a global-index embedding in the coordinates of one lane-partition list.
///
/// `base_list` and `ext_list` are the *same* partition of the two layouts (lane 0's
/// observation floats, say). For each base position, follow the global embedding and find
/// where that extended index sits in the extended list.
fn positional_map(base_list: &[u32], ext_list: &[u32], global: &[u32], width: usize) -> Vec<u32> {
    let mut position = vec![u32::MAX; width];
    for (k, &i) in ext_list.iter().enumerate() {
        position[i as usize] = k as u32;
    }
    base_list
        .iter()
        .map(|&b| {
            let target = global[b as usize] as usize;
            let at = position[target];
            assert_ne!(
                at,
                u32::MAX,
                "base index {b} embeds to extended index {target}, which the extended \
                 layout assigns to a different partition — the reserve moved a feature \
                 between a lane and the global set"
            );
            at
        })
        .collect()
}

/// The same board shape as `config`, with the canonical powers installed — i.e. `config`'s
/// **base** layout.
///
/// Only the shape fields matter here (`lanes`, `encoding_slots`, `rank_count`), and swapping
/// the powers is what turns [`GameConfig::extended_encoder`] off. Returning a real config
/// rather than threading an `extended: bool` through every width function means the base
/// layout is computed by exactly the same code that computes it for a canonical ruleset,
/// instead of by a parallel path that could drift.
fn with_base_layout(config: &GameConfig) -> GameConfig {
    let mut base = *config;
    for rank in Rank::ALL {
        base.powers[rank.index()] = crate::powers::PowerId::canonical_for(rank);
    }
    debug_assert!(!base.extended_encoder());
    base
}

/// How a base-layout checkpoint's weights move into the extended layout.
///
/// `MODULAR_RULES.md` §7. This is what makes the reserve affordable: without it the first
/// ruleset to claim a status flag pays a 24-hour from-scratch run, and with it that run is a
/// 3-hour warm start from the current champion.
///
/// # Why this is exact and not an approximation
///
/// Every reserve feature is **appended** — status flags after the base slot features, spare
/// phase positions after the seven real ones, `CHOOSE_LANE` and `CHOOSE_OPTION` after
/// `CHOOSE_RANK`. So the base layout embeds in the extended one monotonically, every base
/// feature keeps its meaning, and the features that have no preimage are exactly the reserve
/// ones, which are **zero** in any position a base ruleset could produce.
///
/// A checkpoint widened with this map therefore computes a *bit-identical* forward pass on
/// any position both layouts can express: the input layer walks non-zeros
/// ([`crate::nn`], `FINDINGS.md` F3.3), the same non-zeros reach the same weight rows in the
/// same order, and the new rows are never visited. `reserve_embedding_preserves_the_encoding`
/// asserts the tensor half of that directly against [`encode_observation`].
///
/// # Panics
///
/// If `config` is not an extended ruleset — there is nothing to embed into.
pub fn reserve_embedding(config: &GameConfig) -> ReserveEmbedding {
    assert!(
        config.extended_encoder(),
        "reserve_embedding needs an extended ruleset; `{}` uses the base layout, which is \
         already what a shipped checkpoint is written against",
        config.rules_label()
    );
    let base = with_base_layout(config);
    let (s, l) = (config.encoding_slots, config.lanes);

    // ---- the board: one slot at a time, base features first in both layouts ----
    let (bf, ef) = (slot_features(&base), slot_features(config));
    let mut obs = vec![0u32; obs_dim(&base)];
    for chunk in 0..(l * 2 * s) {
        for k in 0..bf {
            obs[chunk * bf + k] = (chunk * ef + k) as u32;
        }
    }

    // ---- the scalars: identical fields in identical order, `phase_onehot` wider ----
    let (mut b_at, mut e_at) = (board_len(&base), board_len(config));
    for ((b_name, b_w), (e_name, e_w)) in scalar_fields(&base).into_iter().zip(scalar_fields(config))
    {
        assert_eq!(
            b_name, e_name,
            "the reserve reordered the scalar block; the embedding assumes it only widens \
             fields in place"
        );
        assert!(b_w <= e_w, "scalar field {b_name} shrank in the extended layout");
        // A widened one-hot keeps its low positions, because `phase_index` appends.
        for k in 0..b_w {
            obs[b_at + k] = (e_at + k) as u32;
        }
        b_at += b_w;
        e_at += e_w;
    }
    debug_assert_eq!(b_at, obs_dim(&base));
    debug_assert_eq!(e_at, obs_dim(config));

    // ---- the policy head: match blocks by name ----
    let extended_blocks = action_blocks(config);
    let mut action = vec![0u32; action_dim(&base)];
    for b in action_blocks(&base) {
        let e = extended_blocks
            .iter()
            .find(|e| e.name == b.name)
            .unwrap_or_else(|| panic!("the extended layout dropped the `{}` block", b.name));
        assert_eq!(e.len, b.len, "the `{}` block changed width", b.name);
        for k in 0..b.len {
            action[b.offset + k] = (e.offset + k) as u32;
        }
    }

    // ---- the same maps, in lane-partition coordinates ----
    let (bs, es) = (lane_structure(&base), lane_structure(config));
    let (ow, aw) = (obs_dim(config), action_dim(config));
    let lane_obs = positional_map(&bs.lane_obs[0], &es.lane_obs[0], &obs, ow);
    let lane_action = positional_map(&bs.lane_action[0], &es.lane_action[0], &action, aw);
    // The contract of `lane_structure` is that position `k` of every lane's list is the same
    // feature of a different lane. If that holds, the per-lane embedding cannot depend on the
    // lane — and if it does not hold, one shared weight matrix was never meaningful. Checked
    // rather than assumed, because this is the table a widened checkpoint is gathered with.
    for lane in 1..l {
        assert_eq!(
            positional_map(&bs.lane_obs[lane], &es.lane_obs[lane], &obs, ow),
            lane_obs,
            "the observation embedding differs between lane 0 and lane {lane}"
        );
        assert_eq!(
            positional_map(&bs.lane_action[lane], &es.lane_action[lane], &action, aw),
            lane_action,
            "the policy embedding differs between lane 0 and lane {lane}"
        );
    }

    ReserveEmbedding {
        lane_obs,
        global_obs: positional_map(&bs.global_obs, &es.global_obs, &obs, ow),
        lane_action,
        global_action: positional_map(&bs.global_action, &es.global_action, &action, aw),
        obs,
        action,
        base_obs_dim: obs_dim(&base),
        base_action_dim: action_dim(&base),
        extended_obs_dim: obs_dim(config),
        extended_action_dim: action_dim(config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_layout_matches_the_plan() {
        let cfg = GameConfig::default();
        assert_eq!(slot_features(&cfg), 33);
        assert_eq!(board_len(&cfg), 3 * 2 * 16 * 33);
        assert_eq!(action_dim(&cfg), 1324);
        assert_eq!(
            action_blocks(&cfg)
                .iter()
                .map(|b| b.len)
                .collect::<Vec<_>>(),
            vec![39, 48, 768, 360, 96, 13],
            "six blocks: PLAY, FLIP, ATTACK, PAIR, CHOOSE_SLOT, CHOOSE_RANK — and no PASS"
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

    /// Six permutations, the identity first, and each map a bijection of its whole block.
    ///
    /// A table that dropped an index — or wrote two features onto one — would train the
    /// network on a mangled observation with nothing to say so.
    #[test]
    fn lane_permutations_are_bijections_with_the_identity_first() {
        let cfg = GameConfig::default();
        let perms = lane_permutations(&cfg);
        assert_eq!(perms.len(), 6, "|S₃| = 6");
        assert_eq!(perms[0].lanes, vec![0, 1, 2], "the identity comes first");
        assert!(perms[0].obs.iter().copied().eq(0..obs_dim(&cfg) as u32));
        assert!(perms[0].action.iter().copied().eq(0..action_dim(&cfg) as u32));
        for p in &perms {
            for (map, len) in [(&p.obs, obs_dim(&cfg)), (&p.action, action_dim(&cfg))] {
                assert_eq!(map.len(), len);
                let mut hit = vec![false; len];
                for &to in map.iter() {
                    assert!(!hit[to as usize], "{:?} maps two indices onto {to}", p.lanes);
                    hit[to as usize] = true;
                }
            }
        }
    }

    /// The tables compose the way the permutations do: applying `σ` then `τ` has to be the
    /// table of `τ∘σ`. This is what says the six are one group rather than six unrelated
    /// relabellings that happen to be bijections.
    #[test]
    fn lane_permutation_tables_compose_as_s3() {
        let cfg = GameConfig::default();
        let perms = lane_permutations(&cfg);
        for sigma in &perms {
            for tau in &perms {
                let lanes: Vec<usize> = (0..cfg.lanes).map(|l| tau.lanes[sigma.lanes[l]]).collect();
                let rho = perms
                    .iter()
                    .find(|p| p.lanes == lanes)
                    .expect("S₃ is closed, so the composition is in the list");
                assert!(sigma.obs.iter().enumerate().all(|(i, &j)| rho.obs[i] == tau.obs[j as usize]));
                assert!(sigma
                    .action
                    .iter()
                    .enumerate()
                    .all(|(i, &j)| rho.action[i] == tau.action[j as usize]));
            }
        }
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
