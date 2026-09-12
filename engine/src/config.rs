//! Game configuration.
//!
//! `CLAUDE.md`: "Config-driven, no hardcoded constants. Variant selection, deck
//! composition, removal count, draw rules, and stalemate threshold all live in config."
//!
//! Nothing in the rules code reads a magic number; it reads a field of [`GameConfig`].
//! Three presets match the three configurations `PLAN.md` Phase 1 requires, and a tiny
//! `key = value` parser lets `configs/*.toml` override any field.

use std::fmt;

use crate::powers::PowerId;
use crate::rank::Rank;

/// The rules as written, plus this project's two house rules. Every other ruleset is
/// described as a diff against this one.
pub const CANONICAL_POWERS: [PowerId; Rank::COUNT] = [
    PowerId::AceAction,
    PowerId::TwoView,
    PowerId::ThreeTrap,
    PowerId::FourForesight,
    PowerId::FiveFlip,
    PowerId::SixFreeze,
    PowerId::SevenHealAll,
    PowerId::EightRetaliate,
    PowerId::NineNimble,
    PowerId::TenTwinstrike,
    PowerId::JackTaunt,
    PowerId::QueenMove,
    PowerId::KingEmpower,
];

/// A short label naming a ruleset, e.g. `canonical-2026-09`.
///
/// A fixed-size buffer rather than a `String` because [`GameConfig`] is `Copy` and is cloned
/// into every `GameState` — determinization clones states constantly, and a heap allocation
/// per clone would show up in the search path.
///
/// **The name is not part of [`GameConfig::rules_hash`].** Two configs that play the same
/// game are the same ruleset whatever they are called; renaming one must not make old
/// artifacts look foreign.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RulesName([u8; RulesName::CAP]);

impl RulesName {
    pub const CAP: usize = 32;

    /// The rules as written plus the project's house rules, frozen 2026-09-09. Every number
    /// in `FINDINGS.md` that predates the mod system was measured under this.
    pub const CANONICAL: RulesName = RulesName::from_ascii("canonical-2026-09");

    /// Build from a string literal, truncating at [`RulesName::CAP`]. `const` so presets can
    /// use it.
    pub const fn from_ascii(s: &str) -> RulesName {
        let b = s.as_bytes();
        let mut out = [0u8; RulesName::CAP];
        let mut i = 0;
        while i < b.len() && i < RulesName::CAP {
            out[i] = b[i];
            i += 1;
        }
        RulesName(out)
    }

    /// Parse a name from a config file. Restricted to a conservative character set so the
    /// name is safe in a file name, a shard header and a `FINDINGS.md` table cell alike.
    pub fn parse(s: &str) -> Result<RulesName, String> {
        let s = s.trim();
        if s.is_empty() {
            return Err("rules_name must not be empty".into());
        }
        if s.len() > RulesName::CAP {
            return Err(format!(
                "rules_name is {} characters, the limit is {}",
                s.len(),
                RulesName::CAP
            ));
        }
        if let Some(bad) = s
            .chars()
            .find(|c| !(c.is_ascii_alphanumeric() || *c == '-' || *c == '_' || *c == '.'))
        {
            return Err(format!(
                "rules_name may only hold letters, digits, `-`, `_` and `.`; found `{bad}`"
            ));
        }
        Ok(RulesName::from_ascii(s))
    }

    pub fn as_str(&self) -> &str {
        let end = self.0.iter().position(|&b| b == 0).unwrap_or(RulesName::CAP);
        std::str::from_utf8(&self.0[..end]).unwrap_or("?")
    }
}

impl fmt::Display for RulesName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for RulesName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RulesName({:?})", self.as_str())
    }
}

/// Which of the three supported deck configurations is in play.
///
/// `game_rules.md` §9. The split-deck variant is **this project's default**, not the
/// rules-as-written game.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant {
    /// Rules-as-written (`game_rules.md` §2): one shared 52-card deck, one shared draw
    /// pile of 26 after setup, 10 cards removed unseen.
    Base,
    /// §9a. The deck is split by colour; each player owns 26 cards (ranks A–K twice) and
    /// draws only from their own 13-card pile. 5 cards removed unseen *per player*.
    SplitDeck,
    /// §9b. As `SplitDeck`, but both players remove the **same multiset of ranks**, and
    /// that multiset is **revealed to both players**. The two decks are then
    /// rank-identical, which makes this the cleanest target for equilibrium analysis.
    MirroredRemoval,
}

impl Variant {
    /// True when each player draws from their own pile (§9a, §9b) rather than a shared one.
    #[inline]
    pub const fn is_split(self) -> bool {
        matches!(self, Variant::SplitDeck | Variant::MirroredRemoval)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Variant::Base => "base",
            Variant::SplitDeck => "split",
            Variant::MirroredRemoval => "mirrored",
        }
    }

    pub fn parse(s: &str) -> Option<Variant> {
        match s.trim().to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "base" | "raw" | "rulesaswritten" => Some(Variant::Base),
            "split" | "splitdeck" | "9a" => Some(Variant::SplitDeck),
            "mirrored" | "mirroredremoval" | "mirror" | "9b" => Some(Variant::MirroredRemoval),
            _ => None,
        }
    }

    pub const ALL: [Variant; 3] = [
        Variant::Base,
        Variant::SplitDeck,
        Variant::MirroredRemoval,
    ];
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// What the 2's View power does with the card you give back.
///
/// `game_rules.md` §10a. `Bottom` is the project's house rule and the default in every
/// configuration; `Discard` is rules-as-written and exists so Phase 1 can *measure*
/// whether the parity problem the house rule was adopted to fix is real.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TwoPower {
    /// **[HOUSE]** Draw 1, then put a card from hand on the **bottom of your draw pile**.
    /// Pile-neutral and hand-neutral: pure selection.
    Bottom,
    /// **[RAW]** Draw 1, then **discard** a card from hand. Shrinks the pile, which is
    /// exactly the parity lever §10a objects to.
    Discard,
}

impl TwoPower {
    pub const fn label(self) -> &'static str {
        match self {
            TwoPower::Bottom => "bottom",
            TwoPower::Discard => "discard",
        }
    }

    pub fn parse(s: &str) -> Option<TwoPower> {
        match s.trim().to_ascii_lowercase().as_str() {
            "bottom" | "scry" | "house" => Some(TwoPower::Bottom),
            "discard" | "raw" => Some(TwoPower::Discard),
            _ => None,
        }
    }
}

impl fmt::Display for TwoPower {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Everything the engine needs to know that is not part of a position.
///
/// Cloned into each `GameState`, so a state is self-describing and a saved game replays
/// under the rules it was played under.
///
/// `Eq` is deliberately not derived: [`GameConfig::stalemate_value`] is an `f32`, and an
/// `Eq` over a float would be claiming a reflexivity the type does not have. `PartialEq` is
/// what every comparison here actually wants — "were these two runs configured the same" —
/// and nothing uses a config as a hash key.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GameConfig {
    pub variant: Variant,
    pub two_power: TwoPower,

    // ---- Ruleset identity ----
    /// A label for this ruleset. Provenance only — see [`RulesName`]. Excluded from
    /// [`GameConfig::rules_hash`] on purpose.
    pub rules_name: RulesName,

    // ---- Card powers (`MODULAR_RULES.md` §5a) ----
    /// One power variant per rank, indexed by [`Rank::index`].
    ///
    /// Always 13 entries; only `0..=max_rank_index` are reachable. `validate` checks that
    /// each entry belongs to its own rank, so a config cannot put the King's Empower on the
    /// 4.
    pub powers: [PowerId; Rank::COUNT],

    // ---- Tier-1 numeric knobs (`MODULAR_RULES.md` §2) ----
    //
    // Every magic constant the powers used to carry, named and defaulted to the value the
    // rules as written use. Changing one is a config edit with no code change and no layout
    // break, which is what makes Tier 1 the cheapest kind of balance experiment.
    /// Hit points of a card with no HP-granting power, and of **every** face-down card
    /// whatever its rank (`game_rules.md` §5).
    pub default_hp: u8,
    /// Hit points of a face-up card whose live power taunts. 3 in the rules as written.
    ///
    /// Setting this to `default_hp` is a complete ruleset: a Jack that still taunts but dies
    /// as fast as anything else.
    pub jack_hp: u8,
    /// Damage a lone attacker deals (`game_rules.md` §5).
    pub single_attack_damage: u8,
    /// Damage a declared pair deals as one attack (§5).
    pub pair_attack_damage: u8,
    /// Multiplier a nimble attacker gets against a taunting target — the 9's "deals 2 damage
    /// to Jacks" (§6). Applies to the pair total too, so a pair of 9s deals 4.
    pub nimble_vs_taunt_multiplier: u8,
    /// Damage each half of a twinstrike deals (§6).
    pub twinstrike_split_damage: u8,
    /// Damage one retaliate hit deals (§6).
    pub eight_retaliate_damage: u8,
    /// Extra actions an Ace grants when flipped (§6).
    pub ace_bonus_actions: u32,
    /// Attacks an Ace may make on the turn it is flipped (§6).
    pub ace_attack_allowance: u8,
    /// Turns a freeze lasts. 1 in the rules as written — "exactly one of their turns is
    /// lost" (§8).
    pub six_freeze_turns: u32,
    /// Hit points a 7's Heal All restores per card (§6).
    pub seven_heal_amount: u8,

    // ---- Board shape ----
    /// Number of lanes. 3 in every published configuration; a field because Duel52-mini
    /// (`DESIGN.md` §7) uses 1.
    pub lanes: usize,
    /// Lanes a player must win to win the game (`game_rules.md` §7).
    pub lanes_to_win: usize,
    /// Hard bound on cards per side per lane. The rules impose **no limit**
    /// (`game_rules.md` §1: "No limit on cards per lane per side"), so this is not a rule —
    /// it is a capacity the engine asserts against, and it is deliberately set to a value
    /// the game cannot reach.
    ///
    /// `DESIGN.md` §3 suggests 8 as the *encoding* cap on the grounds that it is "far
    /// beyond observed play". That is true of human play and false of random play: a base-
    /// game player pushes up to 31 cards through their hand and random agents spread them
    /// evenly over three lanes, so a lane of 9 or 10 is ordinary. Capping legality at 8
    /// would quietly change the game, and asserting at 8 would crash training runs. So the
    /// presets use the *theoretical* maximum — every card the player could possibly own —
    /// and Phase 1 reports the occupancy actually observed, so Phase 3 can pick a tight
    /// encoding bound from evidence instead of a guess.
    pub max_slots_per_side: usize,
    /// Slots per side per lane that the **neural-network encoder** reserves.
    ///
    /// Separate from [`GameConfig::max_slots_per_side`], and deliberately much smaller.
    /// That field is the engine's legality assertion and is set to a value the game cannot
    /// reach; this one sizes a fixed-shape tensor, so it has to be a bet about what play
    /// actually produces.
    ///
    /// `FINDINGS.md` F2.7 is the authority for the default of **16**. Random play sprawls
    /// to 17–20 cards on one side of one lane, but *competent* play tops out at 8–12
    /// (ISMCTS peaks at 12 over 300 games), because agents that kill things keep lanes
    /// short. 16 sits above every observed value from a real agent and well under the
    /// theoretical 21.
    ///
    /// The encoder **asserts** rather than truncating: silently dropping a card would
    /// change the game the network is looking at. F2.7 also asks for the *distribution* to
    /// be re-measured against the trained agent — not the maximum, which only grows with
    /// the sample — before this is tightened.
    pub encoding_slots: usize,

    // ---- Deck composition ----
    /// Highest rank index in play, inclusive. 12 (King) in the full game; Duel52-mini uses
    /// a smaller value.
    pub max_rank_index: usize,
    /// Copies of each rank in the *shared* deck for `Variant::Base` (4 — one per suit).
    /// In the split variants each player's own deck holds `copies_per_rank / 2` of each
    /// rank, i.e. 2, which is what makes the two decks rank-identical.
    pub copies_per_rank: usize,

    // ---- Deal ----
    /// Cards dealt to each player's hand at setup (`game_rules.md` §2, step 3).
    pub hand_size: usize,
    /// Face-down base cards per player — one per lane (`game_rules.md` §2, step 2).
    pub base_cards_per_player: usize,
    /// Cards removed face-down and unseen at setup. In `Variant::Base` this is the *total*
    /// removed from the shared pile (10). In the split variants it is *per player* (5), so
    /// the overall total is still 10 (`game_rules.md` §9a).
    pub removal_count: usize,

    // ---- Turn structure ----
    /// Actions per turn (`game_rules.md` §4).
    pub actions_per_turn: u32,
    /// Actions on the very first turn of the game, which belongs to `Player::P0`. Two, not
    /// three. The *draw* still happens, so P0 opens at 6 cards in hand.
    pub first_turn_actions: u32,
    /// Cards drawn at the start of each turn, if the relevant pile is non-empty.
    pub draws_per_turn: usize,

    // ---- Termination ----
    /// **[ENGINE]** Consecutive plies (individual player turns) with no damage and no kill
    /// after which the engine declares a draw. Default 20 — ten turns apiece.
    /// `game_rules.md` §7. The published rules define no draw; this is a training
    /// necessity, not a claim about the paper game.
    pub stalemate_quiet_plies: u32,
    /// **[ENGINE]** What an engine-declared stalemate is worth **to a learner**, to both
    /// players, on the `0.0..=1.0` outcome scale. Default `0.5` — the same as any draw,
    /// which is what every measurement before `FINDINGS.md` F3.6 was taken under.
    ///
    /// # Why this exists, and why it is not simply `0.5`
    ///
    /// The stalemate draw is **[ENGINE]**, not a rule: `game_rules.md` §7 records that the
    /// published game defines no draw and that this one exists because the reachable stall
    /// never ends on its own. Scoring it at 0.5 makes "neither player attacks" a *stable
    /// equilibrium of the modified game* — a certain half point beats a risky fight, for
    /// both players, forever. F3.6 is what that looks like when a learner finds it: the
    /// draw rate went 9% → 55% → 88% over three generations while the agent's score against
    /// `random` fell from 0.93 to 0.53.
    ///
    /// Setting this below 0.5 makes refusing to play strictly worse than playing, which is
    /// the incentive the paper game gets for free by having no draw at all. `0.0` — the
    /// training default in `configs/train-fast.toml` — makes it no better than a loss, so a
    /// player who is behind always prefers a gamble to a stall.
    ///
    /// **This is a learning signal, not a scoring rule.** [`crate::Outcome::value_for`] is
    /// untouched and still returns 0.5, so the Elo ladder, `MatchStats` and every number in
    /// `FINDINGS.md` F1 and F2 mean exactly what they meant. Only
    /// [`GameConfig::learning_value`] reads this, and only the search backup and the
    /// training targets call that.
    ///
    /// The **mutual lane win** is deliberately not affected: it is a real outcome that
    /// `game_rules.md` §7 spells out, and a symmetric result deserves a symmetric score.
    pub stalemate_value: f32,
    /// **[ENGINE]** Hard safety cap on total plies. The game is provably finite (total
    /// power activations are bounded, so total healing is bounded, so total damage is
    /// bounded), and the quiet-ply rule already ends stalls, so this should never fire. It
    /// exists so a rules bug during training degrades into a logged draw rather than an
    /// infinite loop.
    pub max_plies: u32,
}

impl GameConfig {
    /// Rules-as-written, one shared deck (`game_rules.md` §2).
    ///
    /// 52 − 6 base − 10 hand − 10 removed = **26 cards** in the shared draw pile.
    pub const fn base() -> GameConfig {
        GameConfig {
            variant: Variant::Base,
            // The house rule is the default in *every* configuration, base game included
            // (`game_rules.md` §10a).
            two_power: TwoPower::Bottom,
            rules_name: RulesName::CANONICAL,
            powers: CANONICAL_POWERS,
            // Every one of these is the rules-as-written value. `MODULAR_RULES.md` §11 step
            // 2: naming a constant must not change it, and `config_round_trips_the_canonical
            // _rules_hash` is what proves none of them moved.
            default_hp: 2,
            jack_hp: 3,
            single_attack_damage: 1,
            pair_attack_damage: 2,
            nimble_vs_taunt_multiplier: 2,
            twinstrike_split_damage: 1,
            eight_retaliate_damage: 1,
            ace_bonus_actions: 1,
            ace_attack_allowance: 2,
            six_freeze_turns: 1,
            seven_heal_amount: 2,
            lanes: 3,
            lanes_to_win: 2,
            // 52 total, minus the 10 removed unseen, minus the opponent's opening 5 and
            // their 3 base cards: 34 cards is the most one player can ever have on the
            // table, so one lane can never exceed it.
            max_slots_per_side: 34,
            // `FINDINGS.md` F2.7. Independent of the variant: the bound is a property of
            // how agents play, not of the deck.
            encoding_slots: 16,
            max_rank_index: 12,
            copies_per_rank: 4,
            hand_size: 5,
            base_cards_per_player: 3,
            removal_count: 10,
            actions_per_turn: 3,
            first_turn_actions: 2,
            draws_per_turn: 1,
            stalemate_quiet_plies: 20,
            // 0.5 keeps every pre-F3.6 measurement meaning what it meant. Training configs
            // override it; see the field's docs.
            stalemate_value: 0.5,
            max_plies: 2000,
        }
    }

    /// **The project default.** Split deck, §9a.
    ///
    /// Per player: 26 − 3 base − 5 hand = 18, remove 5 unseen → a **13-card personal
    /// pile**. Totals match the base game exactly (10 removed overall, 26 cards of draw).
    pub const fn split_deck() -> GameConfig {
        GameConfig {
            variant: Variant::SplitDeck,
            removal_count: 5,
            // A player owns 26 cards and 5 are removed, so 21 is every card they could
            // ever put on the table.
            max_slots_per_side: 21,
            ..GameConfig::base()
        }
    }

    /// Split deck with mirrored removal, §9b. Both players lose the same five ranks, and
    /// the removed multiset is public.
    pub const fn mirrored_removal() -> GameConfig {
        GameConfig {
            variant: Variant::MirroredRemoval,
            ..GameConfig::split_deck()
        }
    }

    /// The preset for a variant, before any per-field overrides.
    pub const fn preset(variant: Variant) -> GameConfig {
        match variant {
            Variant::Base => GameConfig::base(),
            Variant::SplitDeck => GameConfig::split_deck(),
            Variant::MirroredRemoval => GameConfig::mirrored_removal(),
        }
    }

    /// Ranks actually in play. `0..=max_rank_index`.
    #[inline]
    pub const fn rank_count(&self) -> usize {
        self.max_rank_index + 1
    }

    /// Copies of each rank in one player's own deck, in the split variants.
    #[inline]
    pub const fn copies_per_rank_per_player(&self) -> usize {
        self.copies_per_rank / 2
    }

    /// Total cards in one player's colour deck (split variants only).
    #[inline]
    pub const fn split_deck_size(&self) -> usize {
        self.rank_count() * self.copies_per_rank_per_player()
    }

    /// Total cards in the shared deck (base variant).
    #[inline]
    pub const fn full_deck_size(&self) -> usize {
        self.rank_count() * self.copies_per_rank
    }

    /// How many cards end up in the draw pile(s) after setup. For the split variants this
    /// is the size of **each** player's pile; for the base variant it is the single shared
    /// pile.
    pub const fn expected_pile_size(&self) -> usize {
        if self.variant.is_split() {
            self.split_deck_size()
                - self.base_cards_per_player
                - self.hand_size
                - self.removal_count
        } else {
            self.full_deck_size()
                - 2 * self.base_cards_per_player
                - 2 * self.hand_size
                - self.removal_count
        }
    }

    /// Check that the numbers add up, so a bad config fails loudly at setup rather than
    /// producing a subtly wrong game.
    pub fn validate(&self) -> Result<(), String> {
        if self.lanes == 0 {
            return Err("lanes must be at least 1".into());
        }
        if self.lanes_to_win == 0 || self.lanes_to_win > self.lanes {
            return Err(format!(
                "lanes_to_win must be in 1..={}, got {}",
                self.lanes, self.lanes_to_win
            ));
        }
        if self.base_cards_per_player != self.lanes {
            return Err(format!(
                "base_cards_per_player ({}) must equal lanes ({}) — one base card per lane",
                self.base_cards_per_player, self.lanes
            ));
        }
        if self.max_rank_index >= crate::rank::Rank::COUNT {
            return Err(format!(
                "max_rank_index must be < 13, got {}",
                self.max_rank_index
            ));
        }
        if self.variant.is_split() && self.copies_per_rank % 2 != 0 {
            return Err(format!(
                "the split variants halve the deck by colour, so copies_per_rank must be \
                 even; got {}",
                self.copies_per_rank
            ));
        }
        // `expected_pile_size` subtracts on `usize`, so an over-subscribed deal would
        // underflow. Check the arithmetic explicitly instead.
        let (available, needed) = if self.variant.is_split() {
            (
                self.split_deck_size(),
                self.base_cards_per_player + self.hand_size + self.removal_count,
            )
        } else {
            (
                self.full_deck_size(),
                2 * self.base_cards_per_player + 2 * self.hand_size + self.removal_count,
            )
        };
        if needed > available {
            return Err(format!(
                "the deal needs {needed} cards but the deck only has {available}"
            ));
        }
        if self.variant == Variant::MirroredRemoval
            && self.removal_count > self.split_deck_size()
        {
            return Err("removal_count exceeds one player's deck".into());
        }
        if self.max_slots_per_side == 0 {
            return Err("max_slots_per_side must be at least 1".into());
        }
        if self.encoding_slots == 0 {
            return Err("encoding_slots must be at least 1".into());
        }
        if self.encoding_slots > self.max_slots_per_side {
            // Not fatal to the engine, but it means the tensor reserves slots the engine
            // would already have refused to fill — a sign one of the two was edited alone.
            return Err(format!(
                "encoding_slots ({}) exceeds max_slots_per_side ({}), so the encoder \
                 reserves slots the engine would never allow",
                self.encoding_slots, self.max_slots_per_side
            ));
        }
        // A power must belong to the rank it is installed on. Without this, `powers.four =
        // "empower"` would parse (both tokens exist) and produce a game nobody described.
        for (i, power) in self.powers.iter().enumerate() {
            let rank = Rank::from_index(i);
            if power.rank() != rank {
                return Err(format!(
                    "powers.{} is `{}`, which belongs to the {}",
                    rank.config_key(),
                    power.token(),
                    power.rank()
                ));
            }
        }
        if self.default_hp == 0 || self.jack_hp == 0 {
            return Err("hit points must be at least 1".into());
        }
        if self.single_attack_damage == 0 || self.pair_attack_damage == 0 {
            // A zero here would make attacking a no-op action, which reintroduces the pass
            // that `game_rules.md` §4 does not have — see `FINDINGS.md` F2.4b.
            return Err("attack damage must be at least 1, or attacking becomes a pass".into());
        }
        if self.nimble_vs_taunt_multiplier == 0 {
            return Err("nimble_vs_taunt_multiplier must be at least 1".into());
        }
        if self.six_freeze_turns == 0 {
            return Err(
                "six_freeze_turns must be at least 1; use powers.six = \"none\" to remove it".into(),
            );
        }
        if !(0.0..=0.5).contains(&self.stalemate_value) {
            // Above 0.5 would make refusing to play *better* than a draw, which is the
            // pathology in `FINDINGS.md` F3.6 with the sign flipped and worse.
            return Err(format!(
                "stalemate_value must be between 0.0 and 0.5, got {}",
                self.stalemate_value
            ));
        }
        Ok(())
    }

    /// Parse a minimal `key = value` config file.
    ///
    /// Deliberately *not* a real TOML parser — the engine has zero dependencies. Supported
    /// syntax is one `key = value` per line, `#` comments, blank lines, and an optional
    /// `[section]` header which is ignored. Values may be quoted. Unknown keys are an
    /// error rather than being ignored, so a typo in a config file cannot silently change
    /// what was measured.
    ///
    /// `variant` is applied first (it selects the preset); every other key then overrides
    /// the preset, regardless of line order.
    pub fn from_config_str(text: &str) -> Result<GameConfig, String> {
        let mut pairs: Vec<(String, String)> = Vec::new();
        for (lineno, raw) in text.lines().enumerate() {
            let line = match raw.find('#') {
                Some(i) => &raw[..i],
                None => raw,
            }
            .trim();
            if line.is_empty() || (line.starts_with('[') && line.ends_with(']')) {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                return Err(format!("line {}: expected `key = value`, got `{line}`", lineno + 1));
            };
            let value = v.trim().trim_matches(|c| c == '"' || c == '\'').to_string();
            pairs.push((k.trim().to_ascii_lowercase(), value));
        }

        // Pass 1: the variant, which picks the base preset.
        let mut cfg = GameConfig::split_deck();
        for (k, v) in &pairs {
            if k == "variant" {
                cfg.variant = Variant::parse(v)
                    .ok_or_else(|| format!("unknown variant `{v}`"))?;
                cfg = GameConfig::preset(cfg.variant);
            }
        }

        // Pass 2: everything else.
        fn num<T: std::str::FromStr>(k: &str, v: &str) -> Result<T, String> {
            v.parse::<T>()
                .map_err(|_| format!("key `{k}`: `{v}` is not a valid number"))
        }
        for (k, v) in &pairs {
            // `powers.<rank> = <token>`. The rank scopes the token, so `"none"` under
            // `powers.three` and under `powers.eight` are different variants and both
            // round-trip.
            if let Some(rank_key) = k.strip_prefix("powers.") {
                let rank = Rank::from_config_key(rank_key)
                    .ok_or_else(|| format!("unknown rank `{rank_key}` in key `{k}`"))?;
                let power = PowerId::parse(rank, v).ok_or_else(|| {
                    let choices: Vec<&str> = PowerId::variants_for(rank)
                        .iter()
                        .map(|p| p.token())
                        .collect();
                    format!(
                        "unknown power `{v}` for the {rank}; implemented: {}",
                        choices.join(" | ")
                    )
                })?;
                cfg.powers[rank.index()] = power;
                continue;
            }
            match k.as_str() {
                "variant" => {}
                "include" => {
                    return Err(
                        "`include` needs a file to resolve paths against; load this config \
                         with `GameConfig::from_config_file` (the CLI's --config) rather \
                         than from a bare string"
                            .into(),
                    )
                }
                "rules_name" => cfg.rules_name = RulesName::parse(v)?,
                "two_power" => {
                    cfg.two_power =
                        TwoPower::parse(v).ok_or_else(|| format!("unknown two_power `{v}`"))?
                }
                "default_hp" => cfg.default_hp = num(k, v)?,
                "jack_hp" => cfg.jack_hp = num(k, v)?,
                "single_attack_damage" => cfg.single_attack_damage = num(k, v)?,
                "pair_attack_damage" => cfg.pair_attack_damage = num(k, v)?,
                "nimble_vs_taunt_multiplier" => cfg.nimble_vs_taunt_multiplier = num(k, v)?,
                "twinstrike_split_damage" => cfg.twinstrike_split_damage = num(k, v)?,
                "eight_retaliate_damage" => cfg.eight_retaliate_damage = num(k, v)?,
                "ace_bonus_actions" => cfg.ace_bonus_actions = num(k, v)?,
                "ace_attack_allowance" => cfg.ace_attack_allowance = num(k, v)?,
                "six_freeze_turns" => cfg.six_freeze_turns = num(k, v)?,
                "seven_heal_amount" => cfg.seven_heal_amount = num(k, v)?,
                "lanes" => cfg.lanes = num(k, v)?,
                "lanes_to_win" => cfg.lanes_to_win = num(k, v)?,
                "max_slots_per_side" => cfg.max_slots_per_side = num(k, v)?,
                "encoding_slots" => cfg.encoding_slots = num(k, v)?,
                "max_rank_index" => cfg.max_rank_index = num(k, v)?,
                "copies_per_rank" => cfg.copies_per_rank = num(k, v)?,
                "hand_size" => cfg.hand_size = num(k, v)?,
                "base_cards_per_player" => cfg.base_cards_per_player = num(k, v)?,
                "removal_count" => cfg.removal_count = num(k, v)?,
                "actions_per_turn" => cfg.actions_per_turn = num(k, v)?,
                "first_turn_actions" => cfg.first_turn_actions = num(k, v)?,
                "draws_per_turn" => cfg.draws_per_turn = num(k, v)?,
                "stalemate_quiet_plies" => cfg.stalemate_quiet_plies = num(k, v)?,
                "stalemate_value" => {
                    cfg.stalemate_value = v
                        .parse::<f32>()
                        .map_err(|_| format!("{k}: `{v}` is not a number"))?
                }
                "max_plies" => cfg.max_plies = num(k, v)?,
                other => return Err(format!("unknown config key `{other}`")),
            }
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Render back out in the same format `from_config_str` reads. Used to stamp the exact
    /// configuration into a results file, so a finding is reproducible.
    ///
    /// **Fully resolved**: `include` lines never appear here, because this string is what
    /// goes into the shard and the game record and it has to be self-contained
    /// (`MODULAR_RULES.md` §5d).
    pub fn to_config_string(&self) -> String {
        let mut s = format!(
            "rules_name = \"{}\"\n\
             variant = \"{}\"\n\
             two_power = \"{}\"\n\
             lanes = {}\n\
             lanes_to_win = {}\n\
             max_slots_per_side = {}\n\
             encoding_slots = {}\n\
             max_rank_index = {}\n\
             copies_per_rank = {}\n\
             hand_size = {}\n\
             base_cards_per_player = {}\n\
             removal_count = {}\n\
             actions_per_turn = {}\n\
             first_turn_actions = {}\n\
             draws_per_turn = {}\n\
             stalemate_quiet_plies = {}\n\
             stalemate_value = {}\n\
             max_plies = {}\n\
             default_hp = {}\n\
             jack_hp = {}\n\
             single_attack_damage = {}\n\
             pair_attack_damage = {}\n\
             nimble_vs_taunt_multiplier = {}\n\
             twinstrike_split_damage = {}\n\
             eight_retaliate_damage = {}\n\
             ace_bonus_actions = {}\n\
             ace_attack_allowance = {}\n\
             six_freeze_turns = {}\n\
             seven_heal_amount = {}\n",
            self.rules_name,
            self.variant,
            self.two_power,
            self.lanes,
            self.lanes_to_win,
            self.max_slots_per_side,
            self.encoding_slots,
            self.max_rank_index,
            self.copies_per_rank,
            self.hand_size,
            self.base_cards_per_player,
            self.removal_count,
            self.actions_per_turn,
            self.first_turn_actions,
            self.draws_per_turn,
            self.stalemate_quiet_plies,
            self.stalemate_value,
            self.max_plies,
            self.default_hp,
            self.jack_hp,
            self.single_attack_damage,
            self.pair_attack_damage,
            self.nimble_vs_taunt_multiplier,
            self.twinstrike_split_damage,
            self.eight_retaliate_damage,
            self.ace_bonus_actions,
            self.ace_attack_allowance,
            self.six_freeze_turns,
            self.seven_heal_amount,
        );
        // One line per card, always all thirteen, in rank order. Emitting the whole table
        // rather than only the diffs is what makes a stamped config answer "what were the
        // rules" without needing to know what the defaults were on the day it was written.
        for rank in Rank::ALL {
            let _ = std::fmt::Write::write_fmt(
                &mut s,
                format_args!(
                    "powers.{} = \"{}\"\n",
                    rank.config_key(),
                    self.powers[rank.index()].token()
                ),
            );
        }
        s
    }

    /// The power installed on `rank` in this ruleset.
    #[inline]
    pub fn power(&self, rank: Rank) -> PowerId {
        self.powers[rank.index()]
    }

    /// Does this ruleset need the **extended encoder layout**? `MODULAR_RULES.md` §7.
    ///
    /// True exactly when some installed power declares
    /// [`PowerId::needs_extended_encoder`]. Everything the reserve adds — five spare phase
    /// one-hot positions, eight per-slot status flags, and the `CHOOSE_LANE` and
    /// `CHOOSE_OPTION` policy blocks — is switched on and off by this one predicate, so
    /// there are exactly **two** layouts in the codebase and never a spectrum of them.
    ///
    /// # Why it is derived rather than a config key
    ///
    /// A key would be a third thing to keep in step with the powers and the layout, and the
    /// way it fails is silent: a ruleset that installs a flag-using power but forgets the
    /// key writes a status nobody encodes, and the network simply never learns the mechanic.
    /// Deriving it makes that state unrepresentable. It also means **the canonical ruleset
    /// can never accidentally move**: no canonical power declares the reserve, so
    /// `obs_layout_hash` is bit-identical to the pre-reserve build and every checkpoint and
    /// shard in the repository still loads.
    ///
    /// The cost is that a reserve ruleset cannot warm-start from a base-layout checkpoint
    /// directly — `encode::reserve_embedding` and `python -m duel52.nn widen` are the bridge
    /// that makes that a 3-hour run instead of a 24-hour one.
    #[inline]
    pub const fn extended_encoder(&self) -> bool {
        // A plain loop rather than `iter().any()` so this stays usable from `const fn`
        // callers in `encode`, and because 13 entries is not worth an iterator.
        let mut i = 0;
        while i < self.powers.len() {
            if self.powers[i].needs_extended_encoder() {
                return true;
            }
            i += 1;
        }
        false
    }

    /// A 64-bit fingerprint of **the game these rules describe**.
    ///
    /// `MODULAR_RULES.md` §6. This is the provenance the project did not have: nothing
    /// recorded which ruleset produced a number, so a checkpoint trained on the split deck
    /// would play `--variant base` at full speed with no warning, and the resulting score
    /// looked exactly like a result.
    ///
    /// # What is in, and what is out
    ///
    /// In: the variant, every card power, every Tier-1 number, deck composition, the deal,
    /// and turn structure — everything that changes what a legal game looks like.
    ///
    /// Out, deliberately:
    ///
    /// - [`GameConfig::rules_name`], because a label is not a rule.
    /// - [`GameConfig::stalemate_value`], because it is a *learning* weight and never
    ///   reaches [`crate::Outcome`]. Two runs that differ only here played the same game.
    /// - [`GameConfig::encoding_slots`], because it sizes a tensor rather than the game, and
    ///   the layout hashes in `encode.rs` already pin it.
    ///
    /// ⚠️ Do **not** extend that exclusion list on a judgment call about relevance. A field
    /// wrongly excluded makes two different rulesets hash the same, which is the silent
    /// collision this exists to prevent; a field wrongly included merely makes two identical
    /// rulesets look different, which is a labelling annoyance you will notice.
    pub fn rules_hash(&self) -> u64 {
        // FNV-1a, the same construction `encode.rs` uses for the layout hashes. Chosen for
        // the same reason: it is four lines, has no dependencies, and this is a fingerprint
        // rather than a security primitive.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in self.rules_string().bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// The exact text [`GameConfig::rules_hash`] fingerprints. Public so a mismatch can be
    /// diffed rather than guessed at.
    pub fn rules_string(&self) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(512);
        let _ = writeln!(s, "variant={}", self.variant);
        let _ = writeln!(s, "two_power={}", self.two_power);
        for rank in Rank::ALL {
            let _ = writeln!(
                s,
                "power.{}={}",
                rank.config_key(),
                self.powers[rank.index()].token()
            );
        }
        let _ = writeln!(s, "default_hp={}", self.default_hp);
        let _ = writeln!(s, "jack_hp={}", self.jack_hp);
        let _ = writeln!(s, "single_attack_damage={}", self.single_attack_damage);
        let _ = writeln!(s, "pair_attack_damage={}", self.pair_attack_damage);
        let _ = writeln!(
            s,
            "nimble_vs_taunt_multiplier={}",
            self.nimble_vs_taunt_multiplier
        );
        let _ = writeln!(s, "twinstrike_split_damage={}", self.twinstrike_split_damage);
        let _ = writeln!(s, "eight_retaliate_damage={}", self.eight_retaliate_damage);
        let _ = writeln!(s, "ace_bonus_actions={}", self.ace_bonus_actions);
        let _ = writeln!(s, "ace_attack_allowance={}", self.ace_attack_allowance);
        let _ = writeln!(s, "six_freeze_turns={}", self.six_freeze_turns);
        let _ = writeln!(s, "seven_heal_amount={}", self.seven_heal_amount);
        let _ = writeln!(s, "lanes={}", self.lanes);
        let _ = writeln!(s, "lanes_to_win={}", self.lanes_to_win);
        let _ = writeln!(s, "max_slots_per_side={}", self.max_slots_per_side);
        let _ = writeln!(s, "max_rank_index={}", self.max_rank_index);
        let _ = writeln!(s, "copies_per_rank={}", self.copies_per_rank);
        let _ = writeln!(s, "hand_size={}", self.hand_size);
        let _ = writeln!(s, "base_cards_per_player={}", self.base_cards_per_player);
        let _ = writeln!(s, "removal_count={}", self.removal_count);
        let _ = writeln!(s, "actions_per_turn={}", self.actions_per_turn);
        let _ = writeln!(s, "first_turn_actions={}", self.first_turn_actions);
        let _ = writeln!(s, "draws_per_turn={}", self.draws_per_turn);
        let _ = writeln!(s, "stalemate_quiet_plies={}", self.stalemate_quiet_plies);
        let _ = writeln!(s, "max_plies={}", self.max_plies);
        s
    }

    /// `name/hash`, the form that goes in a result header and a `FINDINGS.md` row.
    pub fn rules_label(&self) -> String {
        format!("{}/{:016x}", self.rules_name, self.rules_hash())
    }

    /// True when this is the rules as written plus the project's house rules — the ruleset
    /// every pre-2026-09-09 number in `FINDINGS.md` was measured under.
    pub fn is_canonical_rules(&self) -> bool {
        self.rules_hash() == GameConfig::preset(self.variant).rules_hash()
    }

    /// Load a config file, resolving `include` directives relative to the file's own
    /// directory.
    ///
    /// `MODULAR_RULES.md` §5d. Semantics:
    ///
    /// - Includes are pulled in **where they appear**, depth first.
    /// - Later keys win, so a key written in the including file overrides the same key from
    ///   an include above it.
    /// - A file may be included more than once in one resolution as long as it does not
    ///   include itself, directly or through a chain. A cycle is an error rather than a
    ///   truncation, because a silently-dropped include is a ruleset nobody described.
    pub fn from_config_file(path: &std::path::Path) -> Result<GameConfig, String> {
        let mut stack = Vec::new();
        let text = resolve_includes(path, &mut stack)?;
        GameConfig::from_config_str(&text).map_err(|e| format!("`{}`: {e}", path.display()))
    }

    /// What `outcome` is worth **to a learner**, for `player`, on the `0.0..=1.0` scale.
    ///
    /// This is [`Outcome::value_for`] with one substitution: an engine-declared stalemate
    /// (and the ply-cap draw, which is a bug report) is worth [`Self::stalemate_value`] to
    /// *both* players rather than half a point each. A mutual lane win stays at 0.5, because
    /// it is a real outcome in `game_rules.md` §7 rather than an engine artefact.
    ///
    /// **Deliberately not zero-sum.** Both players can score 0 here, and that is the point:
    /// a stalemate is both players declining to play, so both should prefer nearly anything
    /// else. `net_mcts` banks a reward per player rather than one number and a sign, so a
    /// non-zero-sum terminal backs up correctly — see the caveat in that module about the
    /// value head, which can only report one side's estimate at a leaf.
    ///
    /// Called by exactly two places: the terminal backup in `agents/net_mcts.rs` and the
    /// value targets in `selfplay.rs`. **Scoring never calls it** — `MatchStats`, the Elo
    /// fit and every `FINDINGS.md` F1/F2 number go through [`Outcome::value_for`], so a
    /// training run cannot move the benchmark it is measured against.
    pub fn learning_value(&self, outcome: crate::outcome::Outcome, player: crate::Player) -> f32 {
        use crate::outcome::{DrawReason, Outcome};
        match outcome {
            Outcome::Draw(DrawReason::Stalemate) | Outcome::Draw(DrawReason::PlyLimit) => {
                self.stalemate_value
            }
            other => other.value_for(player),
        }
    }

    /// One-line summary for log headers.
    pub fn summary(&self) -> String {
        format!(
            "rules={} variant={} two_power={} stalemate={}plies",
            self.rules_label(),
            self.variant,
            self.two_power,
            self.stalemate_quiet_plies
        )
    }

    /// The cards whose power differs from the rules as written, as `3=trap_vengeance…`.
    /// Empty when nothing was modded. Used by result headers, which want the diff rather
    /// than all thirteen rows.
    pub fn power_diff(&self) -> Vec<String> {
        Rank::ALL
            .into_iter()
            .filter(|r| r.index() <= self.max_rank_index)
            .filter(|r| self.powers[r.index()] != CANONICAL_POWERS[r.index()])
            .map(|r| format!("{r}={}", self.powers[r.index()].token()))
            .collect()
    }
}

/// Read `path` and splice in every `include`, depth first, returning one flat config text.
///
/// `stack` carries the chain of files currently being resolved, which is both the cycle
/// guard and the error message.
fn resolve_includes(
    path: &std::path::Path,
    stack: &mut Vec<std::path::PathBuf>,
) -> Result<String, String> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if stack.contains(&canonical) {
        let chain: Vec<String> = stack
            .iter()
            .chain(std::iter::once(&canonical))
            .map(|p| p.display().to_string())
            .collect();
        return Err(format!("include cycle: {}", chain.join(" -> ")));
    }
    stack.push(canonical);

    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read `{}`: {e}", path.display()))?;
    let dir = path.parent().unwrap_or(std::path::Path::new("."));

    let mut out = String::with_capacity(text.len() * 2);
    for raw in text.lines() {
        let line = match raw.find('#') {
            Some(i) => &raw[..i],
            None => raw,
        }
        .trim();
        let included = line.split_once('=').and_then(|(k, v)| {
            (k.trim().eq_ignore_ascii_case("include"))
                .then(|| v.trim().trim_matches(|c| c == '"' || c == '\'').to_string())
        });
        match included {
            Some(rel) => {
                let child = dir.join(&rel);
                out.push_str(&format!("# --- begin include {rel} ---\n"));
                out.push_str(&resolve_includes(&child, stack)?);
                out.push_str(&format!("# --- end include {rel} ---\n"));
            }
            None => {
                out.push_str(raw);
                out.push('\n');
            }
        }
    }

    stack.pop();
    Ok(out)
}

impl Default for GameConfig {
    /// The project default is the **split-deck** variant, not rules-as-written
    /// (`game_rules.md` §9).
    fn default() -> GameConfig {
        GameConfig::split_deck()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `game_rules.md` §2 step 5: "Remaining shared draw pile: 52 − 6 − 10 − 10 = 26."
    #[test]
    fn rule_2_base_game_pile_is_26_cards() {
        assert_eq!(GameConfig::base().expected_pile_size(), 26);
    }

    /// `game_rules.md` §9a: "per player, 26 − 3 base − 5 hand = 18, then remove 5 unseen →
    /// a 13-card personal draw pile."
    #[test]
    fn rule_9a_split_deck_pile_is_13_cards_per_player() {
        assert_eq!(GameConfig::split_deck().expected_pile_size(), 13);
        assert_eq!(GameConfig::split_deck().split_deck_size(), 26);
    }

    /// §9a: the split variant "preserves the base game's totals exactly (10 cards removed
    /// overall, 26 cards of draw across both players)."
    #[test]
    fn rule_9a_split_deck_preserves_base_game_totals() {
        let base = GameConfig::base();
        let split = GameConfig::split_deck();
        assert_eq!(2 * split.removal_count, base.removal_count);
        assert_eq!(2 * split.expected_pile_size(), base.expected_pile_size());
    }

    #[test]
    fn every_preset_validates() {
        for v in Variant::ALL {
            GameConfig::preset(v).validate().expect("preset must be valid");
        }
    }

    #[test]
    fn config_files_round_trip() {
        for v in Variant::ALL {
            let cfg = GameConfig::preset(v);
            let text = cfg.to_config_string();
            assert_eq!(GameConfig::from_config_str(&text).unwrap(), cfg);
        }
    }

    #[test]
    fn config_parsing_applies_variant_before_overrides_regardless_of_order() {
        // `removal_count` is written *before* `variant`; the variant preset must not
        // clobber it.
        let cfg = GameConfig::from_config_str(
            "removal_count = 4\n# a comment\nvariant = \"base\"\n",
        )
        .unwrap();
        assert_eq!(cfg.variant, Variant::Base);
        assert_eq!(cfg.removal_count, 4);
    }

    #[test]
    fn unknown_config_keys_are_rejected() {
        assert!(GameConfig::from_config_str("stalemate_quiet_ply = 9\n").is_err());
    }

    /// `FINDINGS.md` F2.7: 16, above every value competent self-play produced and below
    /// the theoretical 21. Separate from `max_slots_per_side`, which stays the engine's
    /// legality assertion — F2.7's saving is the *tensor*, not the rules.
    #[test]
    fn finding_2_7_encoding_slots_default_to_sixteen_in_every_variant() {
        for v in Variant::ALL {
            let cfg = GameConfig::preset(v);
            assert_eq!(cfg.encoding_slots, 16);
            assert!(
                cfg.encoding_slots < cfg.max_slots_per_side,
                "the encoding bound must stay strictly under the legality bound"
            );
        }
    }

    #[test]
    fn encoding_slots_above_the_legality_bound_is_rejected() {
        let mut cfg = GameConfig::split_deck();
        cfg.encoding_slots = cfg.max_slots_per_side + 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn over_subscribed_deals_are_rejected_not_underflowed() {
        let mut cfg = GameConfig::base();
        cfg.hand_size = 30;
        assert!(cfg.validate().is_err());
    }
}
