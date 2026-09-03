//! Game configuration.
//!
//! `CLAUDE.md`: "Config-driven, no hardcoded constants. Variant selection, deck
//! composition, removal count, draw rules, and stalemate threshold all live in config."
//!
//! Nothing in the rules code reads a magic number; it reads a field of [`GameConfig`].
//! Three presets match the three configurations `PLAN.md` Phase 1 requires, and a tiny
//! `key = value` parser lets `configs/*.toml` override any field.

use std::fmt;

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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GameConfig {
    pub variant: Variant,
    pub two_power: TwoPower,

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
            lanes: 3,
            lanes_to_win: 2,
            // 52 total, minus the 10 removed unseen, minus the opponent's opening 5 and
            // their 3 base cards: 34 cards is the most one player can ever have on the
            // table, so one lane can never exceed it.
            max_slots_per_side: 34,
            max_rank_index: 12,
            copies_per_rank: 4,
            hand_size: 5,
            base_cards_per_player: 3,
            removal_count: 10,
            actions_per_turn: 3,
            first_turn_actions: 2,
            draws_per_turn: 1,
            stalemate_quiet_plies: 20,
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
            match k.as_str() {
                "variant" => {}
                "two_power" => {
                    cfg.two_power =
                        TwoPower::parse(v).ok_or_else(|| format!("unknown two_power `{v}`"))?
                }
                "lanes" => cfg.lanes = num(k, v)?,
                "lanes_to_win" => cfg.lanes_to_win = num(k, v)?,
                "max_slots_per_side" => cfg.max_slots_per_side = num(k, v)?,
                "max_rank_index" => cfg.max_rank_index = num(k, v)?,
                "copies_per_rank" => cfg.copies_per_rank = num(k, v)?,
                "hand_size" => cfg.hand_size = num(k, v)?,
                "base_cards_per_player" => cfg.base_cards_per_player = num(k, v)?,
                "removal_count" => cfg.removal_count = num(k, v)?,
                "actions_per_turn" => cfg.actions_per_turn = num(k, v)?,
                "first_turn_actions" => cfg.first_turn_actions = num(k, v)?,
                "draws_per_turn" => cfg.draws_per_turn = num(k, v)?,
                "stalemate_quiet_plies" => cfg.stalemate_quiet_plies = num(k, v)?,
                "max_plies" => cfg.max_plies = num(k, v)?,
                other => return Err(format!("unknown config key `{other}`")),
            }
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Render back out in the same format `from_config_str` reads. Used to stamp the exact
    /// configuration into a results file, so a finding is reproducible.
    pub fn to_config_string(&self) -> String {
        format!(
            "variant = \"{}\"\n\
             two_power = \"{}\"\n\
             lanes = {}\n\
             lanes_to_win = {}\n\
             max_slots_per_side = {}\n\
             max_rank_index = {}\n\
             copies_per_rank = {}\n\
             hand_size = {}\n\
             base_cards_per_player = {}\n\
             removal_count = {}\n\
             actions_per_turn = {}\n\
             first_turn_actions = {}\n\
             draws_per_turn = {}\n\
             stalemate_quiet_plies = {}\n\
             max_plies = {}\n",
            self.variant,
            self.two_power,
            self.lanes,
            self.lanes_to_win,
            self.max_slots_per_side,
            self.max_rank_index,
            self.copies_per_rank,
            self.hand_size,
            self.base_cards_per_player,
            self.removal_count,
            self.actions_per_turn,
            self.first_turn_actions,
            self.draws_per_turn,
            self.stalemate_quiet_plies,
            self.max_plies,
        )
    }

    /// One-line summary for log headers.
    pub fn summary(&self) -> String {
        format!(
            "variant={} two_power={} stalemate={}plies",
            self.variant, self.two_power, self.stalemate_quiet_plies
        )
    }
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

    #[test]
    fn over_subscribed_deals_are_rejected_not_underflowed() {
        let mut cfg = GameConfig::base();
        cfg.hand_size = 30;
        assert!(cfg.validate().is_err());
    }
}
