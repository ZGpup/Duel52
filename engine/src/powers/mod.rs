//! Card powers, one module per rank.
//!
//! `MODULAR_RULES.md` §5. Every rule that is specific to a *card* lives here, so that
//! changing what the 3 does touches [`three`] and nothing else. The turn machinery in
//! `apply.rs` no longer names a rank: it looks up `config.powers[rank]` and dispatches.
//!
//! # Why an enum and not a trait object
//!
//! [`crate::config::GameConfig`] is `Copy + PartialEq` and is serialised into every shard
//! and every game record. `[PowerId; 13]` preserves all three properties for free;
//! `[&'static dyn CardRules; 13]` preserves none of them. Dispatch is a `match` on a small
//! enum, so there is no dynamic dispatch in the search hot path, and `GameState` stays
//! `Clone + PartialEq`, which matters because determinization clones states constantly.
//!
//! # Why a variant carries no data
//!
//! `MODULAR_RULES.md` §5a. A number that varies within a shape gets its own **name** —
//! [`PowerId::ThreeTrapVengeance1`] and [`PowerId::ThreeTrapVengeance2`] rather than one
//! variant plus a `three_vengeance_damage` config key. A config key that only applies under
//! some other key's value is a key `from_config_str` cannot reject when it is inapplicable,
//! and an inert-but-hashed key is a silent provenance bug. The named form cannot express one.
//!
//! The **Tier-1** numeric fields on `GameConfig` are a different thing and still exist: they
//! parameterise powers that are present in the canonical ruleset (`jack_hp`,
//! `eight_retaliate_damage`, `seven_heal_amount`). The rule is only that a *new variant*
//! gets a name rather than a knob.
//!
//! # Adding a power
//!
//! 1. Add the variant to [`PowerId`], next to its rank's other variants.
//! 2. Add it to that rank's `VARIANTS` list, and give it a config token in `token()`.
//! 3. Implement its behaviour in the rank's module.
//! 4. The compiler will now list every `match` that does not handle it. That exhaustiveness
//!    is the whole reason this is an enum, so do not add a `_ =>` arm to silence it.
//! 5. Give it a named test in the rank's module, per `CLAUDE.md`.

use crate::card::CardId;
use crate::damage::DamageSource;
use crate::player::Player;
use crate::rank::Rank;
use crate::state::GameState;

pub mod ace;
pub mod eight;
pub mod five;
pub mod four;
pub mod jack;
pub mod king;
pub mod nine;
pub mod queen;
pub mod seven;
pub mod six;
pub mod ten;
pub mod three;
pub mod two;

/// Where a power is firing from. Resolved fresh at each firing, because slots compact on a
/// kill and shift when a Queen moves a card.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PowerCtx {
    pub id: CardId,
    pub owner: Player,
    pub lane: usize,
    pub side: usize,
    pub slot: usize,
}

/// What happens to a card whose damage has reached its hit points.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LethalOutcome {
    /// Goes to the discard pile.
    Die,
    /// Stayed on the table. The hook has already put the card into whatever state it
    /// survives in — the 3's Trap returns it face-up at full HP.
    Restored,
}

/// When a card with a retaliate-shaped power hits back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RetaliateMode {
    /// No retaliation.
    Never,
    /// **Rules as written** (`game_rules.md` §6, §8): "fires even if that damage killed the
    /// 8". The set of retaliators is read *before* damage lands.
    Always,
    /// Only if the card is still alive once the attack's damage has been applied.
    OnSurvival,
}

/// Every implemented power.
///
/// The name is `<Rank><Shape>`, and the rank prefix is not decoration: `powers[i]` is
/// validated to hold a variant whose [`PowerId::rank`] is `i`, so a config cannot put the
/// King's Empower on the 4.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PowerId {
    // ---- A ----
    AceAction,
    // ---- 2 ----
    TwoView,
    // ---- 3 ----
    ThreeTrap,
    ThreeTrapVengeance1,
    ThreeTrapVengeance2,
    ThreeNone,
    // ---- 4 ----
    FourForesight,
    // ---- 5 ----
    FiveFlip,
    // ---- 6 ----
    SixFreeze,
    // ---- 7 ----
    SevenHealAll,
    // ---- 8 ----
    EightRetaliate,
    EightRetaliateOnSurvival,
    EightNone,
    // ---- 9 ----
    NineNimble,
    // ---- 10 ----
    TenTwinstrike,
    // ---- J ----
    JackTaunt,
    // ---- Q ----
    QueenMove,
    // ---- K ----
    KingEmpower,
}

impl PowerId {
    /// Every implemented power, in rank order.
    pub const ALL: &'static [PowerId] = &[
        PowerId::AceAction,
        PowerId::TwoView,
        PowerId::ThreeTrap,
        PowerId::ThreeTrapVengeance1,
        PowerId::ThreeTrapVengeance2,
        PowerId::ThreeNone,
        PowerId::FourForesight,
        PowerId::FiveFlip,
        PowerId::SixFreeze,
        PowerId::SevenHealAll,
        PowerId::EightRetaliate,
        PowerId::EightRetaliateOnSurvival,
        PowerId::EightNone,
        PowerId::NineNimble,
        PowerId::TenTwinstrike,
        PowerId::JackTaunt,
        PowerId::QueenMove,
        PowerId::KingEmpower,
    ];

    /// The rank this power belongs to. A power is never legal on another rank.
    pub const fn rank(self) -> Rank {
        match self {
            PowerId::AceAction => Rank::ACE,
            PowerId::TwoView => Rank::TWO,
            PowerId::ThreeTrap
            | PowerId::ThreeTrapVengeance1
            | PowerId::ThreeTrapVengeance2
            | PowerId::ThreeNone => Rank::THREE,
            PowerId::FourForesight => Rank::FOUR,
            PowerId::FiveFlip => Rank::FIVE,
            PowerId::SixFreeze => Rank::SIX,
            PowerId::SevenHealAll => Rank::SEVEN,
            PowerId::EightRetaliate
            | PowerId::EightRetaliateOnSurvival
            | PowerId::EightNone => Rank::EIGHT,
            PowerId::NineNimble => Rank::NINE,
            PowerId::TenTwinstrike => Rank::TEN,
            PowerId::JackTaunt => Rank::JACK,
            PowerId::QueenMove => Rank::QUEEN,
            PowerId::KingEmpower => Rank::KING,
        }
    }

    /// The canonical power for a rank — the rules as written, before any mod.
    pub const fn canonical_for(rank: Rank) -> PowerId {
        match rank.index() {
            0 => PowerId::AceAction,
            1 => PowerId::TwoView,
            2 => PowerId::ThreeTrap,
            3 => PowerId::FourForesight,
            4 => PowerId::FiveFlip,
            5 => PowerId::SixFreeze,
            6 => PowerId::SevenHealAll,
            7 => PowerId::EightRetaliate,
            8 => PowerId::NineNimble,
            9 => PowerId::TenTwinstrike,
            10 => PowerId::JackTaunt,
            11 => PowerId::QueenMove,
            12 => PowerId::KingEmpower,
            _ => PowerId::AceAction,
        }
    }

    /// The token this power is written as in a config file, e.g. `powers.three = "trap"`.
    ///
    /// Tokens are scoped to their rank, so `"none"` can mean a different variant under
    /// `powers.three` than under `powers.eight` and still round-trip.
    pub const fn token(self) -> &'static str {
        match self {
            PowerId::AceAction => "action",
            PowerId::TwoView => "view",
            PowerId::ThreeTrap => "trap",
            PowerId::ThreeTrapVengeance1 => "trap_vengeance_one_damage",
            PowerId::ThreeTrapVengeance2 => "trap_vengeance_two_damage",
            PowerId::ThreeNone => "none",
            PowerId::FourForesight => "foresight",
            PowerId::FiveFlip => "flip",
            PowerId::SixFreeze => "freeze",
            PowerId::SevenHealAll => "heal_all",
            PowerId::EightRetaliate => "retaliate",
            PowerId::EightRetaliateOnSurvival => "retaliate_on_survival",
            PowerId::EightNone => "none",
            PowerId::NineNimble => "nimble",
            PowerId::TenTwinstrike => "twinstrike",
            PowerId::JackTaunt => "taunt",
            PowerId::QueenMove => "move",
            PowerId::KingEmpower => "empower",
        }
    }

    /// Parse a token *in the context of a rank*, so `"none"` resolves unambiguously.
    pub fn parse(rank: Rank, token: &str) -> Option<PowerId> {
        let t = token.trim().to_ascii_lowercase().replace('-', "_");
        PowerId::ALL
            .iter()
            .copied()
            .find(|p| p.rank() == rank && p.token() == t)
    }

    /// Every implemented variant for a rank. Used by the CLI and by the config error
    /// message that lists the alternatives.
    pub fn variants_for(rank: Rank) -> Vec<PowerId> {
        PowerId::ALL
            .iter()
            .copied()
            .filter(|p| p.rank() == rank)
            .collect()
    }

    /// The power's name as printed on the card, for the CLI.
    pub const fn display_name(self) -> &'static str {
        match self {
            PowerId::AceAction => "Action",
            PowerId::TwoView => "View",
            PowerId::ThreeTrap => "Trap",
            PowerId::ThreeTrapVengeance1 | PowerId::ThreeTrapVengeance2 => "Trap + Vengeance",
            PowerId::ThreeNone | PowerId::EightNone => "(none)",
            PowerId::FourForesight => "Foresight",
            PowerId::FiveFlip => "Flip",
            PowerId::SixFreeze => "Freeze",
            PowerId::SevenHealAll => "Heal All",
            PowerId::EightRetaliate => "Retaliate",
            PowerId::EightRetaliateOnSurvival => "Retaliate (on survival)",
            PowerId::NineNimble => "Nimble",
            PowerId::TenTwinstrike => "Twinstrike",
            PowerId::JackTaunt => "Taunt",
            PowerId::QueenMove => "Move",
            PowerId::KingEmpower => "Empower",
        }
    }

    /// A one-line summary, for the CLI's `powers` screen.
    pub const fn text(self) -> &'static str {
        match self {
            PowerId::AceAction => {
                "one-shot: +1 action this turn; this Ace may attack twice this turn"
            }
            PowerId::TwoView => {
                "one-shot: draw 1 from your pile, then put a card from hand on the bottom"
            }
            PowerId::ThreeTrap => {
                "if killed while FACE-DOWN, returns face-up at full HP in the same lane"
            }
            PowerId::ThreeTrapVengeance1 => {
                "as Trap, and deals 1 damage to the attacker that killed it"
            }
            PowerId::ThreeTrapVengeance2 => {
                "as Trap, and deals 2 damage to the attacker that killed it"
            }
            PowerId::ThreeNone => "no power (ablation): dies like any other card",
            PowerId::FourForesight => {
                "one-shot: privately look at any one face-down card on the board"
            }
            PowerId::FiveFlip => {
                "one-shot: flip all your face-down cards in this lane (skips frozen)"
            }
            PowerId::SixFreeze => {
                "one-shot: freeze enemy cards in this lane for one of their turns (not 9s)"
            }
            PowerId::SevenHealAll => "one-shot: heal all your damaged cards, in every lane",
            PowerId::EightRetaliate => {
                "constant: any card that attacks this 8 takes damage (a 9 does not)"
            }
            PowerId::EightRetaliateOnSurvival => {
                "constant: as Retaliate, but only if the 8 survives the attack"
            }
            PowerId::EightNone => "no power (ablation): does not hit back",
            PowerId::NineNimble => "constant: cannot be frozen; no retaliate damage; deals 2 to Jacks",
            PowerId::TenTwinstrike => "constant: attacks split 1 damage across two enemy cards",
            PowerId::JackTaunt => {
                "constant: must be killed before anything else in the lane; extra HP face-up"
            }
            PowerId::QueenMove => "one-shot: move one allied card from another lane into this lane",
            PowerId::KingEmpower => {
                "one-shot: all your other face-up cards in this lane refire their powers"
            }
        }
    }

    // ------------------------------------------------------------------ classification --

    /// True when this power does something at the moment the card is turned face-up.
    ///
    /// The complement is not "constant": [`PowerId::ThreeTrap`] is neither, because it is
    /// conditional and fires only from `damage_card`.
    pub const fn fires_on_flip(self) -> bool {
        matches!(
            self,
            PowerId::AceAction
                | PowerId::TwoView
                | PowerId::FourForesight
                | PowerId::FiveFlip
                | PowerId::SixFreeze
                | PowerId::SevenHealAll
                | PowerId::QueenMove
                | PowerId::KingEmpower
        )
    }

    /// True when this power is read live during combat and targeting rather than firing on
    /// the flip (`game_rules.md` §6). These are exactly the powers a King cannot refire.
    pub const fn is_constant(self) -> bool {
        matches!(
            self,
            PowerId::EightRetaliate
                | PowerId::EightRetaliateOnSurvival
                | PowerId::NineNimble
                | PowerId::TenTwinstrike
                | PowerId::JackTaunt
        )
    }

    /// True when a King's Empower can meaningfully refire this power.
    ///
    /// `game_rules.md` §6: "Ranks a King can meaningfully reactivate: A, 2, 4, 5, 6, 7, Q.
    /// Ranks a King cannot reactivate: 8, 9, 10, J (constant), K (excluded by rule),
    /// 3 (conditional, and only relevant face-down)." Under the mod system this falls out
    /// of `fires_on_flip` rather than being a second list to keep in sync — the King is the
    /// one hand-written exclusion, because it fires on flip and is still excluded.
    pub const fn is_king_reactivatable(self) -> bool {
        self.fires_on_flip() && !matches!(self, PowerId::KingEmpower)
    }

    /// When this power hits back at whatever attacked it.
    pub const fn retaliate_mode(self) -> RetaliateMode {
        match self {
            PowerId::EightRetaliate => RetaliateMode::Always,
            PowerId::EightRetaliateOnSurvival => RetaliateMode::OnSurvival,
            _ => RetaliateMode::Never,
        }
    }

    /// Nimble (`game_rules.md` §6): immune to freeze, takes no retaliate damage, dodges a
    /// twinstrike's spread, and deals double to a taunter.
    pub const fn is_nimble(self) -> bool {
        matches!(self, PowerId::NineNimble)
    }

    /// Taunt (§6): must be killed before anything else in the lane, and carries extra HP.
    pub const fn taunts(self) -> bool {
        matches!(self, PowerId::JackTaunt)
    }

    /// Twinstrike (§6): attacks split their damage across two enemy cards.
    pub const fn twinstrikes(self) -> bool {
        matches!(self, PowerId::TenTwinstrike)
    }

    /// True when this power reacts to its own card's death. Used by `apply.rs` to skip the
    /// hook entirely for the common case, and by the invariant suite to know which rulesets
    /// can produce a cascade at all.
    pub const fn has_death_trigger(self) -> bool {
        matches!(
            self,
            PowerId::ThreeTrap | PowerId::ThreeTrapVengeance1 | PowerId::ThreeTrapVengeance2
        )
    }
}

impl std::fmt::Display for PowerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.token())
    }
}

// ============================================================================ dispatch ==

/// Fire a one-shot power. Called on a flip, and again for each King reactivation.
///
/// `game_rules.md` §8: "A one-shot power is **mandatory** on flip", and "A power with **no
/// legal target simply fizzles**". Both are visible in the rank modules as: push a
/// sub-decision when there is something to choose, and do nothing at all when there is not.
pub(crate) fn on_flip(state: &mut GameState, power: PowerId, ctx: PowerCtx) {
    match power {
        PowerId::AceAction => ace::action(state, ctx),
        PowerId::TwoView => two::view(state, ctx),
        PowerId::FourForesight => four::foresight(state, ctx),
        PowerId::FiveFlip => five::flip_lane(state, ctx),
        PowerId::SixFreeze => six::freeze(state, ctx),
        PowerId::SevenHealAll => seven::heal_all(state, ctx),
        PowerId::QueenMove => queen::mv(state, ctx),
        PowerId::KingEmpower => king::empower(state, ctx),

        // Conditional and constant powers do nothing at the moment of the flip.
        PowerId::ThreeTrap
        | PowerId::ThreeTrapVengeance1
        | PowerId::ThreeTrapVengeance2
        | PowerId::ThreeNone
        | PowerId::EightRetaliate
        | PowerId::EightRetaliateOnSurvival
        | PowerId::EightNone
        | PowerId::NineNimble
        | PowerId::TenTwinstrike
        | PowerId::JackTaunt => {}
    }
}

/// A card's damage has reached its hit points. Decide whether it dies.
///
/// The hook may enqueue further damage (a vengeance variant does), which lands after every
/// hit already in flight — see [`crate::damage`] for why the ordering is a queue and not a
/// call stack.
pub(crate) fn on_lethal_damage(
    state: &mut GameState,
    power: PowerId,
    ctx: PowerCtx,
    source: DamageSource,
) -> LethalOutcome {
    match power {
        PowerId::ThreeTrap => three::trap(state, ctx, source, 0),
        PowerId::ThreeTrapVengeance1 => three::trap(state, ctx, source, 1),
        PowerId::ThreeTrapVengeance2 => three::trap(state, ctx, source, 2),

        PowerId::ThreeNone
        | PowerId::AceAction
        | PowerId::TwoView
        | PowerId::FourForesight
        | PowerId::FiveFlip
        | PowerId::SixFreeze
        | PowerId::SevenHealAll
        | PowerId::EightRetaliate
        | PowerId::EightRetaliateOnSurvival
        | PowerId::EightNone
        | PowerId::NineNimble
        | PowerId::TenTwinstrike
        | PowerId::JackTaunt
        | PowerId::QueenMove
        | PowerId::KingEmpower => LethalOutcome::Die,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_power_belongs_to_its_rank_and_has_a_unique_token() {
        for rank in Rank::ALL {
            let variants = PowerId::variants_for(rank);
            assert!(
                !variants.is_empty(),
                "rank {rank} has no implemented power at all"
            );
            let mut tokens: Vec<&str> = variants.iter().map(|p| p.token()).collect();
            tokens.sort_unstable();
            let before = tokens.len();
            tokens.dedup();
            assert_eq!(before, tokens.len(), "duplicate token within rank {rank}");
            for p in variants {
                assert_eq!(p.rank(), rank);
                assert_eq!(PowerId::parse(rank, p.token()), Some(p));
            }
        }
    }

    /// The canonical table must reproduce `rank.rs`'s original hand-written lists, which is
    /// what makes the `PowerId` refactor a no-op for the rules as written.
    #[test]
    fn canonical_powers_reproduce_the_rulebook_classification() {
        let constant: Vec<Rank> = Rank::ALL
            .into_iter()
            .filter(|r| PowerId::canonical_for(*r).is_constant())
            .collect();
        assert_eq!(
            constant,
            vec![Rank::EIGHT, Rank::NINE, Rank::TEN, Rank::JACK]
        );

        let reactivatable: Vec<Rank> = Rank::ALL
            .into_iter()
            .filter(|r| PowerId::canonical_for(*r).is_king_reactivatable())
            .collect();
        assert_eq!(
            reactivatable,
            vec![
                Rank::ACE,
                Rank::TWO,
                Rank::FOUR,
                Rank::FIVE,
                Rank::SIX,
                Rank::SEVEN,
                Rank::QUEEN
            ]
        );
    }

    #[test]
    fn canonical_for_round_trips_through_rank() {
        for rank in Rank::ALL {
            assert_eq!(PowerId::canonical_for(rank).rank(), rank);
        }
    }
}
