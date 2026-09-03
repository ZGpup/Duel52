//! Card ranks.
//!
//! **Suits do not exist in this engine.** `game_rules.md` §1: "Suits are mechanically
//! irrelevant. Only rank matters." The one place colour appears in the paper game is the
//! split-deck variant (§9a), where colour means *which player owns the deck* — and that is
//! tracked per player, not per card. So a card is fully described by its rank.

use std::fmt;

/// A card rank, stored as an index `0..=12` where `0` is the Ace and `12` is the King.
///
/// Using an index rather than a 13-way enum keeps the neural-network encoding (a 13-wide
/// one-hot, `DESIGN.md` §5) a direct cast, and makes "count of each rank" tables plain
/// arrays.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Rank(u8);

// Named constants for every rank. Rules code should always use these rather than bare
// numbers, so that a line like `if rank == Rank::NINE` reads the way the rulebook does.
impl Rank {
    pub const ACE: Rank = Rank(0);
    pub const TWO: Rank = Rank(1);
    pub const THREE: Rank = Rank(2);
    pub const FOUR: Rank = Rank(3);
    pub const FIVE: Rank = Rank(4);
    pub const SIX: Rank = Rank(5);
    pub const SEVEN: Rank = Rank(6);
    pub const EIGHT: Rank = Rank(7);
    pub const NINE: Rank = Rank(8);
    pub const TEN: Rank = Rank(9);
    pub const JACK: Rank = Rank(10);
    pub const QUEEN: Rank = Rank(11);
    pub const KING: Rank = Rank(12);

    /// How many distinct ranks exist. Ace through King.
    pub const COUNT: usize = 13;

    /// Every rank in ascending order, Ace first.
    pub const ALL: [Rank; Rank::COUNT] = [
        Rank::ACE,
        Rank::TWO,
        Rank::THREE,
        Rank::FOUR,
        Rank::FIVE,
        Rank::SIX,
        Rank::SEVEN,
        Rank::EIGHT,
        Rank::NINE,
        Rank::TEN,
        Rank::JACK,
        Rank::QUEEN,
        Rank::KING,
    ];

    /// Build a rank from its index. Panics on an out-of-range index, because every caller
    /// inside the engine derives the index from `Rank::ALL` or from a deck we built
    /// ourselves — an out-of-range value is a bug, not user input.
    #[inline]
    pub fn from_index(i: usize) -> Rank {
        assert!(i < Rank::COUNT, "rank index {i} out of range 0..13");
        Rank(i as u8)
    }

    /// Build a rank from an index, returning `None` instead of panicking. Use this on
    /// anything that came from outside the engine (the CLI, the Python bindings).
    #[inline]
    pub fn try_from_index(i: usize) -> Option<Rank> {
        if i < Rank::COUNT {
            Some(Rank(i as u8))
        } else {
            None
        }
    }

    /// The rank's index, `0..=12`. This is the one-hot position used by the encoders.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Maximum hit points **while face-up**.
    ///
    /// `game_rules.md` §5: "Every card has 2 hit points, except the Jack, which has 3."
    ///
    /// # Do not call this on a card in play — use [`crate::card::Card::max_hp`]
    ///
    /// A **face-down card is a blank 2-HP card regardless of rank** (§5), so a face-down
    /// Jack dies to two hits like anything else. The Jack's third hit point arrives when it
    /// is flipped. This method knows nothing about face-up state, which is why it is named
    /// for the case it describes; combat, death, healing and rendering all go through
    /// `Card::max_hp` instead.
    #[inline]
    pub const fn face_up_max_hp(self) -> u8 {
        if self.0 == Rank::JACK.0 {
            3
        } else {
            2
        }
    }

    /// The short label used everywhere a human reads a rank: `A 2 3 ... 10 J Q K`.
    pub const fn label(self) -> &'static str {
        match self.0 {
            0 => "A",
            1 => "2",
            2 => "3",
            3 => "4",
            4 => "5",
            5 => "6",
            6 => "7",
            7 => "8",
            8 => "9",
            9 => "10",
            10 => "J",
            11 => "Q",
            12 => "K",
            _ => "?",
        }
    }

    /// The power's name, as printed on the card. Used by the CLI so the owner can check the
    /// engine is firing the power they expect.
    pub const fn power_name(self) -> &'static str {
        match self.0 {
            0 => "Action",
            1 => "View",
            2 => "Trap",
            3 => "Foresight",
            4 => "Flip",
            5 => "Freeze",
            6 => "Heal All",
            7 => "Retaliate",
            8 => "Nimble",
            9 => "Twinstrike",
            10 => "Taunt",
            11 => "Move",
            12 => "Empower",
            _ => "?",
        }
    }

    /// A one-line summary of the power, for the CLI's `help` screen.
    pub const fn power_text(self) -> &'static str {
        match self.0 {
            0 => "one-shot: +1 action this turn; this Ace may attack twice this turn",
            1 => "one-shot: draw 1 from your pile, then put a card from hand on the bottom",
            2 => "if killed while FACE-DOWN, returns face-up at full HP in the same lane",
            3 => "one-shot: privately look at any one face-down card on the board",
            4 => "one-shot: flip all your face-down cards in this lane (skips frozen)",
            5 => "one-shot: freeze enemy cards in this lane for one of their turns (not 9s)",
            6 => "one-shot: heal all your damaged cards by 2 HP, in every lane",
            7 => "constant: any card that attacks this 8 takes 1 damage (a 9 does not)",
            8 => "constant: cannot be frozen; no retaliate damage; deals 2 to Jacks",
            9 => "constant: attacks split 1 damage across two enemy cards",
            10 => "constant: must be killed before anything else in the lane; 3 HP face-up",
            11 => "one-shot: move one allied card from another lane into this lane",
            12 => "one-shot: all your other face-up cards in this lane refire their powers",
            _ => "?",
        }
    }

    /// True when this rank's power is *constant* — continuously live while face-up, rather
    /// than firing on the flip. `game_rules.md` §6.
    ///
    /// Constant powers are exactly the ones a King cannot reactivate: 8, 9, 10, J.
    #[inline]
    pub const fn is_constant_power(self) -> bool {
        matches!(self.0, 7 | 8 | 9 | 10)
    }

    /// True when a King's Empower can meaningfully refire this rank.
    ///
    /// `game_rules.md` §6: "Ranks a King can meaningfully reactivate: A, 2, 4, 5, 6, 7, Q.
    /// Ranks a King cannot reactivate: 8, 9, 10, J (constant), K (excluded by rule),
    /// 3 (conditional, and only relevant face-down)."
    #[inline]
    pub const fn is_king_reactivatable(self) -> bool {
        matches!(self.0, 0 | 1 | 3 | 4 | 5 | 6 | 11)
    }

    /// Parse a rank from text the way a player would type it: `a`, `A`, `1`, `10`, `t`,
    /// `j`, `q`, `k`. Case-insensitive. Returns `None` if it is not a rank.
    pub fn parse(s: &str) -> Option<Rank> {
        match s.trim().to_ascii_lowercase().as_str() {
            "a" | "1" | "ace" => Some(Rank::ACE),
            "2" => Some(Rank::TWO),
            "3" => Some(Rank::THREE),
            "4" => Some(Rank::FOUR),
            "5" => Some(Rank::FIVE),
            "6" => Some(Rank::SIX),
            "7" => Some(Rank::SEVEN),
            "8" => Some(Rank::EIGHT),
            "9" => Some(Rank::NINE),
            "10" | "t" => Some(Rank::TEN),
            "j" | "11" | "jack" => Some(Rank::JACK),
            "q" | "12" | "queen" => Some(Rank::QUEEN),
            "k" | "13" | "king" => Some(Rank::KING),
            _ => None,
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// A count of cards per rank. Used for hands, discards, and belief features — anywhere a
/// multiset of ranks is the natural representation (`DESIGN.md` §5).
pub type RankCounts = [u8; Rank::COUNT];

/// Tally a slice of ranks into per-rank counts.
pub fn rank_counts(ranks: &[Rank]) -> RankCounts {
    let mut counts: RankCounts = [0; Rank::COUNT];
    for r in ranks {
        counts[r.index()] += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `game_rules.md` §5: "Every card has 2 hit points, except the Jack, which has 3."
    /// This is the *face-up* table; the face-down case is uniform and lives on `Card`.
    #[test]
    fn rule_5_face_up_jack_has_three_hp_everything_else_has_two() {
        for r in Rank::ALL {
            let expected = if r == Rank::JACK { 3 } else { 2 };
            assert_eq!(r.face_up_max_hp(), expected, "wrong face-up max HP for {r}");
        }
    }

    #[test]
    fn constant_powers_are_exactly_eight_nine_ten_jack() {
        let constant: Vec<Rank> = Rank::ALL
            .into_iter()
            .filter(|r| r.is_constant_power())
            .collect();
        assert_eq!(
            constant,
            vec![Rank::EIGHT, Rank::NINE, Rank::TEN, Rank::JACK]
        );
    }

    #[test]
    fn king_reactivatable_set_matches_the_rulebook() {
        let reactivatable: Vec<Rank> = Rank::ALL
            .into_iter()
            .filter(|r| r.is_king_reactivatable())
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
    fn labels_round_trip_through_parse() {
        for r in Rank::ALL {
            assert_eq!(Rank::parse(r.label()), Some(r));
        }
    }
}
