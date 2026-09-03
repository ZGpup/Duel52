//! The two players.

use std::fmt;

/// One of the two players. `P0` always moves first, and takes only two actions on the
/// opening turn (`game_rules.md` §2).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Player {
    P0 = 0,
    P1 = 1,
}

impl Player {
    pub const BOTH: [Player; 2] = [Player::P0, Player::P1];

    /// Index into the engine's various two-element arrays (hands, piles, discards, lane
    /// sides). Always `0` or `1`.
    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// The opponent.
    #[inline]
    pub const fn other(self) -> Player {
        match self {
            Player::P0 => Player::P1,
            Player::P1 => Player::P0,
        }
    }

    /// Bit for this player in a two-player bitmask (see `Card::known_to`).
    #[inline]
    pub const fn bit(self) -> u8 {
        1 << (self as u8)
    }

    #[inline]
    pub fn from_index(i: usize) -> Player {
        match i {
            0 => Player::P0,
            1 => Player::P1,
            _ => panic!("player index {i} out of range 0..2"),
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Player::P0 => f.write_str("P0"),
            Player::P1 => f.write_str("P1"),
        }
    }
}
