//! How a game ends.

use crate::player::Player;
use std::fmt;

/// Why a game was declared a draw.
///
/// Tracked separately from the draw itself because `PLAN.md` Phase 1's deliverable asks
/// specifically "how often games reach the stalemate cutoff".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DrawReason {
    /// **[ENGINE]** `config.stalemate_quiet_plies` consecutive plies with no damage and no
    /// kill (`game_rules.md` §7). The published rules define no draw; this is required for
    /// training, because the reachable stall — both hands empty post-unlock, and neither
    /// player wanting to attack first — never ends on its own.
    Stalemate,
    /// **[ENGINE]** Both players reached the lane-win threshold on the same terminal check.
    ///
    /// `game_rules.md` §7 spells out the one way this happens: your last card in a lane
    /// attacks the opponent's last card in that lane, which is an 8, and retaliate kills
    /// yours as your damage kills theirs. Both sides of the lane empty, so both players win
    /// it. "A symmetric outcome gets a symmetric result, with no arbitrary tiebreak."
    MutualLaneWin,
    /// **[ENGINE]** `config.max_plies` safety cap. Should be unreachable; if this ever
    /// fires it is a bug report, not a game result.
    PlyLimit,
}

impl fmt::Display for DrawReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DrawReason::Stalemate => "stalemate (quiet-ply limit reached)",
            DrawReason::MutualLaneWin => "both players won the game on the same check",
            DrawReason::PlyLimit => "engine ply-limit safety cap (this is a bug)",
        };
        f.write_str(s)
    }
}

/// The result of a game.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    Ongoing,
    Win(Player),
    Draw(DrawReason),
}

impl Outcome {
    #[inline]
    pub fn is_over(self) -> bool {
        !matches!(self, Outcome::Ongoing)
    }

    /// Zero-sum value from `player`'s point of view: `1.0` win, `0.5` draw, `0.0` loss.
    /// `Ongoing` returns `0.5` so that a value target is always well defined; callers that
    /// care should check `is_over()` first.
    pub fn value_for(self, player: Player) -> f32 {
        match self {
            Outcome::Ongoing => 0.5,
            Outcome::Draw(_) => 0.5,
            Outcome::Win(w) => {
                if w == player {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Ongoing => f.write_str("in progress"),
            Outcome::Win(p) => write!(f, "{p} wins"),
            Outcome::Draw(r) => write!(f, "draw — {r}"),
        }
    }
}
