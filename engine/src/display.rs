//! Rendering a position for a **specific observer**.
//!
//! This is the only place a human ever sees a board, so it is also where information
//! hiding has to be right. `game_rules.md` §5 defines what is public:
//!
//! - public: which lanes hold how many cards, every face-up rank, `is_base` status, **all
//!   damage including on face-down cards**, both hand *sizes*, the whole discard pile;
//! - private: your hand contents, your own played face-down cards, anything a 4 revealed to
//!   you, and the identity/position of a card you bottomed with a 2;
//! - hidden from **everyone**: base cards, and the cards removed unseen at setup.
//!
//! One detail is easy to get wrong and is handled explicitly below: **`{?  ²♥}` must be
//! used for the owner's own base cards.** Base cards are hidden from their owner too (§3),
//! which is exactly the fact a careless renderer misses.
//!
//! Hit points, by contrast, need no filtering at all: §5 makes every face-down card a blank
//! 2-HP card whatever its rank, so a face-down card's hit points are common knowledge and
//! printing them reveals nothing. (Were it otherwise — were a face-down Jack really 3 HP —
//! then rendering `³♥` would announce the Jack, and so would simply watching it survive two
//! hits, since damage is public.)
//!
//! # The board is a grid, and every cell is the same width
//!
//! Lanes run left to right as columns, the opponent at the top and the observer at the
//! bottom, with each side's base card at its far end — the way the cards sit on a table.
//! Every card is a six-column token ([`card_token`]) whatever its rank, damage or state, so
//! a column never shifts sideways as the position changes.
//!
//! # Lanes and cards are numbered from 1 here, and only here
//!
//! The engine indexes lanes and slots from 0, and so do [`Action`], the Python action dicts,
//! and every test. Humans count from 1, so this module — the display layer, which nothing
//! else reads back — converts on the way out via [`lane_label`] and [`card_number`].
//!
//! [`card_number`] is more than an off-by-one: **slots are not display order**. Slots are
//! the engine's storage order, base card first; the board draws the observer's own base card
//! *last*, at the bottom of their column. So a card's number is its position in the column
//! the observer is looking at, which is what makes "the second card in lane 2" mean the same
//! thing on the board and at the prompt. [`column_slots`] is the single definition of that
//! order, and both the renderer and the CLI's menus go through it.

use crate::action::{Action, Side};
use crate::card::Card;
use crate::player::Player;
use crate::rank::Rank;
use crate::state::{GameState, Pending};

/// Who is looking.
///
/// `Some(p)` renders the board as player `p` is entitled to see it. `None` is omniscient
/// and is for debugging only — the CLI hides it behind an explicit `--reveal` flag.
pub type Observer = Option<Player>;

/// The lane number a human reads, given the engine's 0-based lane index.
#[inline]
pub fn lane_label(lane: impl Into<usize>) -> usize {
    lane.into() + 1
}

/// The slots of one side of one lane, in the order the board draws them, top to bottom.
///
/// The observer's own base card is drawn at the bottom of their column and the opponent's at
/// the top of theirs, so both sides read outward from the front line. Everything else keeps
/// slot order. Menus number cards by position in this list, so the number a player types
/// always counts down the column they are looking at.
pub fn column_slots(
    state: &GameState,
    lane: usize,
    owner: Player,
    observer: Observer,
) -> Vec<usize> {
    let side = state.lanes[lane].side(owner);
    let (bases, played): (Vec<usize>, Vec<usize>) =
        (0..side.len()).partition(|&slot| side[slot].is_base);
    if owner == observer.unwrap_or(state.to_move) {
        played.into_iter().chain(bases).collect()
    } else {
        bases.into_iter().chain(played).collect()
    }
}

/// The number a human reads for a card, given the engine's slot index: its position in the
/// column the board draws, counting from 1.
pub fn card_number(
    state: &GameState,
    lane: usize,
    owner: Player,
    slot: usize,
    observer: Observer,
) -> usize {
    column_slots(state, lane, owner, observer)
        .iter()
        .position(|&s| s == slot)
        .map(|i| i + 1)
        // A slot that is no longer on the board: only reachable from a description of an
        // action whose target has already died, which the CLI renders as `<gone>` anyway.
        .unwrap_or(slot + 1)
}

/// Does `observer` know this card's rank?
pub(crate) fn knows(card: &Card, observer: Observer) -> bool {
    match observer {
        None => true,
        Some(p) => card.rank_known_to(p),
    }
}

/// Is `observer` entitled to the acting player's private knowledge — their hand, and the
/// ranks of the face-down cards they have played?
///
/// Only the omniscient debug view and the acting player themselves are. This is what stops
/// the CLI's move log from announcing "P1 played a 9 face-down" to the human sitting
/// opposite: the rank travels inside [`Action::Play`], so it has to be filtered here rather
/// than relying on the card's knowledge mask.
pub(crate) fn entitled_to_actors_hand(state: &GameState, observer: Observer) -> bool {
    match observer {
        None => true,
        Some(p) => p == state.acting_player(),
    }
}

/// Hit points as a superscript. Nothing on the board ever exceeds 3 — a face-up Jack (§5).
const HP_GLYPH: [&str; 4] = ["⁰", "¹", "²", "³"];

/// The width of [`card_token`], in columns. Every token is exactly this wide.
pub const TOKEN_WIDTH: usize = 6;

/// One card as a **fixed-width six-column token**: `[3 ²♥]`, `(? ²♥)`, `{K ¹♥}`, `[10³♥]`.
///
/// The width never varies, so a lane column never shifts sideways as cards take damage or a
/// 10 arrives. The rank field is two columns, left-aligned, which is exactly what lets the
/// 10 eat the space that separates rank from hit points for every other rank.
///
/// The brackets carry the two facts a bare rank cannot:
///
/// - `{…}` a **base card**: untouchable until every draw pile is empty (§3), and hidden from
///   its owner as well as from the opponent — which is why its rank is normally `?`.
/// - `[…]` **face-up**: power live, can attack.
/// - `(…)` **face-down**: a blank 2-HP card with no power (§4, §6). Shows a rank only to an
///   observer entitled to know it.
///
/// The number is the hit points **remaining**, which is public for every card: §5 makes a
/// face-down card a blank 2 HP whatever its rank, so this leaks nothing.
pub fn card_token(card: &Card, observer: Observer) -> String {
    let label = if knows(card, observer) {
        card.rank.label()
    } else {
        "?"
    };
    let hp = HP_GLYPH[card.hp_remaining().min(3) as usize];
    let (open, close) = if card.is_base {
        ('{', '}')
    } else if card.face_up {
        ('[', ']')
    } else {
        ('(', ')')
    };
    format!("{open}{label:<2}{hp}♥{close}")
}

/// The two columns that follow a card's token: which pair it belongs to, then its condition.
///
/// Two columns of symbols rather than words, because these hang off every card on the board
/// and the board has to stay a board. What they mean lives in the CLI's `help`.
fn card_status(state: &GameState, lane: usize, owner: Player, slot: usize) -> String {
    let Some(card) = state.at(lane, owner, slot) else {
        return "  ".to_string();
    };
    let pair = pair_letter(state, lane, owner, slot).unwrap_or(' ');
    let condition = if card.is_frozen(state.ply) {
        // §8: frozen blocks attacking and being flipped, by anyone.
        '*'
    } else if card.face_up && card.owner == state.to_move {
        // Only meaningful on your own turn, which is the only time attack budgets move.
        if card.attacks_used >= card.attack_allowance {
            '·'
        } else if card.attack_allowance - card.attacks_used > 1 {
            '+'
        } else {
            ' '
        }
    } else {
        ' '
    };
    format!("{pair}{condition}")
}

/// Which declared pair a card belongs to, as a letter unique within its side of its lane.
///
/// `PairId`s are global and unbounded, so they are useless as a one-column marker. A pair is
/// confined to one side of one lane (§5), so numbering them within that side is enough to
/// tell two pairs apart wherever a player can actually see them side by side.
fn pair_letter(state: &GameState, lane: usize, owner: Player, slot: usize) -> Option<char> {
    let wanted = state.at(lane, owner, slot)?.pair_id?;
    let mut seen: Vec<crate::card::PairId> = Vec::new();
    for card in state.lanes[lane].side(owner) {
        if let Some(id) = card.pair_id {
            if !seen.contains(&id) {
                seen.push(id);
            }
        }
    }
    let index = seen.iter().position(|&id| id == wanted)?;
    Some((b'a' + index as u8) as char)
}

/// A hand: contents if the observer owns it, otherwise just the count.
fn hand_text(state: &GameState, owner: Player, observer: Observer) -> String {
    let hand = state.hand(owner);
    let entitled = observer.is_none() || observer == Some(owner);
    if entitled {
        if hand.is_empty() {
            "—".to_string()
        } else {
            hand.iter()
                .map(|r| r.label())
                .collect::<Vec<_>>()
                .join(" ")
        }
    } else {
        // Hand *size* is public (§5); the contents are not.
        format!("{}", hand.len())
    }
}

/// The discard pile — public to both players at any time (§5), so no filtering.
fn discard_text(state: &GameState, owner: Player) -> String {
    let d = &state.discards[owner.idx()];
    if d.is_empty() {
        "—".to_string()
    } else {
        let mut sorted = d.clone();
        sorted.sort_unstable();
        sorted
            .iter()
            .map(|r| r.label())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// What this observer privately knows about the bottom of a pile, from the 2's scry (§10a).
fn bottom_knowledge_text(state: &GameState, observer: Observer) -> Option<String> {
    let p = observer?;
    let mut parts = Vec::new();
    for owner in Player::BOTH {
        let idx = state.pile_index(owner);
        if state.shared_pile() && owner == Player::P1 {
            continue; // one shared pile; do not report it twice
        }
        let known = state.piles[idx].known_from_bottom(p);
        // Report the run of known cards at the bottom; a `None` ends it, because anything
        // deeper was not put there by this observer.
        let run: Vec<Rank> = known.into_iter().map_while(|k| k).collect();
        if run.is_empty() {
            continue;
        }
        let label = if state.shared_pile() {
            "shared pile".to_string()
        } else {
            format!("{owner}'s pile")
        };
        parts.push(format!(
            "{label} bottom-up: {}",
            run.iter().map(|r| r.label()).collect::<Vec<_>>().join(" ")
        ));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" | "))
    }
}

/// One lane column, including the space either side of the eight-column cell.
const CELL: usize = TOKEN_WIDTH + 2 + 2;

/// Render the whole position from `observer`'s point of view.
///
/// Lanes are columns, left to right; the opponent is at the top and the observer at the
/// bottom; each side's base card sits at the far end of its column. The double rule across
/// the middle is the front line — the only place cards can reach each other.
pub fn render(state: &GameState, observer: Observer) -> String {
    let me = observer.unwrap_or(state.to_move);
    let them = me.other();
    let lanes = state.lane_count();

    // --- The grid ---------------------------------------------------------------------
    // Every cell is exactly `CELL` columns, built by its caller, so a row is a plain join.
    // That is what keeps the columns still: nothing here depends on a card's rank, damage
    // or state.
    let cells = |cells: Vec<String>| -> String { format!("  {}\n", cells.join("│")) };
    let bar = |fill: &str, cross: &str| -> String {
        let segment = fill.repeat(CELL);
        format!(
            "  {}\n",
            vec![segment; lanes].join(cross)
        )
    };
    // Two leading spaces put the token's left edge under the `l` of `lane N`.
    let empty = " ".repeat(CELL);

    // Base cards are drawn on their own row at the far end of each column; the rows in
    // between hold only what has been played. This is the same partition [`column_slots`]
    // is built on, so what the board draws and what the menus number cannot disagree.
    let split = |lane: usize, owner: Player| -> (Vec<usize>, Vec<usize>) {
        let side = state.lanes[lane].side(owner);
        (0..side.len()).partition(|&slot| side[slot].is_base)
    };
    let cell = |lane: usize, owner: Player, slot: Option<usize>| -> String {
        match slot {
            Some(slot) => format!(
                "  {}{}",
                card_token(&state.lanes[lane].side(owner)[slot], observer),
                card_status(state, lane, owner, slot)
            ),
            None => empty.clone(),
        }
    };
    // `1` keeps an empty side from collapsing the grid, which would make the board jump
    // about from turn to turn.
    let played_rows = |owner: Player| -> usize {
        (0..lanes)
            .map(|lane| split(lane, owner).1.len())
            .max()
            .unwrap_or(0)
            .max(1)
    };
    let base_row = |owner: Player| -> String {
        cells(
            (0..lanes)
                .map(|lane| cell(lane, owner, split(lane, owner).0.first().copied()))
                .collect(),
        )
    };
    let played_row = |owner: Player, row: usize| -> String {
        cells(
            (0..lanes)
                .map(|lane| cell(lane, owner, split(lane, owner).1.get(row).copied()))
                .collect(),
        )
    };

    let mut out = String::new();

    // --- Above the board --------------------------------------------------------------
    if state.shared_pile() {
        out.push_str(&format!(" Deck: {}\n", state.piles[0].len()));
    } else {
        out.push_str(&format!(
            " Deck: you {} · opponent {}\n",
            state.pile(me).len(),
            state.pile(them).len()
        ));
    }
    out.push_str(&format!(" {}\n", "═".repeat(lanes * (CELL + 1))));
    if observer.is_none() {
        out.push_str(" *** REVEAL MODE: showing hidden information ***\n");
    }
    out.push_str(&format!(
        " {them}   hand {}   discard {}\n\n",
        hand_text(state, them, observer),
        discard_text(state, them),
    ));

    // --- The board --------------------------------------------------------------------
    out.push_str(&cells(
        (0..lanes)
            .map(|lane| format!("  lane {:<width$}", lane_label(lane), width = CELL - 7))
            .collect(),
    ));
    out.push_str(&base_row(them));
    out.push_str(&bar("─", "┼"));
    for row in 0..played_rows(them) {
        out.push_str(&played_row(them, row));
    }
    out.push_str(&bar("═", "╪"));
    for row in 0..played_rows(me) {
        out.push_str(&played_row(me, row));
    }
    out.push_str(&bar("─", "┼"));
    out.push_str(&base_row(me));

    // --- Below the board --------------------------------------------------------------
    out.push_str(&format!(
        "\n {me}   hand {}   discard {}\n",
        hand_text(state, me, observer),
        discard_text(state, me),
    ));
    if let Some(bottom) = bottom_knowledge_text(state, observer) {
        out.push_str(&format!(" you know: {bottom}\n"));
    }
    if state.removed_revealed {
        // §9b only: the removed multiset is public in the mirrored-removal variant.
        let mut ranks: Vec<Rank> = state.removed[0].clone();
        ranks.sort_unstable();
        out.push_str(&format!(
            " removed from each deck (§9b): {}\n",
            ranks.iter().map(|r| r.label()).collect::<Vec<_>>().join(" ")
        ));
    } else if observer.is_none() {
        let mut ranks: Vec<Rank> = state.all_removed().collect();
        ranks.sort_unstable();
        out.push_str(&format!(
            " removed unseen (hidden in play): {}\n",
            ranks.iter().map(|r| r.label()).collect::<Vec<_>>().join(" ")
        ));
    }
    out.push_str(&format!(" {}\n", "═".repeat(lanes * (CELL + 1))));

    if state.outcome.is_over() {
        out.push_str(&format!(" GAME OVER: {}\n", state.outcome));
    } else {
        // Base lock is the whole shape of the game — nothing can be won until it lifts
        // (§3, §7) — so it stays on screen even though everything else here is a counter.
        out.push_str(&format!(
            " ply {} · base {} · quiet {}/{}{}\n",
            state.ply,
            if state.base_unlocked {
                "UNLOCKED"
            } else {
                "locked"
            },
            state.quiet_plies,
            state.config.stalemate_quiet_plies,
            if state.base_unlocked {
                format!(
                    " · lanes won: you {} · opp {}",
                    state.lanes_won_by(me),
                    state.lanes_won_by(them)
                )
            } else {
                String::new()
            },
        ));
    }
    out
}

/// The face-up-only facts about an attack that a human wants in front of them before
/// committing to it: the attacker's spread powers, and the defender's constant ones.
///
/// Every note comes from a **face-up** card, so this leaks nothing and needs no observer —
/// you are never told that the face-down card you are about to attack is a Jack. An
/// out-of-range slot contributes nothing, which is what lets a caller that has only one
/// half of the matchup pass `usize::MAX` for the other.
pub(crate) fn combat_notes(
    state: &GameState,
    lane: usize,
    attacker: usize,
    target: usize,
) -> Vec<String> {
    let me = state.acting_player();
    let mut notes: Vec<String> = Vec::new();
    if let Some(atk) = state.at(lane, me, attacker) {
        if atk.has_live_power(Rank::TEN) {
            notes.push("twinstrike".to_string());
        }
        if atk.has_live_power(Rank::NINE) {
            notes.push("nimble".to_string());
        }
    }
    if let Some(def) = state.at(lane, me.other(), target) {
        if def.has_live_power(Rank::EIGHT) {
            notes.push("8 retaliates for 1".to_string());
        }
        if def.has_live_power(Rank::JACK) {
            notes.push(format!("Jack, {} HP left of 3", def.hp_remaining()));
        }
        if def.has_live_power(Rank::NINE) {
            notes.push("9 blocks the twinstrike split".to_string());
        }
    }
    notes
}

/// How much explanation a description carries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Detail {
    /// Spell out what the card's power will do. For an offer, where the player is deciding
    /// and may not have the rules to hand.
    Teaching,
    /// State what happened and stop. For a move log, where the player has just read the
    /// teaching version in the menu and only needs the record.
    Brief,
}

/// Describe an action **on offer**, from `observer`'s point of view: what it does, and what
/// the power involved means.
///
/// Like [`render`], it must not leak: every rank it names comes from a card the observer is
/// entitled to know, and the combat notes are derived only from **face-up** information —
/// you do not get told that the face-down card you are about to attack is a Jack.
///
/// Two variants carry a rank *in the action itself* rather than on a card, so they need
/// explicit filtering rather than a knowledge-mask check:
///
/// - [`Action::Play`] names the card leaving the hand. It lands face-down, so an observer
///   who is not the actor may not be told which card it was.
/// - [`Action::GiveBack`] names a card leaving the hand for the bottom of a draw pile,
///   which `game_rules.md` §5 makes private to its owner. Under `two_power = discard` the
///   same card goes to the public discard pile instead, and then the rank is public.
pub fn describe_action(state: &GameState, action: Action, observer: Observer) -> String {
    describe(state, action, observer, Detail::Teaching)
}

/// Describe an action **that has been taken**, for a move log: the same filtering, without
/// the rules tuition, so a run of moves stays one line each.
pub fn describe_move(state: &GameState, action: Action, observer: Observer) -> String {
    describe(state, action, observer, Detail::Brief)
}

fn describe(state: &GameState, action: Action, observer: Observer, detail: Detail) -> String {
    let me = state.acting_player();
    let them = me.other();
    let entitled = entitled_to_actors_hand(state, observer);
    let teaching = detail == Detail::Teaching;

    let token = |lane: usize, owner: Player, slot: usize| -> String {
        match state.at(lane, owner, slot) {
            Some(card) => card_token(card, observer),
            None => "<gone>".to_string(),
        }
    };
    // A card is named by where it sits in the column the observer is looking at, not by its
    // storage slot — see the module docs on [`card_number`].
    let num = |lane: u8, owner: Player, slot: u8| -> usize {
        card_number(state, lane as usize, owner, slot as usize, observer)
    };

    // "your" and "opp" are relative to the actor, so they only read correctly when the
    // actor is the one reading. In the CLI's move log — where the observer is watching
    // somebody *else* act — they would be exactly backwards, so name the sides instead.
    let (ours, theirs) = if entitled {
        ("your".to_string(), "opp".to_string())
    } else {
        (format!("{me}"), format!("{them}"))
    };

    // The power's name and text are public knowledge about a rank, so wherever the rank is
    // shown at all this can be appended without leaking anything further.
    let power_of = |rank: Rank| {
        if teaching {
            format!("          [{}: {}]", rank.power_name(), rank.power_text())
        } else {
            String::new()
        }
    };

    match action {
        Action::Play { rank, lane } if entitled => format!(
            "PLAY  {rank} face-down into lane {}{}",
            lane_label(lane),
            power_of(rank)
        ),
        // Someone else's play: the card is face-down, so all that is public is the lane.
        Action::Play { lane, .. } => format!(
            "PLAY  a card from hand, face-down, into lane {}",
            lane_label(lane)
        ),

        Action::Flip { lane, slot } => {
            let card = state.at(lane as usize, me, slot as usize);
            let head = format!(
                "FLIP  lane {} #{} {}",
                lane_label(lane),
                num(lane, me, slot),
                token(lane as usize, me, slot as usize)
            );
            match card {
                Some(c) if knows(c, observer) => {
                    format!("{head} -> reveals {}{}", c.rank, power_of(c.rank))
                }
                // A base card, or a Queen-moved ex-base card: even you do not know it (§3).
                _ if entitled => format!("{head} -> you do not know what this is"),
                // Somebody else's face-down card. The flip makes it public a moment later;
                // it is not public yet, so the log line says only that it happened.
                _ => format!("{head} -> turns it face-up"),
            }
        }

        Action::Attack {
            lane,
            attacker,
            target,
        } => {
            let lane_i = lane as usize;
            let paired = state
                .pair_partner(lane_i, me, attacker as usize)
                .map(|p| format!(" (PAIR with #{}, 2 dmg)", num(lane, me, p as u8)))
                .unwrap_or_default();
            let notes = combat_notes(state, lane_i, attacker as usize, target as usize);
            let note = if notes.is_empty() {
                String::new()
            } else {
                format!("   <{}>", notes.join("; "))
            };
            format!(
                "ATK   lane {}: {ours} #{} {}{paired} -> {theirs} #{} {}{note}",
                lane_label(lane),
                num(lane, me, attacker),
                token(lane_i, me, attacker as usize),
                num(lane, them, target),
                token(lane_i, them, target as usize),
            )
        }

        Action::DeclarePair {
            lane,
            slot_a,
            slot_b,
        } => {
            // Both members are face-up, so their rank is public whoever is looking.
            let rank = state
                .at(lane as usize, me, slot_a as usize)
                .map(|c| c.rank.label())
                .unwrap_or("?");
            let caveat = if teaching {
                " — one action for 2 damage, but they can never attack separately again"
            } else {
                ""
            };
            format!(
                "PAIR  lane {}: #{} + #{} (two {rank}s){caveat}",
                lane_label(lane),
                num(lane, me, slot_a),
                num(lane, me, slot_b),
            )
        }

        Action::Peek { side, lane, slot } => {
            let owner = match side {
                Side::Mine => me,
                Side::Theirs => them,
            };
            let whose = match (entitled, side) {
                (true, Side::Mine) => "your".to_string(),
                (true, Side::Theirs) => "opponent's".to_string(),
                (false, _) => format!("{owner}'s"),
            };
            format!(
                "PEEK  {whose} lane {} #{} {}",
                lane_label(lane),
                num(lane, owner, slot),
                token(lane as usize, owner, slot as usize)
            )
        }

        Action::ResolveNext { lane, slot } => format!(
            "NEXT  lane {} #{} {}",
            lane_label(lane),
            num(lane, me, slot),
            token(lane as usize, me, slot as usize)
        ),

        Action::MoveHere { lane, slot } => format!(
            "MOVE  lane {} #{} {} into the Queen's lane{}",
            lane_label(lane),
            num(lane, me, slot),
            token(lane as usize, me, slot as usize),
            if teaching {
                " (keeps damage, keeps freeze, stops being a base card)"
            } else {
                ""
            }
        ),

        Action::GiveBack { rank } => match state.config.two_power {
            // §5: the identity of a card you bottom is private, so only its owner is told.
            crate::config::TwoPower::Bottom if entitled => {
                format!("BACK  put {rank} on the bottom of your draw pile")
            }
            crate::config::TwoPower::Bottom => {
                "BACK  put a card from hand on the bottom of their draw pile".to_string()
            }
            // A discard goes to the public discard pile, so this one leaks nothing.
            crate::config::TwoPower::Discard => format!("BACK  discard {rank}"),
        },

        Action::SplitTarget { slot } => {
            // The twinstrike's lane is whatever the pending node says.
            let lane = match state.pending.last() {
                Some(Pending::SplitTarget { lane, .. }) => *lane as usize,
                _ => 0,
            };
            format!(
                "2ND   twinstrike's second target: {theirs} lane {} #{} {}",
                lane_label(lane),
                num(lane as u8, them, slot),
                token(lane, them, slot as usize)
            )
        }
    }
}

/// The card-power reference, for the CLI's `powers` command.
pub fn power_reference() -> String {
    let mut out = String::from("Card powers (game_rules.md §6). Powers are inert face-down.\n");
    out.push_str(&format!(
        "{:>4}  {:<11} {:<9} {}\n",
        "rank", "name", "type", "effect"
    ));
    for rank in Rank::ALL {
        let kind = if rank == Rank::THREE {
            "condition"
        } else if rank.is_constant_power() {
            "constant"
        } else {
            "one-shot"
        };
        out.push_str(&format!(
            "{:>4}  {:<11} {:<9} {}\n",
            rank.label(),
            rank.power_name(),
            kind,
            rank.power_text()
        ));
    }
    out.push_str(
        "\nHit points: every FACE-DOWN card is a blank 2 HP card, whatever its rank. Face-up, \
         a card has 2 HP — 3 for the Jack. So flipping a Jack raises its ceiling, and a \
         face-down Jack dies to two hits like anything else.\n",
    );
    out.push_str(
        "Lane wins need ALL of: opponent's side of the lane empty, every draw pile empty, \
         and the opponent's hand empty. Win two lanes to win.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GameConfig;
    use crate::testkit::Position;

    /// A card played from hand lands **face-down** (`game_rules.md` §4), so an observer who
    /// is not the player making the move may not be told which card it was.
    ///
    /// The rank travels inside [`Action::Play`] rather than on a card, which is how the
    /// CLI's move log came to announce "P1 played a 9 face-down" to the human sitting
    /// opposite: the knowledge mask that protects everything else on the board was never
    /// consulted.
    #[test]
    fn rule_4_a_play_does_not_name_the_card_to_the_opponent() {
        let mut p = Position::new(GameConfig::split_deck());
        p.hand(Player::P0, &[Rank::NINE]);
        let state = p.build();
        let action = Action::Play {
            rank: Rank::NINE,
            lane: 0,
        };

        // Both views of an action filter alike: the move log is a second entry point, not
        // a second policy.
        for describe in [describe_action, describe_move] {
            let mine = describe(&state, action, Some(Player::P0));
            assert!(mine.contains('9'), "the player making the move sees it\n{mine}");

            let theirs = describe(&state, action, Some(Player::P1));
            assert!(!theirs.contains('9'), "the opponent must not see it\n{theirs}");
            assert!(theirs.contains("lane 1"), "the lane is public\n{theirs}");
        }

        // The brief view drops the tuition, so a log of ten moves is ten lines.
        let brief = describe_move(&state, action, Some(Player::P0));
        assert!(!brief.contains("Nimble"), "{brief}");
        assert!(
            describe_action(&state, action, Some(Player::P0)).contains("Nimble"),
            "the menu still explains the power"
        );
    }

    /// Under the house 2 the card given back goes to the bottom of a draw pile, and §5
    /// makes its identity private to its owner. Under the rules-as-written 2 it goes to the
    /// public discard pile, and then naming it leaks nothing.
    #[test]
    fn rule_10a_a_bottomed_card_is_named_only_to_its_owner() {
        let mut p = Position::new(GameConfig::split_deck());
        p.hand(Player::P0, &[Rank::NINE]);
        let mut state = p.build();
        let action = Action::GiveBack { rank: Rank::NINE };

        for describe in [describe_action, describe_move] {
            assert!(describe(&state, action, Some(Player::P0)).contains('9'));
            assert!(!describe(&state, action, Some(Player::P1)).contains('9'));
        }

        state.config.two_power = crate::config::TwoPower::Discard;
        assert!(
            describe_action(&state, action, Some(Player::P1)).contains('9'),
            "a discard is public"
        );
    }

    /// The renderer must never show a base card's rank to anybody — including its owner
    /// (`game_rules.md` §3).
    #[test]
    fn rule_3_render_hides_base_cards_from_their_owner() {
        let state = GameState::new(GameConfig::split_deck(), 5);
        for observer in [Some(Player::P0), Some(Player::P1)] {
            let text = render(&state, observer);
            // Six base cards, all unknown, and nothing else is on the board yet.
            let unknown = text.matches("{? ²♥}").count();
            assert_eq!(
                unknown, 6,
                "expected all six base cards to render as unknown\n{text}"
            );
        }
    }

    /// A player must not see the opponent's hand contents, only its size.
    #[test]
    fn rule_5_render_hides_the_opponent_hand_contents() {
        let state = GameState::new(GameConfig::split_deck(), 5);
        let text = render(&state, Some(Player::P0));
        assert!(
            text.contains("P1   hand 5 "),
            "P1's hand size must be shown, and only its size\n{text}"
        );
        let own = hand_text(&state, Player::P0, Some(Player::P0));
        assert!(text.contains(&own), "P0 must see their own hand\n{text}");
    }

    /// Every card is the same width whatever its rank, damage or state, so a lane column
    /// never shifts sideways. The 10 is the case that forces it: two digits of rank in a
    /// field that has to hold `A` and `10` alike.
    #[test]
    fn every_card_token_is_the_same_width() {
        let mut p = Position::new(GameConfig::split_deck());
        p.face_up(0, Player::P0, Rank::TEN);
        p.face_up(0, Player::P0, Rank::JACK);
        p.face_down(0, Player::P0, Rank::ACE);
        p.base(0, Player::P0, Rank::KING);
        p.damage(0, Player::P0, 0, 1); // the 10, down to one hit point
        let state = p.build();

        for observer in [Some(Player::P0), Some(Player::P1), None] {
            for card in state.lanes[0].side(Player::P0) {
                let token = card_token(card, observer);
                assert_eq!(
                    token.chars().count(),
                    TOKEN_WIDTH,
                    "token {token:?} is not {TOKEN_WIDTH} columns"
                );
            }
            // And the grid rows built from them all agree.
            let widths: Vec<usize> = render(&state, observer)
                .lines()
                .filter(|l| l.contains('│'))
                .map(|l| l.chars().count())
                .collect();
            assert!(
                widths.windows(2).all(|w| w[0] == w[1]),
                "grid rows have differing widths: {widths:?}"
            );
        }
    }

    /// A damaged card shows the hit points it has left, and the Jack's third point appears
    /// only once it is face-up (§5).
    #[test]
    fn rule_5_a_token_shows_the_hit_points_remaining() {
        let mut p = Position::new(GameConfig::split_deck());
        p.face_up(0, Player::P0, Rank::JACK);
        p.face_down(0, Player::P0, Rank::JACK);
        p.damage(0, Player::P0, 0, 1);
        let state = p.build();
        let side = state.lanes[0].side(Player::P0);

        assert_eq!(card_token(&side[0], None), "[J ²♥]", "3 HP less 1 damage");
        assert_eq!(
            card_token(&side[1], None),
            "(J ²♥)",
            "a face-down Jack is a blank 2-HP card"
        );
    }

    /// Reveal mode is the only way to see the removed-unseen pool.
    #[test]
    fn removed_pool_is_hidden_unless_revealed() {
        let state = GameState::new(GameConfig::split_deck(), 5);
        assert!(!render(&state, Some(Player::P0)).contains("removed unseen"));
        assert!(render(&state, None).contains("removed unseen"));

        // §9b publishes it to both players.
        let mirrored = GameState::new(GameConfig::mirrored_removal(), 5);
        assert!(render(&mirrored, Some(Player::P0)).contains("removed from each deck"));
    }
}
