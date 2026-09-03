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
//! One detail is easy to get wrong and is handled explicitly below: **`(?)` must be used
//! for the owner's own base cards.** Base cards are hidden from their owner too (§3), which
//! is exactly the fact a careless renderer misses.
//!
//! Hit points, by contrast, need no filtering at all: §5 makes every face-down card a blank
//! 2-HP card whatever its rank, so a face-down card's hit points are common knowledge and
//! printing them reveals nothing. (Were it otherwise — were a face-down Jack really 3 HP —
//! then rendering "1/3 hp" would announce the Jack, and so would simply watching it survive
//! two hits, since damage is public.)
//!
//! # Lanes and slots are numbered from 1 here, and only here
//!
//! The engine indexes lanes and slots from 0, and so do [`Action`], the Python action
//! dicts, and every test. Humans count from 1, so this module — the display layer, which
//! nothing else reads back — adds one on the way out via [`lane_label`] and [`slot_label`].
//! Nothing parses these strings, so the two conventions cannot collide.

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

/// The slot number a human reads, given the engine's 0-based slot index.
#[inline]
pub fn slot_label(slot: impl Into<usize>) -> usize {
    slot.into() + 1
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

/// One card as a short token, e.g. `[J]`, `(K)`, `(?)`.
///
/// Brackets mean face-up (public); parentheses mean face-down. A face-down card shows its
/// rank only when this observer is entitled to know it.
pub(crate) fn card_token(state: &GameState, card: &Card, observer: Observer) -> String {
    let visible = knows(card, observer);
    let label = if visible { card.rank.label() } else { "?" };
    let body = if card.face_up {
        format!("[{label}]")
    } else {
        format!("({label})")
    };

    let mut tags: Vec<String> = Vec::new();
    if card.is_base {
        tags.push("base".to_string());
    } else if card.entered_as_base {
        // Public: a card that entered as a base card and was moved by a Queen is no longer
        // a base card, but its owner still may not look at it (§3).
        tags.push("ex-base".to_string());
    }
    if card.damage > 0 {
        // Safe to print for any card: damage is public (§5), and so is max HP, because a
        // face-down card is always 2 HP regardless of rank. There is nothing to leak.
        tags.push(format!("{}/{}hp", card.hp_remaining(), card.max_hp()));
    }
    if card.is_frozen(state.ply) {
        tags.push("FROZEN".to_string());
    }
    if let Some(pid) = card.pair_id {
        tags.push(format!("pair{}", pid.0));
    }
    if card.face_up && card.owner == state.to_move && card.attacks_used >= card.attack_allowance {
        tags.push("spent".to_string());
    } else if card.face_up && card.owner == state.to_move && card.attack_allowance > 1 {
        tags.push(format!(
            "{}atk left",
            card.attack_allowance - card.attacks_used
        ));
    }

    if tags.is_empty() {
        body
    } else {
        format!("{body}{}", tags.join(","))
    }
}

/// A hand: contents if the observer owns it, otherwise just the count.
fn hand_text(state: &GameState, owner: Player, observer: Observer) -> String {
    let hand = state.hand(owner);
    let entitled = observer.is_none() || observer == Some(owner);
    if entitled {
        if hand.is_empty() {
            "(empty)".to_string()
        } else {
            hand.iter()
                .map(|r| r.label())
                .collect::<Vec<_>>()
                .join(" ")
        }
    } else {
        format!("{} card(s)", hand.len())
    }
}

/// The discard pile — public to both players at any time (§5), so no filtering.
fn discard_text(state: &GameState, owner: Player) -> String {
    let d = &state.discards[owner.idx()];
    if d.is_empty() {
        "-".to_string()
    } else {
        let mut sorted = d.clone();
        sorted.sort_unstable();
        format!(
            "{} ({})",
            sorted.len(),
            sorted
                .iter()
                .map(|r| r.label())
                .collect::<Vec<_>>()
                .join(" ")
        )
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

/// Render the whole position from `observer`'s point of view.
pub fn render(state: &GameState, observer: Observer) -> String {
    let mut out = String::new();
    let me = observer.unwrap_or(state.to_move);
    let them = me.other();

    let rule = "=".repeat(78);
    let thin = "-".repeat(78);

    out.push_str(&rule);
    out.push('\n');
    out.push_str(&format!(
        " Duel 52 · {} · engine {}\n",
        state.config.summary(),
        crate::VERSION
    ));
    out.push_str(&format!(" {}\n", state.header()));
    if observer.is_none() {
        out.push_str(" *** REVEAL MODE: showing hidden information ***\n");
    }
    out.push_str(&thin);
    out.push('\n');

    // --- Resources -------------------------------------------------------------------
    let pile_text = |p: Player| {
        if state.shared_pile() {
            format!("shared draw pile: {}", state.piles[0].len())
        } else {
            format!("draw pile: {}", state.pile(p).len())
        }
    };
    out.push_str(&format!(
        " {label:<14} hand: {hand:<26} discard: {disc}\n",
        label = format!("{them} (opponent)"),
        hand = hand_text(state, them, observer),
        disc = discard_text(state, them),
    ));
    out.push_str(&format!(
        " {label:<14} hand: {hand:<26} discard: {disc}\n",
        label = format!("{me} (you)"),
        hand = hand_text(state, me, observer),
        disc = discard_text(state, me),
    ));
    if state.shared_pile() {
        out.push_str(&format!(" {}\n", pile_text(me)));
    } else {
        out.push_str(&format!(
            " {} · opponent {}\n",
            pile_text(me),
            state.pile(them).len()
        ));
    }
    if let Some(bottom) = bottom_knowledge_text(state, observer) {
        out.push_str(&format!(" you privately know: {bottom}\n"));
    }
    if state.removed_revealed {
        // §9b only: the removed multiset is public in the mirrored-removal variant.
        let mut ranks: Vec<Rank> = state.removed[0].clone();
        ranks.sort_unstable();
        out.push_str(&format!(
            " removed from each deck (public, §9b): {}\n",
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
    out.push_str(&thin);
    out.push('\n');

    // --- Lanes -----------------------------------------------------------------------
    for lane in 0..state.lane_count() {
        let won_note = {
            let mut notes = Vec::new();
            if state.base_unlocked {
                if state.lanes[lane].side(them).is_empty() && state.hand(them).is_empty() {
                    notes.push(format!("{me} has won this lane"));
                }
                if state.lanes[lane].side(me).is_empty() && state.hand(me).is_empty() {
                    notes.push(format!("{them} has won this lane"));
                }
            }
            if notes.is_empty() {
                String::new()
            } else {
                format!("   <- {}", notes.join(" and "))
            }
        };
        out.push_str(&format!(" LANE {}{won_note}\n", lane_label(lane)));

        for (label, owner) in [("opp", them), ("you", me)] {
            let side = state.lanes[lane].side(owner);
            let cells: Vec<String> = side
                .iter()
                .enumerate()
                .map(|(slot, card)| {
                    format!(
                        "#{} {}",
                        slot_label(slot),
                        card_token(state, card, observer)
                    )
                })
                .collect();
            let body = if cells.is_empty() {
                "(empty)".to_string()
            } else {
                cells.join("   ")
            };
            out.push_str(&format!("   {label} {owner} | {body}\n"));
        }
    }
    out.push_str(&thin);
    out.push('\n');

    // --- What is being asked ---------------------------------------------------------
    if state.outcome.is_over() {
        out.push_str(&format!(" GAME OVER: {}\n", state.outcome));
    } else {
        out.push_str(&format!(" {}\n", state.prompt()));
        if let Some(context) = pending_context(state) {
            out.push_str(&format!(" {context}\n"));
        }
    }
    out.push_str(&rule);
    out.push('\n');
    out
}

/// A sentence explaining *why* a sub-decision is being asked, so the owner can check the
/// engine is doing what the rules say.
fn pending_context(state: &GameState) -> Option<String> {
    let text = match state.pending.last()? {
        Pending::Foresight { .. } => {
            "A 4 flipped: Foresight. Look at any one face-down card on the board — including \
             base cards, yours or the opponent's. Only you learn it."
                .to_string()
        }
        Pending::ResolveOrder { kind, lane, remaining, .. } => match kind {
            crate::state::ResolveKind::FiveFlip => format!(
                "A 5 flipped in lane {}: it flips ALL your face-down cards there \
                 ({} left). Pick the order; each power fully resolves before the next.",
                lane_label(*lane),
                remaining.len()
            ),
            crate::state::ResolveKind::KingEmpower => format!(
                "A King flipped in lane {}: your face-up cards there refire their \
                 powers ({} left). Not Kings, not constant powers.",
                lane_label(*lane),
                remaining.len()
            ),
        },
        Pending::QueenSource { lane, .. } => {
            let lane = lane_label(*lane);
            format!(
                "A Queen flipped in lane {lane}: move one allied card from ANOTHER lane \
                 into lane {lane}. It keeps its damage and does not refire its power."
            )
        }
        Pending::GiveBack { .. } => match state.config.two_power {
            crate::config::TwoPower::Bottom => {
                "A 2 flipped: you drew a card, now put one from hand on the BOTTOM of your \
                 pile. You may give back the card you just drew."
                    .to_string()
            }
            crate::config::TwoPower::Discard => {
                "A 2 flipped (rules-as-written): you drew a card, now DISCARD one from \
                 hand."
                    .to_string()
            }
        },
        Pending::SplitTarget { .. } => {
            "A 10 attacked: Twinstrike deals 1 damage each to TWO cards. Choose the second \
             target. No damage has landed yet."
                .to_string()
        }
    };
    Some(text)
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
            Some(card) => card_token(state, card, observer),
            None => "<gone>".to_string(),
        }
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
                slot_label(slot),
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
                .map(|p| format!(" (PAIR with #{}, 2 dmg)", slot_label(p)))
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
                slot_label(attacker),
                token(lane_i, me, attacker as usize),
                slot_label(target),
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
                slot_label(slot_a),
                slot_label(slot_b),
            )
        }

        Action::Pass => "PASS  forfeit the rest of this turn".to_string(),

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
                slot_label(slot),
                token(lane as usize, owner, slot as usize)
            )
        }

        Action::ResolveNext { lane, slot } => format!(
            "NEXT  lane {} #{} {}",
            lane_label(lane),
            slot_label(slot),
            token(lane as usize, me, slot as usize)
        ),

        Action::MoveHere { lane, slot } => format!(
            "MOVE  lane {} #{} {} into the Queen's lane{}",
            lane_label(lane),
            slot_label(slot),
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
                slot_label(slot),
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
            // Six base cards, all unknown, so six `(?)` tokens and nothing else on board.
            let unknown = text.matches("(?)").count();
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
        assert!(text.contains("5 card(s)"), "P1's hand size must be shown\n{text}");
        let own = hand_text(&state, Player::P0, Some(Player::P0));
        assert!(text.contains(&own), "P0 must see their own hand\n{text}");
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
