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
//! Two details are easy to leak by accident and are handled explicitly below:
//!
//! 1. **Never print max HP for a card whose rank the observer does not know.** A face-down
//!    card on 1 damage shows one damage marker at the table; rendering it as "1/3 hp" would
//!    announce that it is a Jack.
//! 2. **`??` must be used for the owner's own base cards.** Base cards are hidden from their
//!    owner too (§3), which is exactly the fact a careless renderer gets wrong.

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

/// Does `observer` know this card's rank?
fn knows(card: &Card, observer: Observer) -> bool {
    match observer {
        None => true,
        Some(p) => card.rank_known_to(p),
    }
}

/// One card as a short token, e.g. `[J]`, `(K)`, `(?)`.
///
/// Brackets mean face-up (public); parentheses mean face-down. A face-down card shows its
/// rank only when this observer is entitled to know it.
fn card_token(state: &GameState, card: &Card, observer: Observer) -> String {
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
        if visible {
            tags.push(format!("{}/{}hp", card.hp_remaining(), card.rank.max_hp()));
        } else {
            // Damage is public; max HP would give the rank away. See the module docs.
            tags.push(format!("dmg{}", card.damage));
        }
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
        out.push_str(&format!(" LANE {lane}{won_note}\n"));

        for (label, owner) in [("opp", them), ("you", me)] {
            let side = state.lanes[lane].side(owner);
            let cells: Vec<String> = side
                .iter()
                .enumerate()
                .map(|(slot, card)| format!("#{slot} {}", card_token(state, card, observer)))
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
                "A 5 flipped in lane {lane}: it flips ALL your face-down cards there \
                 ({} left). Pick the order; each power fully resolves before the next.",
                remaining.len()
            ),
            crate::state::ResolveKind::KingEmpower => format!(
                "A King flipped in lane {lane}: your face-up cards there refire their \
                 powers ({} left). Not Kings, not constant powers.",
                remaining.len()
            ),
        },
        Pending::QueenSource { lane, .. } => format!(
            "A Queen flipped in lane {lane}: move one allied card from ANOTHER lane into \
             lane {lane}. It keeps its damage and does not refire its power."
        ),
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

/// Describe a legal action in plain language, from `observer`'s point of view.
///
/// Used for the CLI's numbered menu. Like [`render`], it must not leak: every rank it names
/// comes from a card the observer is entitled to know, and the combat notes are derived
/// only from **face-up** information — you do not get told that the face-down card you are
/// about to attack is a Jack.
pub fn describe_action(state: &GameState, action: Action, observer: Observer) -> String {
    let me = state.acting_player();
    let them = me.other();

    let token = |lane: usize, owner: Player, slot: usize| -> String {
        match state.at(lane, owner, slot) {
            Some(card) => card_token(state, card, observer),
            None => "<gone>".to_string(),
        }
    };

    match action {
        Action::Play { rank, lane } => format!(
            "PLAY  {rank} face-down into lane {lane}          [{}: {}]",
            rank.power_name(),
            rank.power_text()
        ),

        Action::Flip { lane, slot } => {
            let card = state.at(lane as usize, me, slot as usize);
            match card {
                Some(c) if knows(c, observer) => format!(
                    "FLIP  lane {lane} #{slot} {} -> reveals {} [{}: {}]",
                    token(lane as usize, me, slot as usize),
                    c.rank,
                    c.rank.power_name(),
                    c.rank.power_text()
                ),
                // A base card, or a Queen-moved ex-base card: even you do not know it (§3).
                _ => format!(
                    "FLIP  lane {lane} #{slot} {} -> you do not know what this is",
                    token(lane as usize, me, slot as usize)
                ),
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
                .map(|p| format!(" (PAIR with #{p}, 2 dmg)"))
                .unwrap_or_default();
            let mut notes: Vec<String> = Vec::new();
            if let Some(atk) = state.at(lane_i, me, attacker as usize) {
                if atk.rank == Rank::TEN {
                    notes.push("twinstrike".to_string());
                }
                if atk.rank == Rank::NINE {
                    notes.push("nimble".to_string());
                }
            }
            // Only face-up defenders reveal anything, and their powers are public anyway.
            if let Some(def) = state.at(lane_i, them, target as usize) {
                if def.has_live_power(Rank::EIGHT) {
                    notes.push("8 retaliates for 1".to_string());
                }
                if def.has_live_power(Rank::JACK) {
                    notes.push("Jack, 3 HP".to_string());
                }
                if def.has_live_power(Rank::NINE) {
                    notes.push("9 blocks the twinstrike split".to_string());
                }
            }
            let note = if notes.is_empty() {
                String::new()
            } else {
                format!("   <{}>", notes.join("; "))
            };
            format!(
                "ATK   lane {lane}: your #{attacker} {}{paired} -> opp #{target} {}{note}",
                token(lane_i, me, attacker as usize),
                token(lane_i, them, target as usize),
            )
        }

        Action::DeclarePair {
            lane,
            slot_a,
            slot_b,
        } => {
            let rank = state
                .at(lane as usize, me, slot_a as usize)
                .map(|c| c.rank.label())
                .unwrap_or("?");
            format!(
                "PAIR  lane {lane}: #{slot_a} + #{slot_b} (two {rank}s) — one action for 2 damage, \
                 but they can never attack separately again"
            )
        }

        Action::Pass => "PASS  forfeit the rest of this turn".to_string(),

        Action::Peek { side, lane, slot } => {
            let owner = match side {
                Side::Mine => me,
                Side::Theirs => them,
            };
            let whose = match side {
                Side::Mine => "your",
                Side::Theirs => "opponent's",
            };
            format!(
                "PEEK  {whose} lane {lane} #{slot} {}",
                token(lane as usize, owner, slot as usize)
            )
        }

        Action::ResolveNext { lane, slot } => format!(
            "NEXT  lane {lane} #{slot} {}",
            token(lane as usize, me, slot as usize)
        ),

        Action::MoveHere { lane, slot } => format!(
            "MOVE  lane {lane} #{slot} {} into the Queen's lane (keeps damage, keeps freeze, \
             stops being a base card)",
            token(lane as usize, me, slot as usize)
        ),

        Action::GiveBack { rank } => match state.config.two_power {
            crate::config::TwoPower::Bottom => {
                format!("BACK  put {rank} on the bottom of your draw pile")
            }
            crate::config::TwoPower::Discard => format!("BACK  discard {rank}"),
        },

        Action::SplitTarget { slot } => {
            // The twinstrike's lane is whatever the pending node says.
            let lane = match state.pending.last() {
                Some(Pending::SplitTarget { lane, .. }) => *lane as usize,
                _ => 0,
            };
            format!(
                "2ND   twinstrike's second target: opp lane {lane} #{slot} {}",
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
    out.push_str("\nHit points: 2 for every card, 3 for the Jack (face-down too).\n");
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
