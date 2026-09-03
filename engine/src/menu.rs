//! The interactive prompt's action tree.
//!
//! [`GameState::legal_actions`](crate::GameState::legal_actions) returns a flat list, which
//! is the right shape for an agent and the wrong shape for a person: in a live midgame it is
//! sixty-odd lines, most of them the attacker × target cross-product of one lane. This
//! module reshapes that same list into a tree that asks one question at a time — *which
//! verb, which lane, which card* — the order the game is actually played in.
//!
//! # Every number means the same thing every time
//!
//! The tree is built so that what a player types is a property of the game, not of the list:
//!
//! - the five §4 verbs are always in the same order at the same numbers, whether or not they
//!   are available this turn;
//! - a lane's number is its lane number, always;
//! - a card's number is **its position in the column the board draws**, via
//!   [`display::column_slots`](crate::display::column_slots) — so "the second card in lane 2"
//!   means the same thing on the board and at the prompt.
//!
//! Holding that line means listing things that cannot be picked right now. A row with
//! nothing behind it keeps its place and shows `—` rather than closing the gap, because a
//! menu that renumbers itself is a menu you have to read every time.
//!
//! Nothing here decides legality. Every leaf of the tree is an [`Action`] taken verbatim from
//! the list the engine handed over, and every action in that list appears at exactly one
//! leaf, so the tree can neither invent a move nor hide one. `CLAUDE.md`: the engine is the
//! sole authority on legality.
//!
//! Information hiding is inherited rather than reimplemented: every rank this module prints
//! comes from `display::card_token` or from the acting player's own hand, and the combat
//! notes come from `display::combat_notes`, which reads face-up cards only.

use crate::action::{Action, Side};
use crate::display::{
    card_token, column_slots, combat_notes, knows, lane_label, Observer,
};
use crate::player::Player;
use crate::rank::Rank;
use crate::state::{GameState, Pending, ResolveKind};

/// One numbered line of a menu.
pub struct Row {
    /// The heading this row sits under. Consecutive rows sharing a heading are printed under
    /// one copy of it; an empty heading prints a blank separator and no heading.
    pub heading: String,
    /// What the row is called — `PLAY`, `LANE 2`, `CARD`. The number is not part of this:
    /// the renderer appends it, because it *is* the row's position.
    pub name: String,
    /// The card token and whatever is worth knowing before picking.
    pub note: String,
}

/// What picking a row does.
pub enum Pick {
    /// Apply this action.
    Take(Action),
    /// Ask the next question.
    Open(Box<Menu>),
    /// Nothing behind this row right now. It keeps its number so that the rows around it
    /// keep theirs.
    Unavailable,
}

/// A menu: a question, and the numbered answers to it.
///
/// `rows` and `picks` are parallel; row `i` is offered as number `i + 1`.
pub struct Menu {
    /// The question, printed above the rows.
    pub prompt: String,
    /// One line of context under the question. Empty when there is nothing to add, which is
    /// most of the time — the board says it better.
    pub hint: String,
    pub rows: Vec<Row>,
    pub picks: Vec<Pick>,
}

impl Menu {
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Walk `path` down from this menu. A path that has gone stale — which can only happen if
    /// the state changed underneath it — resolves to the deepest menu that still exists
    /// rather than panicking.
    pub fn at(&self, path: &[usize]) -> &Menu {
        match path.split_first() {
            None => self,
            Some((i, rest)) => match self.picks.get(*i) {
                Some(Pick::Open(sub)) => sub.at(rest),
                _ => self,
            },
        }
    }

    /// The menu as printed: the question, then the numbered rows under their headings.
    ///
    /// `nested` adds the row that goes back up a level. It lives here rather than at the
    /// call site so that it lines up with everything above it.
    pub fn render(&self, nested: bool) -> String {
        let mut out = format!("\n {}\n", self.prompt);
        if !self.hint.is_empty() {
            out.push_str(&format!("   {}\n", self.hint));
        }
        let width = self
            .rows
            .iter()
            .map(|r| r.name.chars().count())
            .max()
            .unwrap_or(0)
            .max(if nested { 4 } else { 0 });

        let mut current: Option<&str> = None;
        for (i, row) in self.rows.iter().enumerate() {
            if current != Some(row.heading.as_str()) {
                out.push('\n');
                if !row.heading.is_empty() {
                    out.push_str(&format!("   {}\n", row.heading));
                }
                current = Some(row.heading.as_str());
            }
            let number = match self.picks[i] {
                Pick::Unavailable => "—".to_string(),
                _ => format!("#{}", i + 1),
            };
            let head = format!("   {:<width$} {number:>2}", row.name);
            if row.note.is_empty() {
                out.push_str(&format!("{head}\n"));
            } else {
                out.push_str(&format!("{head}   {}\n", row.note));
            }
        }
        if nested {
            out.push_str(&format!("\n   {:<width$} {:>2}\n", "BACK", "#0"));
        }
        out
    }
}

/// Accumulates rows and picks together, so the two can never fall out of step.
struct Builder {
    prompt: String,
    hint: String,
    heading: String,
    rows: Vec<Row>,
    picks: Vec<Pick>,
}

impl Builder {
    fn new(prompt: impl Into<String>) -> Builder {
        Builder {
            prompt: prompt.into(),
            hint: String::new(),
            heading: String::new(),
            rows: Vec::new(),
            picks: Vec::new(),
        }
    }

    fn hint(mut self, hint: impl Into<String>) -> Builder {
        self.hint = hint.into();
        self
    }

    /// Rows added from here on sit under `heading`.
    fn heading(&mut self, heading: impl Into<String>) {
        self.heading = heading.into();
    }

    fn push(&mut self, name: impl Into<String>, note: impl Into<String>, pick: Pick) {
        self.rows.push(Row {
            heading: self.heading.clone(),
            name: name.into(),
            note: note.into(),
        });
        self.picks.push(pick);
    }

    /// A row that leads somewhere if `sub` is `Some`, and holds its number if not.
    fn step(&mut self, name: impl Into<String>, note: impl Into<String>, sub: Option<Menu>) {
        let pick = match sub {
            Some(menu) => Pick::Open(Box::new(menu)),
            None => Pick::Unavailable,
        };
        self.push(name, note, pick);
    }

    fn done(self) -> Menu {
        Menu {
            prompt: self.prompt,
            hint: self.hint,
            rows: self.rows,
            picks: self.picks,
        }
    }
}

/// Build the menu for whatever decision is on the table.
///
/// `legal` must be the list the engine just returned for this state; it is the only source of
/// actions.
pub fn build(state: &GameState, legal: &[Action], observer: Observer) -> Menu {
    if legal.is_empty() {
        return Builder::new("No decision to make.").done();
    }
    match state.pending.last() {
        None => build_main(state, legal, observer),
        Some(Pending::Foresight { .. }) => build_foresight(state, legal, observer),
        Some(Pending::ResolveOrder { kind, lane, remaining, .. }) => {
            build_resolve_order(state, legal, observer, *kind, *lane, remaining.len())
        }
        Some(Pending::QueenSource { lane, .. }) => build_queen_source(state, legal, observer, *lane),
        Some(Pending::GiveBack { .. }) => build_give_back(state, legal),
        Some(Pending::SplitTarget { lane, attackers, .. }) => {
            build_split_target(state, legal, observer, *lane, attackers.first().copied())
        }
    }
}

// ========================================================================== helpers ==

/// The lane an action happens in, for the actions that name one.
fn lane_of(action: Action) -> Option<usize> {
    match action {
        Action::Play { lane, .. }
        | Action::Flip { lane, .. }
        | Action::Attack { lane, .. }
        | Action::DeclarePair { lane, .. }
        | Action::Peek { lane, .. }
        | Action::ResolveNext { lane, .. }
        | Action::MoveHere { lane, .. } => Some(lane as usize),
        Action::Pass | Action::GiveBack { .. } | Action::SplitTarget { .. } => None,
    }
}

fn in_lane(actions: &[Action], lane: usize) -> Vec<Action> {
    actions
        .iter()
        .copied()
        .filter(|a| lane_of(*a) == Some(lane))
        .collect()
}

/// A menu over the lanes. Every lane keeps its number whether or not this step can reach it,
/// so lane 2 is `#2` in every menu that asks for a lane.
fn lane_menu(
    state: &GameState,
    prompt: impl Into<String>,
    mut sub: impl FnMut(usize) -> Option<Menu>,
) -> Menu {
    let mut b = Builder::new(prompt);
    for lane in 0..state.lane_count() {
        let name = format!("LANE {}", lane_label(lane));
        b.step(name, String::new(), sub(lane));
    }
    b.done()
}

/// A menu over one side of one lane.
///
/// Lists **every** card in the column, in the order the board draws it, so a card's number is
/// where it sits on the board. `detail` returns what to say about a card and the action to
/// take for it; returning `None` for the action leaves the row in place, showing `—`.
fn card_menu(
    state: &GameState,
    observer: Observer,
    prompt: impl Into<String>,
    lane: usize,
    owner: Player,
    name: &str,
    mut detail: impl FnMut(usize) -> (String, Option<Pick>),
) -> Menu {
    let mut b = Builder::new(prompt);
    for slot in column_slots(state, lane, owner, observer) {
        let token = card_token(&state.lanes[lane].side(owner)[slot], observer);
        let (text, pick) = detail(slot);
        let note = if text.is_empty() {
            token
        } else {
            format!("{token}   {text}")
        };
        b.push(name, note, pick.unwrap_or(Pick::Unavailable));
    }
    b.done()
}

/// The power's name, for a card whose rank this observer is entitled to know.
fn power_note(state: &GameState, lane: usize, owner: Player, slot: usize, observer: Observer) -> String {
    match state.at(lane, owner, slot) {
        Some(card) if knows(card, observer) => card.rank.power_name().to_string(),
        // A base card is hidden from its owner too (`game_rules.md` §3).
        _ => String::new(),
    }
}

// ====================================================================== the main phase ==

/// The five §4 actions, always in this order at these numbers.
///
/// Fixing the numbers costs a row for each verb that has nothing behind it and buys the
/// thing a player actually wants from a menu they will see a thousand times: `3` is attack,
/// this turn and every turn, whether or not there is anything to attack.
fn build_main(state: &GameState, legal: &[Action], observer: Observer) -> Menu {
    let of = |f: fn(&Action) -> bool| -> Vec<Action> {
        legal.iter().copied().filter(f).collect()
    };
    let plays = of(|a| matches!(a, Action::Play { .. }));
    let flips = of(|a| matches!(a, Action::Flip { .. }));
    let attacks = of(|a| matches!(a, Action::Attack { .. }));
    let pairs = of(|a| matches!(a, Action::DeclarePair { .. }));

    let mut b = Builder::new(format!(
        "Your move — {} action(s) left.",
        state.actions_remaining
    ));
    b.step(
        "PLAY",
        String::new(),
        (!plays.is_empty()).then(|| play_menu(state, &plays)),
    );
    b.step(
        "FLIP",
        String::new(),
        (!flips.is_empty()).then(|| flip_menu(state, &flips, observer)),
    );
    b.step(
        "ATTACK",
        String::new(),
        (!attacks.is_empty()).then(|| attack_menu(state, &attacks, observer)),
    );
    b.step(
        "PAIR",
        String::new(),
        (!pairs.is_empty()).then(|| pair_menu(state, &pairs, observer)),
    );
    b.push("PASS", String::new(), Pick::Take(Action::Pass));
    b.done()
}

/// PLAY: which card, then which lane. Identical ranks in hand collapse to one row — the
/// engine treats them as interchangeable, so offering both would be two rows for one choice.
fn play_menu(state: &GameState, plays: &[Action]) -> Menu {
    let hand = state.hand(state.acting_player());
    let mut ranks: Vec<Rank> = Vec::new();
    for action in plays {
        if let Action::Play { rank, .. } = action {
            if !ranks.contains(rank) {
                ranks.push(*rank);
            }
        }
    }

    let mut b = Builder::new("PLAY — which card?");
    b.heading("IN HAND");
    for rank in ranks {
        let copies = hand.iter().filter(|r| **r == rank).count();
        let label = if copies > 1 {
            format!("{rank} ×{copies}")
        } else {
            format!("{rank}")
        };
        let lanes: Vec<u8> = plays
            .iter()
            .filter_map(|a| match a {
                Action::Play { rank: r, lane } if *r == rank => Some(*lane),
                _ => None,
            })
            .collect();
        b.step(
            "CARD",
            format!("{label:<6} {}", rank.power_name()),
            Some(play_lane_menu(state, rank, &lanes)),
        );
    }
    b.done()
}

fn play_lane_menu(state: &GameState, rank: Rank, lanes: &[u8]) -> Menu {
    let mut b = Builder::new(format!("PLAY the {rank} face-down — which lane?"))
        .hint(format!("{}: {}", rank.power_name(), rank.power_text()));
    for lane in 0..state.lane_count() {
        let name = format!("LANE {}", lane_label(lane));
        match lanes.iter().find(|&&l| l as usize == lane) {
            Some(&l) => b.push(name, String::new(), Pick::Take(Action::Play { rank, lane: l })),
            None => b.push(name, String::new(), Pick::Unavailable),
        }
    }
    b.done()
}

/// FLIP: which lane, then which card.
fn flip_menu(state: &GameState, flips: &[Action], observer: Observer) -> Menu {
    let me = state.acting_player();
    lane_menu(state, "FLIP — which lane?", |lane| {
        let here = in_lane(flips, lane);
        if here.is_empty() {
            return None;
        }
        Some(card_menu(
            state,
            observer,
            format!("FLIP in lane {} — which card?", lane_label(lane)),
            lane,
            me,
            "CARD",
            |slot| {
                let action = here.iter().copied().find(
                    |a| matches!(a, Action::Flip { slot: s, .. } if *s as usize == slot),
                );
                match action {
                    Some(a) => (
                        power_note(state, lane, me, slot, observer),
                        Some(Pick::Take(a)),
                    ),
                    None => (String::new(), None),
                }
            },
        ))
    })
}

/// ATTACK: which lane, then which of your cards, then which of theirs.
fn attack_menu(state: &GameState, attacks: &[Action], observer: Observer) -> Menu {
    let me = state.acting_player();
    lane_menu(state, "ATTACK — which lane?", |lane| {
        let here = in_lane(attacks, lane);
        if here.is_empty() {
            return None;
        }
        Some(card_menu(
            state,
            observer,
            format!("ATTACK from lane {} — which card?", lane_label(lane)),
            lane,
            me,
            "CARD",
            |slot| {
                let mine: Vec<Action> = here
                    .iter()
                    .copied()
                    .filter(|a| matches!(a, Action::Attack { attacker, .. } if *attacker as usize == slot))
                    .collect();
                if mine.is_empty() {
                    return (String::new(), None);
                }
                let mut notes = Vec::new();
                if let Some(partner) = state.pair_partner(lane, me, slot) {
                    notes.push(format!(
                        "pair with #{}",
                        crate::display::card_number(state, lane, me, partner, observer)
                    ));
                }
                notes.extend(combat_notes(state, lane, slot, usize::MAX));
                (
                    notes.join("; "),
                    Some(Pick::Open(Box::new(target_menu(
                        state, observer, lane, slot, &mine,
                    )))),
                )
            },
        ))
    })
}

fn target_menu(
    state: &GameState,
    observer: Observer,
    lane: usize,
    attacker: usize,
    actions: &[Action],
) -> Menu {
    let them = state.acting_player().other();
    card_menu(
        state,
        observer,
        format!("ATTACK in lane {} — which enemy card?", lane_label(lane)),
        lane,
        them,
        "OPPONENT CARD",
        |slot| {
            let action = actions.iter().copied().find(
                |a| matches!(a, Action::Attack { target, .. } if *target as usize == slot),
            );
            match action {
                Some(a) => (
                    combat_notes(state, lane, attacker, slot).join("; "),
                    Some(Pick::Take(a)),
                ),
                None => (String::new(), None),
            }
        },
    )
}

/// PAIR: which lane, then the two cards. §5 — both must be face-up, the same rank, in the
/// same lane, and unpaired.
fn pair_menu(state: &GameState, pairs: &[Action], observer: Observer) -> Menu {
    let me = state.acting_player();
    lane_menu(state, "PAIR — which lane?", |lane| {
        let here = in_lane(pairs, lane);
        if here.is_empty() {
            return None;
        }
        Some(card_menu(
            state,
            observer,
            format!("PAIR in lane {} — which card?", lane_label(lane)),
            lane,
            me,
            "CARD",
            |slot| {
                let mine: Vec<Action> = here
                    .iter()
                    .copied()
                    .filter(|a| {
                        matches!(a, Action::DeclarePair { slot_a, slot_b, .. }
                                 if *slot_a as usize == slot || *slot_b as usize == slot)
                    })
                    .collect();
                if mine.is_empty() {
                    return (String::new(), None);
                }
                (
                    String::new(),
                    Some(Pick::Open(Box::new(partner_menu(
                        state, observer, lane, slot, &mine,
                    )))),
                )
            },
        ))
    })
}

fn partner_menu(
    state: &GameState,
    observer: Observer,
    lane: usize,
    first: usize,
    actions: &[Action],
) -> Menu {
    let me = state.acting_player();
    card_menu(
        state,
        observer,
        format!("PAIR in lane {} — with which card?", lane_label(lane)),
        lane,
        me,
        "CARD",
        |slot| {
            if slot == first {
                return (String::new(), None);
            }
            let action = actions.iter().copied().find(|a| {
                matches!(a, Action::DeclarePair { slot_a, slot_b, .. }
                         if *slot_a as usize == slot || *slot_b as usize == slot)
            });
            match action {
                Some(a) => ("never attack separately again".to_string(), Some(Pick::Take(a))),
                None => (String::new(), None),
            }
        },
    )
}

// ======================================================================= sub-decisions ==

/// The 4's Foresight reaches both sides of the board, so it asks for a lane, then a side,
/// then a card — rather than one long list in which a card's number would stop matching its
/// position in its own column.
fn build_foresight(state: &GameState, legal: &[Action], observer: Observer) -> Menu {
    let me = state.acting_player();
    lane_menu(state, "FORESIGHT — look in which lane?", |lane| {
        let here = in_lane(legal, lane);
        if here.is_empty() {
            return None;
        }
        let mut b = Builder::new(format!(
            "FORESIGHT in lane {} — whose side?",
            lane_label(lane)
        ))
        .hint("Only you learn it, and you keep knowing it. Base cards included.");
        for (name, side, owner) in [
            ("OPPONENT", Side::Theirs, me.other()),
            ("YOURS", Side::Mine, me),
        ] {
            let on_side: Vec<Action> = here
                .iter()
                .copied()
                .filter(|a| matches!(a, Action::Peek { side: s, .. } if *s == side))
                .collect();
            let sub = (!on_side.is_empty()).then(|| {
                card_menu(
                    state,
                    observer,
                    format!("FORESIGHT — which card in lane {}?", lane_label(lane)),
                    lane,
                    owner,
                    if owner == me { "CARD" } else { "OPPONENT CARD" },
                    |slot| {
                        let action = on_side.iter().copied().find(
                            |a| matches!(a, Action::Peek { slot: s, .. } if *s as usize == slot),
                        );
                        (String::new(), action.map(Pick::Take))
                    },
                )
            });
            b.step(name, String::new(), sub);
        }
        Some(b.done())
    })
}

/// A 5's flip list or a King's reactivation list — all in one lane, so this is one menu.
fn build_resolve_order(
    state: &GameState,
    legal: &[Action],
    observer: Observer,
    kind: ResolveKind,
    lane: u8,
    remaining: usize,
) -> Menu {
    let me = state.acting_player();
    let lane = lane as usize;
    card_menu(
        state,
        observer,
        format!(
            "{} in lane {} — resolve which card next? ({remaining} left)",
            kind.label(),
            lane_label(lane)
        ),
        lane,
        me,
        "CARD",
        |slot| {
            let action = legal.iter().copied().find(
                |a| matches!(a, Action::ResolveNext { slot: s, .. } if *s as usize == slot),
            );
            match action {
                Some(a) => (
                    power_note(state, lane, me, slot, observer),
                    Some(Pick::Take(a)),
                ),
                None => (String::new(), None),
            }
        },
    )
}

/// The Queen pulls an allied card in from another lane, so this asks for that lane first.
fn build_queen_source(
    state: &GameState,
    legal: &[Action],
    observer: Observer,
    queen_lane: u8,
) -> Menu {
    let me = state.acting_player();
    lane_menu(
        state,
        format!(
            "QUEEN — pull a card into lane {} from which lane?",
            lane_label(queen_lane)
        ),
        |lane| {
            let here = in_lane(legal, lane);
            if here.is_empty() {
                return None;
            }
            Some(card_menu(
                state,
                observer,
                format!("QUEEN — move which card out of lane {}?", lane_label(lane)),
                lane,
                me,
                "CARD",
                |slot| {
                    let action = here.iter().copied().find(
                        |a| matches!(a, Action::MoveHere { slot: s, .. } if *s as usize == slot),
                    );
                    (String::new(), action.map(Pick::Take))
                },
            ))
        },
    )
}

fn build_give_back(state: &GameState, legal: &[Action]) -> Menu {
    let me = state.acting_player();
    let (prompt, hint) = match state.config.two_power {
        crate::config::TwoPower::Bottom => (
            "VIEW — which card goes to the bottom of your pile?",
            "Private to you. You may give back the card you just drew.",
        ),
        crate::config::TwoPower::Discard => (
            "VIEW — which card do you discard?",
            "The discard pile is public, so the opponent will see this.",
        ),
    };
    let mut b = Builder::new(prompt).hint(hint);
    b.heading("IN HAND");
    for &action in legal {
        let Action::GiveBack { rank } = action else {
            continue;
        };
        let copies = state.hand(me).iter().filter(|r| **r == rank).count();
        let label = if copies > 1 {
            format!("{rank} ×{copies}")
        } else {
            format!("{rank}")
        };
        b.push(
            "CARD",
            format!("{label:<6} {}", rank.power_name()),
            Pick::Take(action),
        );
    }
    b.done()
}

fn build_split_target(
    state: &GameState,
    legal: &[Action],
    observer: Observer,
    lane: u8,
    attacker: Option<crate::card::CardId>,
) -> Menu {
    let them = state.acting_player().other();
    let lane = lane as usize;
    // The attacker is still on the board — no damage has landed yet — so its slot is what the
    // combat notes need in order to say whether a 9 or an 8 changes this half of the split.
    let attacker_slot = attacker
        .and_then(|id| state.locate(id))
        .map(|(_, _, slot)| slot)
        .unwrap_or(usize::MAX);

    card_menu(
        state,
        observer,
        "TWINSTRIKE — which card takes the second damage?",
        lane,
        them,
        "OPPONENT CARD",
        |slot| {
            let action = legal.iter().copied().find(
                |a| matches!(a, Action::SplitTarget { slot: s } if *s as usize == slot),
            );
            match action {
                Some(a) => (
                    combat_notes(state, lane, attacker_slot, slot).join("; "),
                    Some(Pick::Take(a)),
                ),
                None => (String::new(), None),
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::Agent;
    use crate::config::GameConfig;
    use crate::testkit::Position;

    /// Every leaf of the tree, in order.
    fn leaves(menu: &Menu) -> Vec<Action> {
        let mut out = Vec::new();
        for pick in &menu.picks {
            match pick {
                Pick::Take(a) => out.push(*a),
                Pick::Open(sub) => out.extend(leaves(sub)),
                Pick::Unavailable => {}
            }
        }
        out
    }

    /// The whole tree as printed, every level of it. `Menu::render` prints only the level it
    /// is called on.
    fn render_all(menu: &Menu) -> String {
        let mut out = menu.render(false);
        for pick in &menu.picks {
            if let Pick::Open(sub) = pick {
                out.push_str(&render_all(sub));
            }
        }
        out
    }

    /// A taunting Jack is the case that shows why an unavailable row still has to be drawn:
    /// §6 makes the Jack the only card in its lane that can be attacked, so every *other*
    /// enemy card is a `—` — and the Jack keeps the number it has on the board.
    #[test]
    fn rule_6_a_taunt_leaves_the_other_targets_numbered_but_unavailable() {
        let mut p = Position::new(GameConfig::split_deck());
        p.face_up(0, Player::P0, Rank::TEN);
        p.base(0, Player::P1, Rank::ACE);
        p.face_up(0, Player::P1, Rank::EIGHT);
        p.face_up(0, Player::P1, Rank::JACK);
        let state = p.build();

        let menu = build(&state, &state.legal_actions(), Some(Player::P0));
        let Pick::Open(attack) = &menu.picks[2] else {
            panic!("attacking must be on\n{}", menu.render(false));
        };
        let Pick::Open(mine) = &attack.picks[0] else {
            panic!("lane 1 must be reachable\n{}", attack.render(false));
        };
        let Pick::Open(targets) = &mine.picks[0] else {
            panic!("the 10 must be able to attack\n{}", mine.render(false));
        };

        let text = targets.render(false);
        assert_eq!(targets.len(), 3, "every enemy card in the lane is listed\n{text}");
        assert!(matches!(targets.picks[0], Pick::Unavailable), "base card\n{text}");
        assert!(matches!(targets.picks[1], Pick::Unavailable), "the 8, taunted\n{text}");
        assert!(matches!(targets.picks[2], Pick::Take(_)), "the Jack\n{text}");
        assert!(targets.rows[2].note.starts_with("[J "), "{text}");
    }

    /// The tree is a reshaping of the engine's list, not a filter on it. Every legal action
    /// must be reachable, or the menu has quietly made a move impossible.
    ///
    /// `DeclarePair` is the one action that appears twice — once from each of its two members
    /// — so the comparison is by set, not by count.
    #[test]
    fn menu_offers_every_legal_action_and_nothing_else() {
        for seed in 0..40u64 {
            let mut state = GameState::new(GameConfig::split_deck(), seed);
            let mut agent = crate::RandomAgent::new(seed ^ 0x5EED);
            for _ in 0..60 {
                if state.outcome.is_over() {
                    break;
                }
                let legal = state.legal_actions();
                let observer = Some(state.acting_player());
                let menu = build(&state, &legal, observer);
                let offered = leaves(&menu);
                for action in &legal {
                    assert!(
                        offered.contains(action),
                        "seed {seed}: menu does not offer {action}\n{}",
                        render_all(&menu)
                    );
                }
                for action in &offered {
                    assert!(
                        legal.contains(action),
                        "seed {seed}: menu offers illegal {action}"
                    );
                }
                let choice = agent.choose(&state, &legal);
                state.apply_trusted(choice);
            }
        }
    }

    /// The five verbs are always in the same order at the same numbers, whatever is legal.
    #[test]
    fn the_five_verbs_keep_their_numbers() {
        let names = ["PLAY", "FLIP", "ATTACK", "PAIR", "PASS"];

        // A fresh deal: cards in hand, nothing face-up, so only PLAY and PASS can be taken.
        let opening = GameState::new(GameConfig::split_deck(), 3);
        let menu = build(&opening, &opening.legal_actions(), Some(Player::P0));
        assert_eq!(menu.len(), 5);
        for (row, name) in menu.rows.iter().zip(names) {
            assert_eq!(row.name, name);
        }
        assert!(matches!(menu.picks[2], Pick::Unavailable), "nothing to attack");
        assert!(matches!(menu.picks[3], Pick::Unavailable), "nothing to pair");
        assert!(matches!(menu.picks[4], Pick::Take(Action::Pass)));

        // A position where attacking is on and playing is not: the numbers do not move.
        let mut p = Position::new(GameConfig::split_deck());
        p.face_up(0, Player::P0, Rank::SEVEN);
        p.face_up(0, Player::P1, Rank::FOUR);
        let midgame = p.build();
        let menu = build(&midgame, &midgame.legal_actions(), Some(Player::P0));
        assert_eq!(menu.len(), 5);
        for (row, name) in menu.rows.iter().zip(names) {
            assert_eq!(row.name, name);
        }
        assert!(matches!(menu.picks[0], Pick::Unavailable), "hand is empty");
        assert!(matches!(menu.picks[2], Pick::Open(_)), "attacking is on");
    }

    /// A card's number is its position in the column the board draws, which is *not* its
    /// slot: the observer's own base card is stored first and drawn last.
    #[test]
    fn a_cards_number_is_where_the_board_draws_it() {
        let mut p = Position::new(GameConfig::split_deck());
        p.base(0, Player::P0, Rank::QUEEN); // slot 0, drawn last
        p.face_up(0, Player::P0, Rank::SEVEN); // slot 1, drawn first
        p.face_up(0, Player::P1, Rank::FOUR);
        let state = p.build();

        assert_eq!(
            column_slots(&state, 0, Player::P0, Some(Player::P0)),
            vec![1, 0],
            "your own base card is drawn at the bottom of your column"
        );
        assert_eq!(
            column_slots(&state, 0, Player::P1, Some(Player::P0)),
            vec![0],
            "the opponent's column starts at their base card"
        );

        // So the 7 — slot 1 — is the card a player asks for as #1.
        let legal = state.legal_actions();
        let menu = build(&state, &legal, Some(Player::P0));
        let Pick::Open(attack) = &menu.picks[2] else {
            panic!("attacking must be on\n{}", menu.render(false));
        };
        let Pick::Open(cards) = &attack.picks[0] else {
            panic!("lane 1 must be reachable\n{}", attack.render(false));
        };
        assert!(cards.rows[0].note.starts_with("[7 "), "{}", cards.render(false));
        assert!(matches!(cards.picks[0], Pick::Open(_)), "the 7 can attack");
        assert!(
            matches!(cards.picks[1], Pick::Unavailable),
            "the base card cannot, but keeps its row"
        );
    }

    /// The menu is built from the acting player's own point of view, and no level of it may
    /// name a rank that point of view does not include — the opponent's played face-down
    /// cards, or the actor's *own* base cards (`game_rules.md` §3).
    #[test]
    fn rule_5_menu_never_names_a_rank_the_actor_cannot_see() {
        let mut p = Position::new(GameConfig::split_deck());
        p.face_up(0, Player::P0, Rank::TEN);
        p.face_down(0, Player::P1, Rank::KING);
        p.base(1, Player::P0, Rank::QUEEN);
        p.unlock(); // so the base card is flippable and reaches the menu at all
        p.hand(Player::P0, &[Rank::THREE]);
        let state = p.build();

        let legal = state.legal_actions();
        assert!(
            legal.iter().any(|a| matches!(a, Action::Flip { lane: 1, .. })),
            "the base card must be offered, or the test proves nothing"
        );
        let text = render_all(&build(&state, &legal, Some(Player::P0)));
        for token in ["(K ", "[K ", "{Q ", "[Q "] {
            assert!(!text.contains(token), "menu leaked a hidden {token}\n{text}");
        }
        // The actor's own hand is theirs to see, so the 3 must still be named.
        assert!(text.contains(" 3 "), "P0's own hand is missing\n{text}");
    }
}
