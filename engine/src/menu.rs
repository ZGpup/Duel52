//! The interactive prompt's action tree.
//!
//! [`GameState::legal_actions`](crate::GameState::legal_actions) returns a flat list, which
//! is the right shape for an agent and the wrong shape for a person: in a live midgame it
//! is sixty-odd lines, most of them the attacker × target cross-product of one lane. This
//! module reshapes that same list into two levels — **pick a card, then pick what it does**
//! — so what a human reads is one line per card they own.
//!
//! Nothing here decides legality. Every leaf of the tree is an [`Action`] taken verbatim
//! from the list the engine handed over, and every action in that list appears at exactly
//! one leaf, so the tree can neither invent a move nor hide one. `CLAUDE.md`: the engine is
//! the sole authority on legality.
//!
//! Information hiding is inherited rather than reimplemented: every rank this module prints
//! comes from `display::card_token` or from the acting player's own hand, and the combat
//! notes come from `display::combat_notes`, which reads face-up cards only.

use crate::action::{Action, Side};
use crate::display::{card_token, combat_notes, knows, lane_label, slot_label, Observer};
use crate::rank::Rank;
use crate::state::{GameState, Pending, ResolveKind};

/// One numbered line of a menu.
pub struct Row {
    /// The heading this row sits under. Consecutive rows sharing a heading are printed
    /// under one copy of it; an empty heading prints a blank separator and no heading.
    pub heading: String,
    /// The row's own name — the card, the lane, the rank.
    pub label: String,
    /// What picking it means.
    pub note: String,
    /// The row compressed onto one line of the menu *above* it, for when a card's only
    /// move has to be stated on the line that takes it. Kept separate from `note` because
    /// the two have different budgets: a second menu has a whole screen for one power's
    /// text, while the card list has to fit a row per card inside the board's width.
    pub summary: String,
}

/// What picking a row does.
pub enum Pick {
    /// Apply this action straight away. Either the row *is* a complete action (`pass`), or
    /// it is the only thing its card can do — a second menu holding one option is a
    /// keystroke that asks nothing.
    Take(Action),
    /// Open a second menu. Only ever one level deeper: a card, then what it does.
    Open(Box<Menu>),
}

/// A menu: a question, and the numbered answers to it.
///
/// `rows` and `picks` are parallel; row `i` is offered as number `i + 1`.
pub struct Menu {
    /// The question, printed above the rows.
    pub prompt: String,
    /// One line of context under the question — a power's full text, a rule that is about
    /// to bite. Empty when there is nothing to add.
    pub hint: String,
    pub rows: Vec<Row>,
    pub picks: Vec<Pick>,
}

impl Menu {
    /// How many numbered rows this menu offers.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Walk `path` down from this menu. A path that has gone stale — which can only happen
    /// if the state changed underneath it — resolves to the deepest menu that still exists
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

    /// The menu as printed: the question, the hint, then the numbered rows under their
    /// headings.
    pub fn render(&self) -> String {
        let mut out = format!("\n {}\n", self.prompt);
        if !self.hint.is_empty() {
            out.push_str(&format!("   {}\n", self.hint));
        }
        // Align the notes into a column, but never let one long card token push the whole
        // board's worth of notes off the right edge.
        let width = self
            .rows
            .iter()
            .map(|r| r.label.chars().count())
            .max()
            .unwrap_or(0)
            .min(30);

        let mut current: Option<&str> = None;
        for (i, row) in self.rows.iter().enumerate() {
            if current != Some(row.heading.as_str()) {
                out.push('\n');
                if !row.heading.is_empty() {
                    out.push_str(&format!("   {}\n", row.heading));
                }
                current = Some(row.heading.as_str());
            }
            let n = i + 1;
            if row.note.is_empty() {
                out.push_str(&format!("   {n:>3}. {}\n", row.label));
            } else {
                out.push_str(&format!(
                    "   {n:>3}. {:<width$}   {}\n",
                    row.label, row.note
                ));
            }
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

    fn push(&mut self, label: impl Into<String>, note: impl Into<String>, pick: Pick) {
        let (label, note) = (label.into(), note.into());
        let summary = if note.is_empty() {
            label.clone()
        } else {
            format!("{label} — {note}")
        };
        self.rows.push(Row {
            heading: self.heading.clone(),
            label,
            note,
            summary,
        });
        self.picks.push(pick);
    }

    fn take(&mut self, label: impl Into<String>, note: impl Into<String>, action: Action) {
        self.push(label, note, Pick::Take(action));
    }

    /// A row whose compressed form is not just its label and note — because the note is
    /// too long for the line above.
    fn take_summarised(
        &mut self,
        label: impl Into<String>,
        note: impl Into<String>,
        summary: impl Into<String>,
        action: Action,
    ) {
        self.push(label, note, Pick::Take(action));
        let last = self.rows.last_mut().expect("just pushed");
        last.summary = summary.into();
    }

    fn open(&mut self, label: impl Into<String>, note: impl Into<String>, sub: Menu) {
        self.push(label, note, Pick::Open(Box::new(sub)));
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
/// `legal` must be the list the engine just returned for this state; it is the only source
/// of actions.
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
        Some(Pending::QueenSource { lane, .. }) => {
            build_queen_source(state, legal, observer, *lane)
        }
        Some(Pending::GiveBack { .. }) => build_give_back(state, legal),
        Some(Pending::SplitTarget { lane, attackers, .. }) => {
            build_split_target(state, legal, observer, *lane, attackers.first().copied())
        }
    }
}

// ====================================================================== the main phase ==

/// Pick a card, then pick what it does.
///
/// The grouping is by *subject*: the card whose action it is. That is the one grouping under
/// which every §4 action falls into exactly one place — a `Play` belongs to a card in hand,
/// a `Flip` and an `Attack` to a card on the board — and it is also how a person thinks
/// about a turn ("what does my 9 do?"), rather than how the action encoding is shaped.
fn build_main(state: &GameState, legal: &[Action], observer: Observer) -> Menu {
    let me = state.acting_player();
    let mut b = Builder::new(format!(
        "Pick a card to act with — {} action(s) left this turn.",
        state.actions_remaining
    ))
    .hint("A number picks a card; a card with one legal move takes it at once.");

    // --- IN HAND: one row per distinct rank, since identical ranks are interchangeable. --
    let hand = state.hand(me);
    let mut ranks: Vec<Rank> = Vec::new();
    for action in legal {
        if let Action::Play { rank, .. } = action {
            if !ranks.contains(rank) {
                ranks.push(*rank);
            }
        }
    }
    if !ranks.is_empty() {
        b.heading(format!("IN HAND ({} card(s))", hand.len()));
        for rank in ranks {
            let lanes: Vec<u8> = legal
                .iter()
                .filter_map(|a| match a {
                    Action::Play { rank: r, lane } if *r == rank => Some(*lane),
                    _ => None,
                })
                .collect();
            let copies = hand.iter().filter(|r| **r == rank).count();
            let label = if copies > 1 {
                format!("{rank} ×{copies}")
            } else {
                format!("{rank}")
            };
            let sub = lane_menu(state, rank, &lanes);
            if let [only] = lanes[..] {
                // A one-lane config: there is no lane to choose.
                b.take(
                    label,
                    format!("play face-down into lane {}", lane_label(only)),
                    Action::Play { rank, lane: only },
                );
            } else {
                b.open(
                    label,
                    format!("play face-down — {}", rank.power_name()),
                    sub,
                );
            }
        }
    }

    // --- ON THE BOARD: one row per card of yours that can do something. ----------------
    for lane in 0..state.lane_count() {
        let mut lane_started = false;
        for slot in 0..state.lanes[lane].side(me).len() {
            let mine: Vec<Action> = legal
                .iter()
                .copied()
                .filter(|a| acts_from(*a, lane, slot))
                .collect();
            if mine.is_empty() {
                continue;
            }
            if !lane_started {
                b.heading(format!("LANE {} — your cards", lane_label(lane)));
                lane_started = true;
            }
            let token = state
                .at(lane, me, slot)
                .map(|c| card_token(state, c, observer))
                .unwrap_or_default();
            let label = format!("#{} {}", slot_label(slot), token);

            let sub = card_menu(state, observer, lane, slot, &label, &mine);
            if let [only] = mine[..] {
                // The one-move case is the common one: a face-down card can only be
                // flipped. The row has to carry the whole consequence, because picking it
                // does it — there is no second menu to reconsider in.
                b.take(label.clone(), sub.rows[0].summary.clone(), only);
            } else {
                b.open(label.clone(), verb_summary(&mine), sub);
            }
        }
    }

    // --- Ending the turn is not a card, so it gets its own heading. --------------------
    if legal.contains(&Action::Pass) {
        b.heading("END THE TURN");
        b.take(
            "pass",
            format!(
                "forfeit the rest of this turn ({} action(s) unused)",
                state.actions_remaining
            ),
            Action::Pass,
        );
    }
    b.done()
}

/// Does `action` belong to the card at `lane`/`slot` on the acting player's side?
///
/// A `DeclarePair` belongs to *both* of its members, so it is offered from either card.
/// Picking it from one or the other yields the same action, since the engine's slots are
/// ordered.
fn acts_from(action: Action, lane: usize, slot: usize) -> bool {
    let (l, s) = (lane as u8, slot as u8);
    match action {
        Action::Flip { lane, slot } => (lane, slot) == (l, s),
        Action::Attack { lane, attacker, .. } => (lane, attacker) == (l, s),
        Action::DeclarePair { lane, slot_a, slot_b } => lane == l && (slot_a == s || slot_b == s),
        _ => false,
    }
}

/// The second menu for a card in hand: which lane.
fn lane_menu(state: &GameState, rank: Rank, lanes: &[u8]) -> Menu {
    let me = state.acting_player();
    let mut b = Builder::new(format!("Play the {rank} face-down into which lane?")).hint(
        format!("{}: {}", rank.power_name(), rank.power_text()),
    );
    for &lane in lanes {
        let l = lane as usize;
        b.take(
            format!("lane {}", lane_label(lane)),
            format!(
                "you have {} card(s) there, opponent {}",
                state.lanes[l].side(me).len(),
                state.lanes[l].side(me.other()).len()
            ),
            Action::Play { rank, lane },
        );
    }
    b.done()
}

/// The second menu for a card on the board: what it does.
///
/// Flip, attack and pair are listed together rather than behind a verb menu. They are never
/// all available at once — flipping needs the card face-down, attacking and pairing need it
/// face-up — so the flat list is short, and it lets the attack rows carry their target and
/// their combat notes on one line.
fn card_menu(
    state: &GameState,
    observer: Observer,
    lane: usize,
    slot: usize,
    subject: &str,
    actions: &[Action],
) -> Menu {
    let me = state.acting_player();
    let them = me.other();
    let mut b = Builder::new(format!(
        "Your {subject} in lane {} — do what?",
        lane_label(lane)
    ));
    if let Some(card) = state.at(lane, me, slot) {
        if knows(card, observer) {
            b = b.hint(format!(
                "{} — {}: {}",
                card.rank,
                card.rank.power_name(),
                card.rank.power_text()
            ));
        }
    }

    for &action in actions {
        match action {
            Action::Flip { .. } => {
                // Flipping is always a card's only legal move — attacking and pairing both
                // need it face-up — so this row is what the card list shows, and the power
                // has to be named there without pushing the row past the board's width.
                // The full text is one `powers` away.
                let (note, summary) = match state.at(lane, me, slot) {
                    Some(c) if knows(c, observer) => (
                        format!(
                            "reveals {} — {}: {}",
                            c.rank,
                            c.rank.power_name(),
                            c.rank.power_text()
                        ),
                        format!("flip face-up — reveals {} ({})", c.rank, c.rank.power_name()),
                    ),
                    // A base card is hidden from its owner too (`game_rules.md` §3), so
                    // this is a genuine gamble and the menu says so.
                    _ => (
                        "a base card — you do not know what this is either".to_string(),
                        "flip face-up — a base card, unknown even to you".to_string(),
                    ),
                };
                b.take_summarised("flip it face-up", note, summary, action);
            }

            Action::Attack { target, .. } => {
                let token = state
                    .at(lane, them, target as usize)
                    .map(|c| card_token(state, c, observer))
                    .unwrap_or_else(|| "<gone>".to_string());
                let mut notes = Vec::new();
                if let Some(partner) = state.pair_partner(lane, me, slot) {
                    notes.push(format!(
                        "PAIR with your #{}: one action, 2 damage",
                        slot_label(partner)
                    ));
                }
                notes.extend(combat_notes(state, lane, slot, target as usize));
                b.take(
                    format!("attack opp #{} {token}", slot_label(target)),
                    notes.join("; "),
                    action,
                );
            }

            Action::DeclarePair { slot_a, slot_b, .. } => {
                let other = if slot_a as usize == slot { slot_b } else { slot_a };
                let token = state
                    .at(lane, me, other as usize)
                    .map(|c| card_token(state, c, observer))
                    .unwrap_or_default();
                b.take_summarised(
                    format!("pair with your #{} {token}", slot_label(other)),
                    "2 damage for one action; they can never attack separately again",
                    format!(
                        "pair with your #{} {token} — they can never attack separately again",
                        slot_label(other)
                    ),
                    action,
                );
            }

            // `acts_from` admits nothing else.
            _ => {}
        }
    }
    b.done()
}

/// The one-line summary of a card that has several moves, for the first menu.
fn verb_summary(actions: &[Action]) -> String {
    let attacks = actions
        .iter()
        .filter(|a| matches!(a, Action::Attack { .. }))
        .count();
    let pairs = actions
        .iter()
        .filter(|a| matches!(a, Action::DeclarePair { .. }))
        .count();
    let mut parts = Vec::new();
    if attacks > 0 {
        parts.push(format!(
            "{attacks} target{}",
            if attacks == 1 { "" } else { "s" }
        ));
    }
    if pairs > 0 {
        parts.push(format!(
            "{pairs} pairing{}",
            if pairs == 1 { "" } else { "s" }
        ));
    }
    parts.join(" · ")
}

// ===================================================================== sub-decisions ==
//
// A sub-decision *is* a choice of card, so there is no second level to build: these are one
// grouped list each. They stay short — the widest is the 4's Foresight at one row per
// face-down card on the board — which is why the flat list that was wrong for the main
// phase is right here.

fn build_foresight(state: &GameState, legal: &[Action], observer: Observer) -> Menu {
    let me = state.acting_player();
    // Sort so the two sides are contiguous and can share a heading; the engine enumerates
    // lane by lane, which interleaves them.
    let mut peeks: Vec<(bool, u8, u8, Action)> = legal
        .iter()
        .filter_map(|a| match a {
            Action::Peek { side, lane, slot } => {
                Some((*side == Side::Theirs, *lane, *slot, *a))
            }
            _ => None,
        })
        .collect();
    peeks.sort_unstable_by_key(|(theirs, lane, slot, _)| (*theirs, *lane, *slot));

    let mut b = Builder::new("Foresight — which face-down card do you look at?").hint(
        "Only you learn it, and you keep knowing it. Base cards count, yours included.",
    );
    let mut current: Option<bool> = None;
    for (theirs, lane, slot, action) in peeks {
        if current != Some(theirs) {
            b.heading(if theirs {
                "THE OPPONENT'S SIDE"
            } else {
                "YOUR SIDE"
            });
            current = Some(theirs);
        }
        let owner = if theirs { me.other() } else { me };
        let token = state
            .at(lane as usize, owner, slot as usize)
            .map(|c| card_token(state, c, observer))
            .unwrap_or_default();
        b.take(
            format!("lane {} #{} {token}", lane_label(lane), slot_label(slot)),
            String::new(),
            action,
        );
    }
    b.done()
}

fn build_resolve_order(
    state: &GameState,
    legal: &[Action],
    observer: Observer,
    kind: ResolveKind,
    lane: u8,
    remaining: usize,
) -> Menu {
    let me = state.acting_player();
    let mut b = Builder::new(format!(
        "{} in lane {} — which card resolves next? ({remaining} left)",
        kind.label(),
        lane_label(lane)
    ))
    .hint("Each power resolves fully before you choose the next one (§8).");
    for &action in legal {
        let Action::ResolveNext { lane, slot } = action else {
            continue;
        };
        let card = state.at(lane as usize, me, slot as usize);
        let token = card
            .map(|c| card_token(state, c, observer))
            .unwrap_or_default();
        let note = match card {
            Some(c) if knows(c, observer) => {
                format!("{}: {}", c.rank.power_name(), c.rank.power_text())
            }
            _ => String::new(),
        };
        b.take(format!("#{} {token}", slot_label(slot)), note, action);
    }
    b.done()
}

fn build_queen_source(
    state: &GameState,
    legal: &[Action],
    observer: Observer,
    queen_lane: u8,
) -> Menu {
    let me = state.acting_player();
    let mut b = Builder::new(format!(
        "Queen — pull which allied card into lane {}?",
        lane_label(queen_lane)
    ))
    .hint("It keeps its damage and its freeze, stops being a base card, and does not refire.");
    let mut current: Option<u8> = None;
    for &action in legal {
        let Action::MoveHere { lane, slot } = action else {
            continue;
        };
        if current != Some(lane) {
            b.heading(format!("LANE {}", lane_label(lane)));
            current = Some(lane);
        }
        let token = state
            .at(lane as usize, me, slot as usize)
            .map(|c| card_token(state, c, observer))
            .unwrap_or_default();
        b.take(format!("#{} {token}", slot_label(slot)), String::new(), action);
    }
    b.done()
}

fn build_give_back(state: &GameState, legal: &[Action]) -> Menu {
    let me = state.acting_player();
    let (prompt, hint) = match state.config.two_power {
        crate::config::TwoPower::Bottom => (
            "View — which card goes on the bottom of your draw pile?",
            "Private to you. You may give back the card you just drew.",
        ),
        crate::config::TwoPower::Discard => (
            "View — which card do you discard?",
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
        b.take(label, rank.power_name(), action);
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
    // The attacker is still on the board — no damage has landed yet — so its slot is what
    // the combat notes need to know whether a 9 or an 8 changes this half of the split.
    let attacker_slot = attacker
        .and_then(|id| state.locate(id))
        .map(|(_, _, slot)| slot)
        .unwrap_or(usize::MAX);

    let mut b = Builder::new("Twinstrike — which card takes the second point of damage?")
        .hint("No damage has landed yet; both halves land together.");
    b.heading(format!("OPPONENT, LANE {}", lane_label(lane)));
    for &action in legal {
        let Action::SplitTarget { slot } = action else {
            continue;
        };
        let token = state
            .at(lane as usize, them, slot as usize)
            .map(|c| card_token(state, c, observer))
            .unwrap_or_default();
        b.take(
            format!("opp #{} {token}", slot_label(slot)),
            combat_notes(state, lane as usize, attacker_slot, slot as usize).join("; "),
            action,
        );
    }
    b.done()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::Agent;
    use crate::config::GameConfig;
    use crate::player::Player;
    use crate::testkit::Position;

    /// Every leaf of the tree, in order.
    fn leaves(menu: &Menu) -> Vec<Action> {
        let mut out = Vec::new();
        for pick in &menu.picks {
            match pick {
                Pick::Take(a) => out.push(*a),
                Pick::Open(sub) => out.extend(leaves(sub)),
            }
        }
        out
    }

    /// The whole tree as printed, every level of it — what a player could see by walking
    /// every branch. `Menu::render` only prints the level it is called on.
    fn render_all(menu: &Menu) -> String {
        let mut out = menu.render();
        for pick in &menu.picks {
            if let Pick::Open(sub) = pick {
                out.push_str(&render_all(sub));
            }
        }
        out
    }

    /// The tree is a reshaping of the engine's list, not a filter on it. Every legal action
    /// must be reachable, or the menu has quietly made a move impossible.
    ///
    /// `DeclarePair` is the one action that appears twice — once under each of its two
    /// members — so the comparison is by set, not by count.
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
                        menu.render()
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

    /// The top level is one row per card, so it must be shorter than the flat list it
    /// replaces in any position where the flat list was the problem.
    #[test]
    fn top_level_is_one_row_per_card() {
        let mut p = Position::new(GameConfig::split_deck());
        // Two attackers against three targets: nine flat actions, five rows.
        p.face_up(0, Player::P0, Rank::SEVEN);
        p.face_up(0, Player::P0, Rank::SEVEN);
        p.face_up(0, Player::P1, Rank::FOUR);
        p.face_up(0, Player::P1, Rank::EIGHT);
        p.face_up(0, Player::P1, Rank::SIX);
        let state = p.build();

        let legal = state.legal_actions();
        let menu = build(&state, &legal, Some(Player::P0));
        assert!(
            menu.len() < legal.len(),
            "the tree must be shorter than the flat list\n{}",
            menu.render()
        );
        // Two of P0's cards can act, plus `pass`.
        assert_eq!(menu.len(), 3, "{}", menu.render());

        // The second level is that one card's own moves: three targets and the pairing,
        // each on one line with its combat notes.
        let Pick::Open(sub) = &menu.picks[0] else {
            panic!("a card with four moves must open a second menu\n{}", menu.render());
        };
        let text = sub.render();
        println!("{text}");
        assert_eq!(sub.len(), 4, "{text}");
        assert!(text.contains("attack opp #1 [4]"), "{text}");
        assert!(text.contains("attack opp #3 [6]"), "{text}");
        assert!(text.contains("pair with your #2 [7]"), "{text}");
        // The 8's retaliation is public and changes the trade, so it has to be on the line.
        assert!(text.contains("8 retaliates for 1"), "{text}");
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
        for token in ["(K)", "[K]", "(Q)", "[Q]"] {
            assert!(
                !text.contains(token),
                "menu leaked a hidden {token}\n{text}"
            );
        }
        // The actor's own hand is theirs to see, so the 3 must still be named.
        assert!(text.contains(" 3 "), "P0's own hand is missing\n{text}");
    }
}
