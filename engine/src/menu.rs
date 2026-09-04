//! The interactive prompt's action tree.
//!
//! [`GameState::legal_actions`](crate::GameState::legal_actions) returns a flat list, which
//! is the right shape for an agent and the wrong shape for a person: in a live midgame it is
//! sixty-odd lines, most of them the attacker × target cross-product of one lane. This
//! module reshapes that same list into a tree that asks one question at a time — *which
//! verb, then which card and which lane* — the order the game is actually played in.
//!
//! # Which card, then which lane — for the verbs that act on the board
//!
//! FLIP and ATTACK ask for a **card** first and a lane only if they have to, because that is
//! how the move is decided: a player wants to flip *the 7*, and only then has to care that
//! there are two of them. So the first question lists ranks with their multiplicity, and the
//! second question — *from which lane* — is asked only when the copies are spread across more
//! than one lane. Two copies in one lane that differ in nothing but their
//! [`CardId`](crate::card::CardId) are one move under two names, so there is nothing to ask
//! and the first is taken; see [`same_move`].
//!
//! PLAY is unchanged — a card in hand has no lane yet, so *which card, which lane* was always
//! its shape — and so is PAIR, which is a choice of two cards inside one lane.
//!
//! # Every number means the same thing every time
//!
//! The tree is built so that what a player types is a property of the game, not of the list:
//!
//! - the four §4 verbs are always in the same order at the same numbers, whether or not they
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
//! The card question is the one list that holds only what is there: a rank is not a fixed
//! coordinate the way a lane is, and thirteen rows of `—` to reach the two ranks you own is a
//! worse trade than reading a short list. It is ordered by rank, with the cards the actor
//! cannot name last, so it is at least the same order every turn.
//!
//! # A number points at the board before it is committed to
//!
//! Numbering a card by its position in its column is only half an answer when three of that
//! column's cards render identically — `(? ²♥)` three times over is exactly the position a
//! player most needs to be sure about. [`Menu::focus`] closes that: it turns a number into
//! the set of cards the row names, which the CLI paints red on the board while the number is
//! being typed, alongside [`Menu::render_with`] painting the row itself. So the number is
//! checked against the board *before* Enter, rather than regretted after it.
//!
//! The focus is derived from the actions under a row rather than stored beside it, so a menu
//! cannot point at a card it does not offer. A row lights only the part of the move it
//! actually settles ([`Reach`]): the card you would act **with**, never the list of things
//! you might aim it **at**, because that list is the next question and reddening all of it
//! says nothing about the choice in front of you. So `ATTACK` lights your attackers,
//! choosing one keeps just that one lit, and the enemy card reddens on the page that asks
//! which enemy card — walking down the tree only ever narrows what is lit.
//!
//! Lanes are not lit alongside cards. A `PLAY` lights the lane it would go into, because
//! there the lane *is* the move and no card is on the board yet; everywhere else the card is
//! the answer, and reddening its column heading as well only widens what the eye has to
//! check.
//!
//! It leaks nothing: it names cards by [`CardId`](crate::card::CardId) and the renderer can
//! only colour tokens it was already drawing.
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
    card_number, card_token, column_slots, combat_notes, knows, lane_label, Focus, Observer,
    HIGHLIGHT, RESET,
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

    /// Everything on the board that typing `number` settles.
    ///
    /// `number` is what the player types, so `0` is `BACK` and names nothing. A row that
    /// leads to another question lights only the part it decides — the cards you could act
    /// *with*, never the list of things you might aim at, which is the next question's job
    /// (see [`Reach`]). So `ATTACK` lights your attackers, choosing one keeps just that one
    /// lit, and the enemy card reddens on the page that asks which enemy card.
    ///
    /// Derived from the actions themselves rather than stored alongside the rows, so a menu
    /// cannot point at a card it does not actually offer.
    pub fn focus(&self, state: &GameState, number: usize) -> Focus {
        let mut focus = Focus::none();
        if let Some(pick) = number.checked_sub(1).and_then(|i| self.picks.get(i)) {
            collect_focus(state, pick, Reach::Whole, &mut focus);
        }
        focus
    }

    /// The menu as printed: the question, then the numbered rows under their headings.
    ///
    /// `nested` adds the row that goes back up a level. It lives here rather than at the
    /// call site so that it lines up with everything above it.
    pub fn render(&self, nested: bool) -> String {
        self.render_with(nested, None)
    }

    /// [`Menu::render`], with the row the player is part-way through typing shown in red —
    /// the same red the matching cards are wearing on the board above.
    ///
    /// `selected` is the number typed, so `Some(0)` picks out `BACK`. A number with no row
    /// behind it highlights nothing, which is what says "that is not one of these".
    pub fn render_with(&self, nested: bool, selected: Option<usize>) -> String {
        let paint = |number: usize, line: String| -> String {
            match selected == Some(number) {
                true => format!("{HIGHLIGHT}{line}{RESET}"),
                false => line,
            }
        };
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
            let line = if row.note.is_empty() {
                head
            } else {
                format!("{head}   {}", row.note)
            };
            out.push_str(&format!("{}\n", paint(i + 1, line)));
        }
        if nested {
            let back = format!("   {:<width$} {:>2}", "BACK", "#0");
            out.push_str(&format!("\n{}\n", paint(0, back)));
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

// =========================================================================== focus ==

/// How much of an action a row has actually settled.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reach {
    /// Picking this row *is* the move. Everything it names is decided, so everything it
    /// names lights up.
    Whole,
    /// This row is one question of several, and the rest of the move is still open. Only
    /// the card being acted **with** lights up; what it might be aimed at is a list the
    /// player has not been shown yet, and reddening all of it says nothing about the choice
    /// in front of them.
    Actor,
}

/// Every card named under one pick, at the reach that pick has settled.
///
/// A row that opens another question drops to [`Reach::Actor`] and stays there however deep
/// the recursion goes, so `ATTACK` lights up the cards that could attack, choosing one keeps
/// only that one lit, and the enemy card lights up on the page that asks for it.
fn collect_focus(state: &GameState, pick: &Pick, reach: Reach, focus: &mut Focus) {
    match pick {
        Pick::Take(action) => focus_action(state, *action, reach, focus),
        Pick::Open(sub) => {
            for pick in &sub.picks {
                collect_focus(state, pick, Reach::Actor, focus);
            }
        }
        Pick::Unavailable => {}
    }
}

/// The cards one action puts its hands on, filtered by what `reach` has settled.
///
/// The split is between the card a move is made **with** and the card it is aimed **at**.
/// The first is settled the moment its row is picked; the second is a question of its own,
/// and is lit only on the page that asks it. [`Action::Play`] is neither — it names no card
/// on the board yet, so it contributes the lane it would go into, which is the whole of
/// what the move decides and is decided only at the step that asks for it.
fn focus_action(state: &GameState, action: Action, reach: Reach, focus: &mut Focus) {
    let me = state.acting_player();
    let them = me.other();
    let settled = reach == Reach::Whole;
    match action {
        Action::Play { lane, .. } => {
            if settled {
                focus.lane(lane as usize)
            }
        }
        Action::Flip { lane, slot }
        | Action::ResolveNext { lane, slot }
        | Action::MoveHere { lane, slot } => focus.at(state, lane as usize, me, slot as usize),
        Action::Attack {
            lane,
            attacker,
            target,
        } => {
            focus.at(state, lane as usize, me, attacker as usize);
            if settled {
                focus.at(state, lane as usize, them, target as usize);
            }
        }
        // Both halves of a pair are the actor's own, and a pair is symmetric — there is no
        // "aimed at" to hold back, so choosing one card lights the partners it could take.
        Action::DeclarePair {
            lane,
            slot_a,
            slot_b,
        } => {
            focus.at(state, lane as usize, me, slot_a as usize);
            focus.at(state, lane as usize, me, slot_b as usize);
        }
        // A peek is all target and no actor: the 4 is already spent, and the card being
        // looked at is what the row picks. The side is in the action.
        Action::Peek { side, lane, slot } => {
            if settled {
                let owner = match side {
                    Side::Mine => me,
                    Side::Theirs => them,
                };
                focus.at(state, lane as usize, owner, slot as usize);
            }
        }
        // The lane is not in the action — the second half of a twinstrike happens wherever
        // the first half did, which only the pending decision knows.
        Action::SplitTarget { slot } => {
            if let Some(Pending::SplitTarget { lane, attackers, .. }) = state.pending.last() {
                for &id in attackers {
                    focus.card(id);
                }
                if settled {
                    focus.at(state, *lane as usize, them, slot as usize);
                }
            }
        }
        // A card in hand is not on the board, so there is nothing to point at.
        Action::GiveBack { .. } => {}
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
        Action::GiveBack { .. } | Action::SplitTarget { .. } => None,
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

/// The pair a card belongs to, named by the number its partner has on the board.
fn pair_note(state: &GameState, lane: usize, slot: usize, observer: Observer) -> String {
    let me = state.acting_player();
    match state.pair_partner(lane, me, slot) {
        Some(partner) => format!(
            "pair with #{}",
            card_number(state, lane, me, partner, observer)
        ),
        None => String::new(),
    }
}

// ================================================================== card-first menus ==

/// One rank's worth of the acting player's cards that a verb can use.
///
/// The unit the first question offers. A rank with a single copy on the board is a complete
/// answer on its own; anything else carries the lanes its copies are sitting in, which is
/// what the second question picks between.
struct CardGroup {
    /// The rank as the acting player is entitled to see it. `None` is one of their own base
    /// cards, hidden from them too (`game_rules.md` §3), so `?` is the most a menu may say.
    rank: Option<Rank>,
    /// The candidates, lane by lane in lane order, slots in the order the board draws them.
    lanes: Vec<(usize, Vec<usize>)>,
}

impl CardGroup {
    fn count(&self) -> usize {
        self.lanes.iter().map(|(_, slots)| slots.len()).sum()
    }

    /// `7`, `7 ×2`, `?`, `? ×3` — the rank, and how many of it there are to use.
    fn label(&self) -> String {
        let rank = match self.rank {
            Some(rank) => rank.to_string(),
            None => "?".to_string(),
        };
        match self.count() {
            1 => rank,
            n => format!("{rank} ×{n}"),
        }
    }

    /// How a prompt refers to the group in a sentence.
    fn phrase(&self) -> String {
        match self.rank {
            Some(rank) => format!("the {rank}"),
            None => "a face-down base card".to_string(),
        }
    }

    fn power(&self) -> &'static str {
        match self.rank {
            Some(rank) => rank.power_name(),
            None => "",
        }
    }
}

/// Group the acting player's own cards named by `actions` by the rank they can see.
///
/// `owned` pulls the *actor's* card out of an action — the card being flipped, or the
/// attacker. Ranks the actor cannot see collapse into one `?` group, because that is all
/// they are entitled to be told about them.
fn group_by_card(
    state: &GameState,
    observer: Observer,
    actions: &[Action],
    owned: impl Fn(&Action) -> Option<(usize, usize)>,
) -> Vec<CardGroup> {
    let me = state.acting_player();
    let mut wanted: Vec<(usize, usize)> = Vec::new();
    for action in actions {
        if let Some(pos) = owned(action) {
            if !wanted.contains(&pos) {
                wanted.push(pos);
            }
        }
    }

    let mut groups: Vec<CardGroup> = Vec::new();
    for lane in 0..state.lane_count() {
        for slot in column_slots(state, lane, me, observer) {
            if !wanted.contains(&(lane, slot)) {
                continue;
            }
            let card = &state.lanes[lane].side(me)[slot];
            let rank = knows(card, observer).then_some(card.rank);
            let index = match groups.iter().position(|g| g.rank == rank) {
                Some(index) => index,
                None => {
                    groups.push(CardGroup {
                        rank,
                        lanes: Vec::new(),
                    });
                    groups.len() - 1
                }
            };
            match groups[index].lanes.iter_mut().find(|(l, _)| *l == lane) {
                Some((_, slots)) => slots.push(slot),
                None => groups[index].lanes.push((lane, vec![slot])),
            }
        }
    }
    // Rank order, with the cards the actor cannot name last: the list is short, and this way
    // it is at least in the same order every turn.
    groups.sort_by_key(|g| (g.rank.is_none(), g.rank));
    groups
}

/// What makes a card-first menu a FLIP or an ATTACK: the wording, and what a settled card
/// does.
struct CardFirst<'a> {
    /// The card question.
    prompt: &'a str,
    /// The lane question for one group, given [`CardGroup::phrase`].
    lane_prompt: &'a dyn Fn(&str) -> String,
    /// The which-card question inside one lane — only ever reached by a group whose copies
    /// there are *not* the same move.
    tie_prompt: &'a dyn Fn(usize) -> String,
    /// What picking one settled card does: take the action, or ask the next question.
    leaf: &'a dyn Fn(usize, usize) -> Pick,
}

/// Ask for a card, and for a lane only when the card does not already name one.
fn card_first_menu(
    state: &GameState,
    observer: Observer,
    groups: &[CardGroup],
    cf: &CardFirst,
) -> Menu {
    let mut b = Builder::new(cf.prompt);
    b.heading("IN PLAY");
    for group in groups {
        let note = format!(
            "{:<6} {:<11} {}",
            group.label(),
            group.power(),
            group
                .lanes
                .iter()
                .map(|(lane, slots)| format!(
                    "lane {} {}",
                    lane_label(*lane),
                    copies_note(state, observer, *lane, slots)
                ))
                .collect::<Vec<_>>()
                .join("   ")
        );
        // One lane is no choice at all — the note above has already said which one it is.
        let pick = if let [(lane, slots)] = &group.lanes[..] {
            settle_in_lane(state, observer, *lane, slots, cf)
        } else {
            let mut lb = Builder::new((cf.lane_prompt)(&group.phrase()));
            for lane in 0..state.lane_count() {
                let name = format!("LANE {}", lane_label(lane));
                match group.lanes.iter().find(|(l, _)| *l == lane) {
                    Some((l, slots)) => lb.push(
                        name,
                        copies_note(state, observer, *l, slots),
                        settle_in_lane(state, observer, *l, slots, cf),
                    ),
                    None => lb.push(name, String::new(), Pick::Unavailable),
                }
            }
            Pick::Open(Box::new(lb.done()))
        };
        b.push("CARD", note, pick);
    }
    b.done()
}

/// The copies of one group that sit in one lane: their tokens, and whether they are in a
/// pair — which is the one thing about an attacker that changes the move without changing
/// the card's rank or its lane.
fn copies_note(state: &GameState, observer: Observer, lane: usize, slots: &[usize]) -> String {
    let me = state.acting_player();
    let tokens: Vec<String> = slots
        .iter()
        .map(|&slot| card_token(&state.lanes[lane].side(me)[slot], observer))
        .collect();
    let tokens = tokens.join(" ");
    if slots
        .iter()
        .any(|&slot| state.pair_partner(lane, me, slot).is_some())
    {
        format!("{tokens} paired")
    } else {
        tokens
    }
}

/// One lane's worth of a group.
///
/// Copies in one lane that are [`same_move`] are one move under two names, so there is
/// nothing to ask: take the first. Anything else — a paired copy beside a loose one, a
/// damaged copy beside a fresh one, an Ace with two attacks left beside one with one — is a
/// real choice and is still put to the player.
fn settle_in_lane(
    state: &GameState,
    observer: Observer,
    lane: usize,
    slots: &[usize],
    cf: &CardFirst,
) -> Pick {
    let me = state.acting_player();
    if same_move(state, lane, me, slots) {
        return (cf.leaf)(lane, slots[0]);
    }
    Pick::Open(Box::new(card_menu(
        state,
        observer,
        (cf.tie_prompt)(lane),
        lane,
        me,
        "CARD",
        |slot| {
            if !slots.contains(&slot) {
                return (String::new(), None);
            }
            (
                pair_note(state, lane, slot, observer),
                Some((cf.leaf)(lane, slot)),
            )
        },
    )))
}

/// Do these cards of `owner`'s differ in nothing but their identity?
///
/// The comparison is the whole [`Card`](crate::card::Card) with its
/// [`CardId`](crate::card::CardId) equalized — rank, damage, freeze, attack budget, pair, and
/// who knows the rank — rather than a list of the fields that seemed to matter. A case this
/// gets wrong is therefore a field nobody added to `Card`, not a case nobody thought of.
///
/// Note that two of the actor's own **base cards** in one lane are never the same move: the
/// actor cannot tell them apart, but they have different ranks and the engine can, so the
/// menu asks rather than choosing for them.
fn same_move(state: &GameState, lane: usize, owner: Player, slots: &[usize]) -> bool {
    let side = state.lanes[lane].side(owner);
    slots.windows(2).all(|pair| {
        let mut a = side[pair[0]].clone();
        a.id = side[pair[1]].id;
        a == side[pair[1]]
    })
}

// ====================================================================== the main phase ==

/// The four §4 actions, always in this order at these numbers.
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
    // No fifth row. §4 makes acting mandatory and the engine ends a turn with nothing in
    // it, so there is never anything to offer here but the four verbs.
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

/// FLIP: which card, then — only if its copies are in more than one lane — which lane.
fn flip_menu(state: &GameState, flips: &[Action], observer: Observer) -> Menu {
    let groups = group_by_card(state, observer, flips, |a| match a {
        Action::Flip { lane, slot } => Some((*lane as usize, *slot as usize)),
        _ => None,
    });
    card_first_menu(
        state,
        observer,
        &groups,
        &CardFirst {
            prompt: "FLIP — which card?",
            lane_prompt: &|phrase| format!("FLIP {phrase} — from which lane?"),
            tie_prompt: &|lane| format!("FLIP in lane {} — which card?", lane_label(lane)),
            leaf: &|lane, slot| {
                Pick::Take(Action::Flip {
                    lane: lane as u8,
                    slot: slot as u8,
                })
            },
        },
    )
}

/// ATTACK: which of your cards, then — only if its copies are in more than one lane — which
/// lane, then which of theirs.
fn attack_menu(state: &GameState, attacks: &[Action], observer: Observer) -> Menu {
    let groups = group_by_card(state, observer, attacks, |a| match a {
        Action::Attack { lane, attacker, .. } => Some((*lane as usize, *attacker as usize)),
        _ => None,
    });
    card_first_menu(
        state,
        observer,
        &groups,
        &CardFirst {
            prompt: "ATTACK — using which card?",
            lane_prompt: &|phrase| format!("ATTACK using {phrase} — from which lane?"),
            tie_prompt: &|lane| {
                format!("ATTACK from lane {} — using which card?", lane_label(lane))
            },
            leaf: &|lane, slot| {
                let mine: Vec<Action> = attacks
                    .iter()
                    .copied()
                    .filter(|a| {
                        matches!(a, Action::Attack { lane: l, attacker, .. }
                                 if *l as usize == lane && *attacker as usize == slot)
                    })
                    .collect();
                Pick::Open(Box::new(target_menu(state, observer, lane, slot, &mine)))
            },
        },
    )
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
        // One 10, in one lane, so the card question is the whole of it — no lane to ask for.
        let Pick::Open(targets) = &attack.picks[0] else {
            panic!("the 10 must be able to attack\n{}", attack.render(false));
        };

        let text = targets.render(false);
        assert_eq!(targets.len(), 3, "every enemy card in the lane is listed\n{text}");
        assert!(matches!(targets.picks[0], Pick::Unavailable), "base card\n{text}");
        assert!(matches!(targets.picks[1], Pick::Unavailable), "the 8, taunted\n{text}");
        assert!(matches!(targets.picks[2], Pick::Take(_)), "the Jack\n{text}");
        assert!(targets.rows[2].note.starts_with("[J "), "{text}");
    }

    /// The case the highlight exists for: three enemy cards in a lane, all face-down, all
    /// rendering as the same `(? ²♥)`. Nothing in the target list tells them apart, so the
    /// number typed has to point at one of them on the board.
    ///
    /// Each number focuses **its own** card and no other, plus the attacker — an attack is
    /// a matchup, and lighting up one half of it answers half the question.
    #[test]
    fn a_target_number_points_at_that_card_and_the_attacker() {
        let mut p = Position::new(GameConfig::split_deck());
        let attacker = p.face_up(0, Player::P0, Rank::TEN);
        p.base(0, Player::P1, Rank::ACE);
        p.face_down(0, Player::P1, Rank::SEVEN);
        p.face_down(0, Player::P1, Rank::NINE);
        // A second lane with a card in it, so a focus that strayed out of lane 1 would show.
        p.face_down(1, Player::P1, Rank::FOUR);
        let state = p.build();

        let menu = build(&state, &state.legal_actions(), Some(Player::P0));
        let Pick::Open(attack) = &menu.picks[2] else {
            panic!("attacking must be on\n{}", menu.render(false));
        };
        // One 10, in one lane, so the card question is a complete answer and picking it
        // lands straight on the target list.
        let Pick::Open(targets) = &attack.picks[0] else {
            panic!("the 10 must be able to attack\n{}", attack.render(false));
        };
        assert_eq!(targets.len(), 3, "every enemy card in lane 1 is listed");

        let mine = |slot: usize| {
            let mut f = Focus::none();
            f.at(&state, 0, Player::P0, attacker);
            f.at(&state, 0, Player::P1, slot);
            f
        };
        // The base card is at slot 0 and untouchable while the base is locked (§3), so it
        // is a `—` and names nothing; the two played cards each name themselves.
        assert_eq!(targets.focus(&state, 1), Focus::none(), "a `—` row is not a card");
        assert_eq!(targets.focus(&state, 2), mine(1));
        assert_eq!(targets.focus(&state, 3), mine(2));
        // Typing a number no row has is not a highlight, which is what says "not one of
        // these" rather than pointing at something arbitrary.
        assert_eq!(targets.focus(&state, 0), Focus::none(), "0 is BACK");
        assert_eq!(targets.focus(&state, 9), Focus::none(), "out of range");

        // Choosing what to attack *with* is a different question from choosing what to
        // attack. Reddening every enemy card the 10 could reach, while the player is still
        // being asked which of their own cards to use, says nothing about that choice and
        // costs the target highlight the thing that makes it worth having: being one card.
        let picking_the_attacker = attack.focus(&state, 1);
        assert_eq!(
            picking_the_attacker.cards(),
            &[crate::testkit::card_at(&state, 0, Player::P0, attacker).id],
            "the attacker alone, no targets yet"
        );
        // And the same one card again from the top, where the verb is all that is settled.
        assert_eq!(menu.focus(&state, 3), picking_the_attacker);
    }

    /// A lane goes red only where the lane *is* the move: a `PLAY` has no card on the board
    /// to point at, so the lane it would go into is the whole of what the step decides.
    ///
    /// Everywhere else the card is the answer. Reddening its column heading too would widen
    /// what the eye has to check, in the one situation — several identical face-down cards
    /// in one column — where narrowing it is the entire point.
    #[test]
    fn only_a_play_lights_a_lane() {
        let mut p = Position::new(GameConfig::split_deck());
        p.hand(Player::P0, &[Rank::NINE]);
        p.face_up(0, Player::P0, Rank::TEN);
        p.face_down(1, Player::P0, Rank::SIX);
        p.face_down(0, Player::P1, Rank::SEVEN);
        let state = p.build();
        let menu = build(&state, &state.legal_actions(), Some(Player::P0));

        // PLAY: no lane until the step that asks for one, and then exactly that lane.
        let Pick::Open(play) = &menu.picks[0] else {
            panic!("playing must be on\n{}", menu.render(false));
        };
        assert!(menu.focus(&state, 1).is_empty(), "the verb decides no lane");
        let Pick::Open(play_lanes) = &play.picks[0] else {
            panic!("the 9 must ask which lane\n{}", play.render(false));
        };
        for lane in 0..state.lane_count() {
            let focus = play_lanes.focus(&state, lane + 1);
            assert_eq!(focus.lanes(), &[lane], "LANE {} lights lane {lane}", lane + 1);
            assert!(focus.cards().is_empty(), "a card in hand is not on the board");
        }

        // Every other verb points at cards and never at a lane, however deep it is walked.
        for number in [2, 3] {
            let mut seen = vec![menu.focus(&state, number)];
            if let Pick::Open(sub) = &menu.picks[number - 1] {
                seen.extend((1..=sub.len()).map(|n| sub.focus(&state, n)));
            }
            for focus in seen {
                assert!(
                    focus.lanes().is_empty(),
                    "#{number} lit lane(s) {:?}",
                    focus.lanes()
                );
            }
        }
    }

    /// A row that leads to another question stands for everything under it, so `ATTACK`
    /// lights up every card that could attack and everything they could attack — and each
    /// step down the tree narrows that to a subset.
    #[test]
    fn walking_the_tree_only_ever_narrows_the_highlight() {
        let mut p = Position::new(GameConfig::split_deck());
        p.face_up(0, Player::P0, Rank::TEN);
        p.face_up(1, Player::P0, Rank::TEN);
        p.face_down(0, Player::P1, Rank::SEVEN);
        p.face_down(1, Player::P1, Rank::NINE);
        let state = p.build();

        let menu = build(&state, &state.legal_actions(), Some(Player::P0));
        let whole_verb = menu.focus(&state, 3);
        assert!(!whole_verb.is_empty(), "ATTACK must point somewhere");

        // Two 10s in two lanes, so the card question is followed by a lane question.
        let Pick::Open(attack) = &menu.picks[2] else {
            panic!("attacking must be on");
        };
        let Pick::Open(lanes) = &attack.picks[0] else {
            panic!("the two 10s must ask which lane\n{}", attack.render(false));
        };
        let one_lane = lanes.focus(&state, 1);
        assert!(!one_lane.is_empty());
        assert_ne!(one_lane, whole_verb, "one lane is less than both");
        for id in one_lane.cards() {
            assert!(
                whole_verb.cards().contains(id),
                "a step down cannot add a card"
            );
        }
    }

    /// Nothing is coloured until something is being typed, and then it is one row.
    #[test]
    fn only_the_row_being_typed_is_coloured() {
        let state = GameState::new(GameConfig::split_deck(), 5);
        let menu = build(&state, &state.legal_actions(), Some(Player::P0));

        let plain = menu.render(false);
        assert!(!plain.contains('\x1b'), "no colour without a selection\n{plain}");
        assert_eq!(menu.render_with(false, None), plain);
        assert_eq!(
            menu.render_with(false, Some(99)),
            plain,
            "a number no row has colours nothing"
        );

        let painted = menu.render_with(false, Some(1));
        assert_eq!(painted.matches(HIGHLIGHT).count(), 1, "exactly one row\n{painted}");
        assert!(
            painted.contains(&format!("{HIGHLIGHT}   PLAY   #1{RESET}")),
            "the whole line, padding included\n{painted}"
        );
        // The escape codes wrap the line, so the menu reads the same without them.
        assert_eq!(painted.replace(HIGHLIGHT, "").replace(RESET, ""), plain);
    }

    /// Which *move* an action is, as distinct from which card it happens to name.
    ///
    /// FLIP and ATTACK collapse copies that are [`same_move`], so the menu offers one action
    /// out of each such class rather than all of them. Two actions share a class only when
    /// the cards behind them are equal in every field but their [`CardId`](crate::card::CardId)
    /// — which is exactly the condition `same_move` merges on, written the other way round.
    fn move_class(state: &GameState, action: Action) -> String {
        let me = state.acting_player();
        let anonymous = |lane: usize, slot: usize| {
            let mut card = state.lanes[lane].side(me)[slot].clone();
            card.id = crate::card::CardId(0);
            format!("{card:?}")
        };
        match action {
            Action::Flip { lane, slot } => {
                format!("flip {lane} {}", anonymous(lane as usize, slot as usize))
            }
            Action::Attack {
                lane,
                attacker,
                target,
            } => format!(
                "attack {lane} -> {target} {}",
                anonymous(lane as usize, attacker as usize)
            ),
            other => format!("{other}"),
        }
    }

    /// The tree is a reshaping of the engine's list, not a filter on it. Every legal *move*
    /// must be reachable, or the menu has quietly made one impossible.
    ///
    /// Two things stop this being weaker than it looks. Nothing the menu offers may be
    /// illegal — that direction is still action-for-action. And a class is defined by the
    /// whole card minus its id, so a copy that differs in anything at all is its own class
    /// and still has to be offered.
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
                let classes: Vec<String> =
                    offered.iter().map(|a| move_class(&state, *a)).collect();
                for action in &legal {
                    assert!(
                        classes.contains(&move_class(&state, *action)),
                        "seed {seed}: menu does not offer {action}, nor anything like it\n{}",
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

    /// The four verbs are always in the same order at the same numbers, whatever is legal.
    ///
    /// Four rows, not five: §4 has no pass, and the engine skips a turn with nothing in it
    /// rather than offering a row for it, so the menu never shows a way to do nothing.
    #[test]
    fn the_four_verbs_keep_their_numbers() {
        let names = ["PLAY", "FLIP", "ATTACK", "PAIR"];

        // A fresh deal: cards in hand, nothing face-up, so only PLAY can be taken.
        let opening = GameState::new(GameConfig::split_deck(), 3);
        let menu = build(&opening, &opening.legal_actions(), Some(Player::P0));
        assert_eq!(menu.len(), 4);
        for (row, name) in menu.rows.iter().zip(names) {
            assert_eq!(row.name, name);
        }
        assert!(matches!(menu.picks[0], Pick::Open(_)), "playing is on");
        assert!(matches!(menu.picks[2], Pick::Unavailable), "nothing to attack");
        assert!(matches!(menu.picks[3], Pick::Unavailable), "nothing to pair");

        // A position where attacking is on and playing is not: the numbers do not move.
        let mut p = Position::new(GameConfig::split_deck());
        p.face_up(0, Player::P0, Rank::SEVEN);
        p.face_up(0, Player::P1, Rank::FOUR);
        let midgame = p.build();
        let menu = build(&midgame, &midgame.legal_actions(), Some(Player::P0));
        assert_eq!(menu.len(), 4);
        for (row, name) in menu.rows.iter().zip(names) {
            assert_eq!(row.name, name);
        }
        assert!(matches!(menu.picks[0], Pick::Unavailable), "hand is empty");
        assert!(matches!(menu.picks[2], Pick::Open(_)), "attacking is on");
    }

    /// A card's number is its position in the column the board draws, which is *not* its
    /// slot: the observer's own base card is stored first and drawn last.
    ///
    /// PAIR is the verb that shows it, because a pair lives inside one lane and so still asks
    /// for a lane and then a card. FLIP and ATTACK reach the same list only to break a tie
    /// between copies in one lane; the numbering they use is this one.
    #[test]
    fn a_cards_number_is_where_the_board_draws_it() {
        let mut p = Position::new(GameConfig::split_deck());
        p.base(0, Player::P0, Rank::QUEEN); // slot 0, drawn last
        p.face_up(0, Player::P0, Rank::SEVEN); // slot 1, drawn first
        p.face_up(0, Player::P0, Rank::SEVEN); // slot 2
        p.face_up(0, Player::P1, Rank::FOUR);
        let state = p.build();

        assert_eq!(
            column_slots(&state, 0, Player::P0, Some(Player::P0)),
            vec![1, 2, 0],
            "your own base card is drawn at the bottom of your column"
        );
        assert_eq!(
            column_slots(&state, 0, Player::P1, Some(Player::P0)),
            vec![0],
            "the opponent's column starts at their base card"
        );

        // So the first 7 — slot 1 — is the card a player asks for as #1.
        let legal = state.legal_actions();
        let menu = build(&state, &legal, Some(Player::P0));
        let Pick::Open(pair) = &menu.picks[3] else {
            panic!("pairing must be on\n{}", menu.render(false));
        };
        let Pick::Open(cards) = &pair.picks[0] else {
            panic!("lane 1 must be reachable\n{}", pair.render(false));
        };
        assert!(cards.rows[0].note.starts_with("[7 "), "{}", cards.render(false));
        assert!(matches!(cards.picks[0], Pick::Open(_)), "the first 7 can pair");
        assert!(matches!(cards.picks[1], Pick::Open(_)), "and so can the second");
        assert!(
            matches!(cards.picks[2], Pick::Unavailable),
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
        for token in ["(K ", "[K ", "(Q ", "[Q "] {
            assert!(!text.contains(token), "menu leaked a hidden {token}\n{text}");
        }
        // The actor's own hand is theirs to see, so the 3 must still be named.
        assert!(text.contains(" 3 "), "P0's own hand is missing\n{text}");
    }

    /// FLIP and ATTACK ask for a card, and a rank with one copy on the board is the whole
    /// answer: no lane question, because the card already names its lane.
    #[test]
    fn menu_asks_for_the_card_first_and_a_single_copy_needs_no_lane() {
        let mut p = Position::new(GameConfig::split_deck());
        p.face_down(0, Player::P0, Rank::SEVEN);
        p.face_down(2, Player::P0, Rank::KING);
        p.face_up(0, Player::P0, Rank::TEN);
        p.face_up(0, Player::P1, Rank::FOUR);
        let state = p.build();
        let menu = build(&state, &state.legal_actions(), Some(Player::P0));

        let Pick::Open(flip) = &menu.picks[1] else {
            panic!("flipping must be on\n{}", menu.render(false));
        };
        let text = flip.render(false);
        assert_eq!(flip.prompt, "FLIP — which card?");
        assert_eq!(flip.len(), 2, "one row per rank, in rank order\n{text}");
        assert!(flip.rows[0].note.starts_with("7 "), "{text}");
        assert!(flip.rows[1].note.starts_with("K "), "{text}");
        assert!(
            flip.rows[0].note.contains("lane 1 (7 ²♥)"),
            "the row says where the card is, since nothing else will\n{text}"
        );
        assert!(
            matches!(flip.picks[0], Pick::Take(Action::Flip { lane: 0, .. })),
            "one 7, one lane, nothing to ask\n{text}"
        );
        assert!(
            matches!(flip.picks[1], Pick::Take(Action::Flip { lane: 2, .. })),
            "{text}"
        );

        let Pick::Open(attack) = &menu.picks[2] else {
            panic!("attacking must be on\n{}", menu.render(false));
        };
        assert_eq!(attack.prompt, "ATTACK — using which card?");
        assert_eq!(attack.len(), 1, "only the 10 is face-up");
        // The lane is settled, so the next question is already the target.
        let Pick::Open(targets) = &attack.picks[0] else {
            panic!("straight to the targets\n{}", attack.render(false));
        };
        assert!(targets.prompt.contains("which enemy card"));
    }

    /// Copies in different lanes are the one case that still needs a lane, and the question
    /// asked is which lane the card is in.
    #[test]
    fn menu_asks_which_lane_only_when_the_copies_are_spread_out() {
        let mut p = Position::new(GameConfig::split_deck());
        p.face_down(0, Player::P0, Rank::SEVEN);
        p.face_down(2, Player::P0, Rank::SEVEN);
        let state = p.build();
        let menu = build(&state, &state.legal_actions(), Some(Player::P0));

        let Pick::Open(flip) = &menu.picks[1] else {
            panic!("flipping must be on\n{}", menu.render(false));
        };
        assert_eq!(flip.len(), 1, "two copies of one rank is one row");
        assert!(flip.rows[0].note.starts_with("7 ×2"), "{}", flip.render(false));

        let Pick::Open(lanes) = &flip.picks[0] else {
            panic!("two lanes, so a lane must be asked for\n{}", flip.render(false));
        };
        let text = lanes.render(false);
        assert_eq!(lanes.prompt, "FLIP the 7 — from which lane?");
        assert_eq!(lanes.len(), 3, "every lane keeps its number\n{text}");
        assert!(matches!(lanes.picks[0], Pick::Take(Action::Flip { lane: 0, .. })), "{text}");
        assert!(matches!(lanes.picks[1], Pick::Unavailable), "no 7 in lane 2\n{text}");
        assert!(matches!(lanes.picks[2], Pick::Take(Action::Flip { lane: 2, .. })), "{text}");
    }

    /// Two copies in one lane that differ in nothing but their id are one move under two
    /// names, so the menu takes the first rather than asking a question with one answer.
    #[test]
    fn menu_does_not_ask_between_two_interchangeable_copies_in_one_lane() {
        let mut p = Position::new(GameConfig::split_deck());
        p.face_down(0, Player::P0, Rank::SEVEN);
        p.face_down(0, Player::P0, Rank::SEVEN);
        let state = p.build();
        let menu = build(&state, &state.legal_actions(), Some(Player::P0));

        let Pick::Open(flip) = &menu.picks[1] else {
            panic!("flipping must be on\n{}", menu.render(false));
        };
        let text = flip.render(false);
        assert_eq!(flip.len(), 1, "{text}");
        assert!(flip.rows[0].note.starts_with("7 ×2"), "{text}");
        assert!(
            matches!(flip.picks[0], Pick::Take(Action::Flip { lane: 0, slot: 0 })),
            "the first of two interchangeable copies, taken without a question\n{text}"
        );
    }

    /// ...but a copy that is *not* the same move is a real choice, and is still asked. A
    /// damaged card and a fresh one of the same rank flip into different cards.
    #[test]
    fn menu_still_asks_when_two_copies_in_one_lane_are_different_moves() {
        let mut p = Position::new(GameConfig::split_deck());
        p.face_down(0, Player::P0, Rank::SEVEN);
        let hurt = p.face_down(0, Player::P0, Rank::SEVEN);
        p.damage(0, Player::P0, hurt, 1);
        let state = p.build();
        let menu = build(&state, &state.legal_actions(), Some(Player::P0));

        let Pick::Open(flip) = &menu.picks[1] else {
            panic!("flipping must be on\n{}", menu.render(false));
        };
        let Pick::Open(cards) = &flip.picks[0] else {
            panic!("a damaged copy is a different move\n{}", flip.render(false));
        };
        let text = cards.render(false);
        assert_eq!(cards.prompt, "FLIP in lane 1 — which card?");
        assert!(matches!(cards.picks[0], Pick::Take(Action::Flip { slot: 0, .. })), "{text}");
        assert!(matches!(cards.picks[1], Pick::Take(Action::Flip { slot: 1, .. })), "{text}");
    }

    /// Attacking with a paired card is a different move from attacking with a loose one of
    /// the same rank in the same lane — §5 sends both members in together — so the menu says
    /// so on the card row and asks which one.
    #[test]
    fn menu_keeps_a_pair_apart_from_a_loose_card_of_the_same_rank() {
        let mut p = Position::new(GameConfig::base());
        let a = p.face_up(0, Player::P0, Rank::SEVEN);
        let b = p.face_up(0, Player::P0, Rank::SEVEN);
        p.face_up(0, Player::P0, Rank::SEVEN);
        p.pair(0, Player::P0, a, b);
        p.face_up(0, Player::P1, Rank::FOUR);
        let state = p.build();
        let menu = build(&state, &state.legal_actions(), Some(Player::P0));

        let Pick::Open(attack) = &menu.picks[2] else {
            panic!("attacking must be on\n{}", menu.render(false));
        };
        let text = attack.render(false);
        assert_eq!(attack.len(), 1, "three 7s are one row\n{text}");
        assert!(attack.rows[0].note.contains("paired"), "{text}");
        let Pick::Open(cards) = &attack.picks[0] else {
            panic!("the pair and the loose 7 are different moves\n{text}");
        };
        assert_eq!(cards.prompt, "ATTACK from lane 1 — using which card?");
        assert!(
            cards.rows[0].note.contains("pair with #2"),
            "{}",
            cards.render(false)
        );
    }
}
