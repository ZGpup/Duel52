//! Shared helpers for the rules tests.
//!
//! Test naming follows `CLAUDE.md`: "Every ruling in `game_rules.md` gets a named test.
//! Test names reference the rule section, e.g. `rule_6_king_reactivates_ace_grants_one_action`."

#![allow(dead_code)]

use duel52_engine::display::render;
use duel52_engine::{Action, GameState};

/// Apply an action, failing loudly with the whole board if it was not legal.
pub fn go(state: &mut GameState, action: Action) {
    if let Err(e) = state.apply(action) {
        let legal: Vec<String> = state
            .legal_actions()
            .iter()
            .map(|a| a.to_string())
            .collect();
        panic!(
            "{e}\n\nlegal actions were:\n  {}\n\n{}",
            legal.join("\n  "),
            render(state, None)
        );
    }
}

/// Assert an action is *not* legal, printing the board if it unexpectedly is.
pub fn refuse(state: &GameState, action: Action) {
    assert!(
        !state.is_legal(action),
        "expected `{action}` to be illegal, but it is allowed\n{}",
        render(state, None)
    );
}

/// Assert an action *is* legal without applying it.
pub fn allow(state: &GameState, action: Action) {
    assert!(
        state.is_legal(action),
        "expected `{action}` to be legal, but it is not\n{}",
        render(state, None)
    );
}

/// Resolve the only remaining sub-decision, whatever it is. Fails if there is more than
/// one choice, so a test that means "there is exactly one thing to do here" says so.
pub fn resolve_only(state: &mut GameState) {
    let legal = state.legal_actions();
    assert_eq!(
        legal.len(),
        1,
        "expected exactly one legal sub-decision, found {}: {:?}\n{}",
        legal.len(),
        legal,
        render(state, None)
    );
    go(state, legal[0]);
}

/// Every legal action, as strings — handy in an assertion message.
pub fn legal_names(state: &GameState) -> Vec<String> {
    state
        .legal_actions()
        .iter()
        .map(|a| a.to_string())
        .collect()
}

/// Count legal actions matching a predicate.
pub fn count_legal(state: &GameState, mut pred: impl FnMut(&Action) -> bool) -> usize {
    state.legal_actions().iter().filter(|a| pred(a)).count()
}
