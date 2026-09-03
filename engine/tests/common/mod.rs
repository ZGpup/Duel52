//! Shared helpers for the rules tests.
//!
//! Test naming follows `CLAUDE.md`: "Every ruling in `game_rules.md` gets a named test.
//! Test names reference the rule section, e.g. `rule_6_king_reactivates_ace_grants_one_action`."

#![allow(dead_code)]

use duel52_engine::display::render;
use duel52_engine::{Action, GameConfig, GameState, Rng, Variant};

// ============================================================ position samples ==

/// Walk a random game forward `decisions` decisions and stop wherever that lands — mid-turn,
/// mid-cascade, anywhere. Returns `None` if the game ended first.
pub fn position_after(config: GameConfig, seed: u64, decisions: usize) -> Option<GameState> {
    let mut state = GameState::new(config, seed);
    let mut rng = Rng::derive(seed, 0xA55E_7000_0000_0001);
    for _ in 0..decisions {
        if state.outcome.is_over() {
            return None;
        }
        let legal = state.legal_actions();
        let action = *rng.choose(&legal).expect("a running game has actions");
        state.apply_trusted(action);
    }
    if state.outcome.is_over() {
        None
    } else {
        Some(state)
    }
}

/// A spread of positions across all three variants, at every stage of a game.
///
/// Shared between the Phase 2 agent tests and the Phase 3 encoder tests, which need exactly
/// the same thing: positions that are *not* hand-built, so nothing about them was chosen to
/// make the property under test hold.
pub fn sample_positions() -> Vec<GameState> {
    let mut out = Vec::new();
    for variant in Variant::ALL {
        let config = GameConfig::preset(variant);
        for seed in 0..12u64 {
            for depth in [1usize, 7, 23, 60, 110, 180] {
                if let Some(state) = position_after(config, seed, depth) {
                    out.push(state);
                }
            }
        }
    }
    assert!(
        out.len() > 100,
        "the position sample is too thin to prove anything"
    );
    out
}

// =============================================================== rules helpers ==

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
