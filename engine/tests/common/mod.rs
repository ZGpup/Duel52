//! Shared helpers for the rules tests.
//!
//! Test naming follows `CLAUDE.md`: "Every ruling in `game_rules.md` gets a named test.
//! Test names reference the rule section, e.g. `rule_6_king_reactivates_ace_grants_one_action`."

#![allow(dead_code)]

use duel52_engine::display::render;
use duel52_engine::testkit::Position;
use duel52_engine::{Action, GameConfig, GameState, Player, Rank, Rng, Variant};

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

/// One position per sub-decision phase, found by walking random games.
///
/// [`sample_positions`] stops at fixed depths, so whether it lands inside a Foresight or a
/// Queen's move is luck — and luck that shifts whenever the legal-action list changes shape.
/// This walks until it has seen each phase instead, so a test that needs every kind of
/// decision node gets them. Still not hand-built: these are positions a random game reached.
pub fn sub_decision_positions() -> Vec<GameState> {
    let mut out: Vec<GameState> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    'search: for variant in Variant::ALL {
        let config = GameConfig::preset(variant);
        for seed in 0..400u64 {
            if seen.len() == 5 {
                break 'search;
            }
            let mut state = GameState::new(config, seed);
            let mut rng = Rng::derive(seed, 0x50B_DEC1_5104_2000);
            while !state.outcome.is_over() {
                let phase = state.phase().to_string();
                if !state.pending.is_empty() && !seen.contains(&phase) {
                    seen.push(phase);
                    out.push(state.clone());
                }
                let legal = state.legal_actions();
                let action = *rng.choose(&legal).expect("a running game has actions");
                state.apply_trusted(action);
            }
        }
    }
    assert_eq!(
        seen.len(),
        5,
        "only reached the sub-decision phases {seen:?}; the encoder tests need all five"
    );
    out
}

/// Give `player` a spare card in hand so their turn survives the action under test.
///
/// `game_rules.md` §4 has no pass, so the engine ends a turn the moment nothing in it is
/// legal (`apply.rs`'s `skip_turns_with_nothing_to_do`). A test that builds a position with
/// exactly one interesting action, takes it, and then asserts on `actions_remaining` or on
/// what is *now* illegal would find the turn already over and the assertions pointing at the
/// opponent. A card in hand is always a legal play (§4), so it keeps the turn alive without
/// touching anything about combat, pairs or powers — which is what those tests are about.
pub fn spare_action(p: &mut Position, player: Player) {
    p.hand(player, &[Rank::KING]);
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
