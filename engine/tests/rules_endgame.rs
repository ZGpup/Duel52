//! Winning, drawing, and the turn structure.
//!
//! `game_rules.md` §7 calls the lane-win condition "the most important structural rule in
//! the game", and §4 the turn structure. Both are tested here, together with the terminal
//! evaluation rules of §7 and the base-unlock latch of §3.

mod common;
use common::*;

use duel52_engine::testkit::*;
use duel52_engine::{
    Action, DrawReason, GameConfig, Outcome, Player::P0, Player::P1, Rank, Variant,
};

// ============================================================ turn structure ==

/// §2: "The first player takes only **two actions** on their opening turn. Every turn
/// thereafter is three actions." §4: the draw still happens, so P0 opens at 6 cards.
#[test]
fn rule_2_first_turn_is_a_draw_plus_two_actions() {
    let mut s = GameState::new_default(1);
    assert_eq!(s.to_move, P0);
    assert_eq!(s.actions_remaining, 2, "two actions on the opening turn");
    assert_eq!(s.hand(P0).len(), 6, "but the draw still happened");
    assert_eq!(s.hand(P1).len(), 5);

    go(&mut s, Action::Pass);
    assert_eq!(s.to_move, P1);
    assert_eq!(s.actions_remaining, 3, "every turn thereafter is three");
    assert_eq!(s.hand(P1).len(), 6, "P1 drew at the start of their turn");
}

/// §4: each of the four actions costs exactly one, and the turn ends when they run out.
#[test]
fn rule_4_a_turn_is_exactly_three_actions() {
    let mut s = GameState::new_default(3);
    go(&mut s, Action::Pass); // skip P0's short opening turn
    assert_eq!(s.to_move, P1);

    for expected in [2, 1] {
        let rank = s.hand(P1)[0];
        go(&mut s, Action::Play { rank, lane: 0 });
        assert_eq!(s.actions_remaining, expected);
        assert_eq!(s.to_move, P1);
    }
    let rank = s.hand(P1)[0];
    go(&mut s, Action::Play { rank, lane: 0 });
    assert_eq!(s.to_move, P0, "the third action ended the turn");
}

/// §4: "A card may be **played, flipped, and attack all in the same turn**, actions
/// permitting."
#[test]
fn rule_4_a_card_may_be_played_flipped_and_attack_in_one_turn() {
    let mut p = Position::empty();
    p.hand(P0, &[Rank::FOUR]);
    p.face_up(0, P1, Rank::SEVEN);
    let mut s = p.build();

    go(&mut s, Action::Play { rank: Rank::FOUR, lane: 0 });
    assert_eq!(s.actions_remaining, 2);
    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    // The 4's Foresight has no face-down card left to look at, so it fizzles (§8).
    assert_eq!(s.actions_remaining, 1);
    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    assert_eq!(damage_at(&s, 0, P1, 0), 1, "all three in one turn");
    // Spending the third action ends the turn, so the counter has already been reset for
    // P1's turn rather than sitting at zero.
    assert_eq!(s.to_move, P1);
}

/// §4: "There is **no hand-size limit**."
#[test]
fn rule_4_there_is_no_hand_size_limit() {
    let mut p = Position::empty();
    p.hand(P0, &[Rank::KING; 20]);
    let s = p.build();
    // Nothing in the rules caps a hand, so a 20-card hand is a perfectly legal position and
    // every card in it is playable.
    assert_eq!(s.hand(P0).len(), 20);
    allow(&s, Action::Play { rank: Rank::KING, lane: 0 });
}

/// §4: "Draw one card from the draw pile, **if it is non-empty**." An empty pile is not an
/// error; the turn simply starts without a draw.
#[test]
fn rule_4_an_empty_pile_means_no_draw_not_an_error() {
    let mut p = Position::empty();
    p.unlock();
    p.hand(P0, &[Rank::KING]);
    p.hand(P1, &[Rank::KING]);
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::FOUR);
    p.face_up(1, P0, Rank::FOUR);
    p.face_up(1, P1, Rank::FOUR);
    p.face_up(2, P0, Rank::FOUR);
    p.face_up(2, P1, Rank::FOUR);
    let mut s = p.build();

    go(&mut s, Action::Pass);
    assert_eq!(s.to_move, P1);
    assert_eq!(s.hand(P1).len(), 1, "no draw, and no crash");
}

/// §4: "You may look at your **own played face-down cards** at any time, for free." §3: that
/// "does *not* extend to your own base cards".
#[test]
fn rule_4_you_know_your_own_played_face_down_cards_but_not_your_base() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::KING);
    p.base(1, P0, Rank::QUEEN);
    let s = p.build();

    assert!(card_at(&s, 0, P0, 0).rank_known_to(P0), "played from hand");
    assert!(!card_at(&s, 0, P0, 0).rank_known_to(P1), "and private");
    assert!(!card_at(&s, 1, P0, 0).rank_known_to(P0), "but not your base card");
}

// =============================================================== lane wins ==

/// §7: a lane is won only when **all three** conditions hold. With the pile non-empty,
/// an empty enemy lane is worth nothing.
#[test]
fn rule_7_no_lane_is_won_while_the_draw_pile_is_non_empty() {
    let mut p = Position::empty(); // piles non-empty by default
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(1, P0, Rank::FOUR);
    // P1 has nothing at all in lanes 0 and 1, and an empty hand.
    p.face_up(2, P1, Rank::FOUR);
    let s = p.build();

    assert_eq!(s.lanes_won_by(P0), 0, "condition 2 is not met");
    assert_eq!(s.outcome, Outcome::Ongoing);
}

/// §7: "So long as the opponent holds any card in hand, they can defend the lane, and it
/// cannot be won."
#[test]
fn rule_7_no_lane_is_won_while_the_opponent_holds_a_card() {
    let mut p = Position::empty();
    p.unlock();
    p.hand(P1, &[Rank::TWO]);
    p.face_up(2, P1, Rank::FOUR);
    let s = p.build();

    assert_eq!(s.lanes_won_by(P0), 0, "condition 3 is not met");
    assert_eq!(s.outcome, Outcome::Ongoing);
}

/// §7: "**Win two lanes to win the game.**"
#[test]
fn rule_7_two_empty_lanes_post_unlock_with_an_empty_hand_wins_the_game() {
    let mut p = Position::empty();
    p.unlock();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(1, P0, Rank::FOUR);
    p.face_up(2, P1, Rank::FOUR); // P1 holds only lane 2
    p.face_up(2, P0, Rank::FOUR);
    p.hand(P0, &[Rank::KING]); // P0's hand is irrelevant to P0's own win
    let mut s = p.build();

    assert_eq!(s.lanes_won_by(P0), 2);
    // The outcome latches on the next terminal check, which runs after an action resolves.
    go(&mut s, Action::Pass);
    assert_eq!(s.outcome, Outcome::Win(P0));
}

/// §7: base cards count as cards in the lane — "The opponent has **no cards remaining in
/// that lane** (including their base card)".
#[test]
fn rule_7_a_base_card_still_holds_a_lane() {
    let mut p = Position::empty();
    p.unlock();
    p.base(0, P1, Rank::SEVEN); // face-down base card holding lane 0
    p.face_up(2, P1, Rank::FOUR);
    let s = p.build();

    assert_eq!(
        s.lanes_won_by(P0),
        1,
        "only lane 1 is empty; the base card holds lane 0"
    );
    assert_eq!(s.outcome, Outcome::Ongoing);
}

/// §7: "A single action **may** complete the second and third lanes at once. This is a plain
/// win — nothing depends on the order, and there is no need to attribute the win to one
/// lane."
///
/// The action that does it is the *loser's*: P1 plays their last card from hand, which
/// satisfies condition 3 for every lane simultaneously, and P1 was already absent from two
/// of them. Both lanes are won on the same check.
#[test]
fn rule_7_a_single_action_may_win_two_lanes_at_once() {
    let mut p = Position::empty();
    p.unlock();
    // P0 holds all three lanes, so P0 can never lose one — that keeps this a plain win
    // rather than the mutual case tested below.
    for lane in 0..3 {
        p.face_up(lane, P0, Rank::FOUR);
    }
    // P1 holds only lane 0, and has exactly one card left in hand.
    p.face_up(0, P1, Rank::FOUR);
    p.hand(P1, &[Rank::KING]);
    p.to_move(P1);
    let mut s = p.build();

    assert_eq!(s.lanes_won_by(P0), 0, "P1's hand still defends every lane");
    go(&mut s, Action::Play { rank: Rank::KING, lane: 0 });

    assert_eq!(s.lanes_won_by(P0), 2, "lanes 1 and 2, both on this one check");
    assert_eq!(s.lanes_won_by(P1), 0);
    assert_eq!(s.outcome, Outcome::Win(P0), "and it is a plain win");
}

/// §7: "You *can* empty your **own** side of a lane and hand it to your opponent — a Queen
/// moving out your last card."
#[test]
fn rule_7_a_queen_can_hand_a_lane_to_your_opponent() {
    let mut p = Position::empty();
    p.unlock();
    // P1 already holds lane 2 — P0 has nothing there and P0's hand is empty — so P1 is one
    // lane short. P0's Queen sits in lane 0 and P0's only lane-1 card is a 5.
    p.face_down(0, P0, Rank::QUEEN);
    p.face_up(1, P0, Rank::FIVE);
    // P1 holds a card in every lane, so P0 can never win one and this stays decisive.
    for lane in 0..3 {
        p.face_up(lane, P1, Rank::FOUR);
    }
    let mut s = p.build();

    assert_eq!(s.lanes_won_by(P1), 1, "lane 2 only, so far");
    assert_eq!(s.lanes_won_by(P0), 0);

    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    go(&mut s, Action::MoveHere { lane: 1, slot: 0 });

    assert_eq!(occupancy(&s, 1, P0), 0, "P0 emptied their own lane 1");
    assert_eq!(s.outcome, Outcome::Win(P1), "and handed P1 the game");
}

/// §7: the one way both players can reach the threshold on the same check — "your last card
/// in a lane attacks the opponent's last card in that lane, an 8, and retaliate kills yours
/// as your damage kills theirs. Both sides of that lane empty, so **both** players win it.
/// If that leaves both at two or more lanes, it is a **draw (0.5/0.5)**."
#[test]
fn rule_7_mutual_lane_win_via_retaliate_is_a_draw() {
    let mut p = Position::empty();
    p.unlock();
    // Lane 0: P0's last card, on 1 damage, attacks P1's last card, an 8 on 1 damage.
    // Both die: P0's to retaliate, P1's to the attack.
    p.face_up(0, P0, Rank::FOUR);
    p.damage(0, P0, 0, 1);
    p.face_up(0, P1, Rank::EIGHT);
    p.damage(0, P1, 0, 1);
    // Lanes 1 and 2 are already empty on both sides, so each player is on 2 lanes and the
    // shared lane 0 takes them both to 3.
    let mut s = p.build();

    assert_eq!(s.lanes_won_by(P0), 2);
    assert_eq!(s.lanes_won_by(P1), 2);

    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    assert_eq!(occupancy(&s, 0, P0), 0, "retaliate killed the attacker");
    assert_eq!(occupancy(&s, 0, P1), 0, "and the attack killed the 8");
    assert_eq!(
        s.outcome,
        Outcome::Draw(DrawReason::MutualLaneWin),
        "a symmetric outcome gets a symmetric result"
    );
    assert_eq!(s.outcome.value_for(P0), 0.5);
    assert_eq!(s.outcome.value_for(P1), 0.5);
}

/// §7: "**A lane win cannot be undone**, so latched and live evaluation are equivalent."
/// The reasoning is that an empty side cannot be refilled: hand empty, pile empty, and a
/// Queen only moves cards *into* the lane she is already in.
#[test]
fn rule_7_a_won_lane_cannot_be_refilled_by_a_queen() {
    let mut p = Position::empty();
    p.unlock();
    // P1 has emptied lane 0 but still holds lanes 1 and 2, so the game is not over. P1 has
    // a Queen in lane 1 — she cannot pull anything back into lane 0.
    p.face_up(1, P1, Rank::QUEEN);
    p.face_up(2, P1, Rank::FOUR);
    p.face_up(0, P0, Rank::FOUR);
    p.hand(P0, &[Rank::KING]); // keeps P1 from winning, so the game continues
    p.to_move(P1);
    let s = p.build();

    let queen_moves: Vec<_> = s
        .legal_actions()
        .into_iter()
        .filter(|a| matches!(a, Action::MoveHere { .. }))
        .collect();
    assert!(
        queen_moves.is_empty(),
        "a Queen is not in the emptied lane, so nothing can return there"
    );
    assert_eq!(s.lanes_won_by(P0), 1, "P0 holds lane 0 alone");
}

// ==================================================== stalemate and draws ==

/// §7: "The engine declares a **draw (0.5/0.5)** after a configurable number of consecutive
/// turns (default 20) with no damage dealt and no kill." "'Turns' here means individual
/// player turns (plies)."
#[test]
fn rule_7_stalemate_is_declared_after_the_quiet_ply_limit() {
    let mut p = Position::empty();
    p.unlock();
    // Both sides hold every lane and neither can attack (nothing is face-up), so passing is
    // all either player can do.
    for lane in 0..3 {
        p.face_down(lane, P0, Rank::EIGHT);
        p.face_down(lane, P1, Rank::EIGHT);
    }
    p.quiet_plies(18);
    let mut s = p.build();

    go(&mut s, Action::Pass); // 19
    assert_eq!(s.outcome, Outcome::Ongoing);
    go(&mut s, Action::Pass); // 20 — the limit
    assert_eq!(s.outcome, Outcome::Draw(DrawReason::Stalemate));
}

/// §7: "The counter resets on damage or a kill and on nothing else."
#[test]
fn rule_7_the_quiet_counter_resets_on_damage_and_nothing_else() {
    let mut p = Position::empty();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(0, P1, Rank::JACK);
    p.hand(P0, &[Rank::KING]);
    p.quiet_plies(5);
    let mut s = p.build();

    // Playing a card is not damage.
    go(&mut s, Action::Play { rank: Rank::KING, lane: 1 });
    go(&mut s, Action::Pass);
    assert_eq!(s.quiet_plies, 6, "a quiet ply still counts up");

    go(&mut s, Action::Pass); // P1's turn, also quiet
    assert_eq!(s.quiet_plies, 7);

    go(
        &mut s,
        Action::Attack {
            lane: 0,
            attacker: 0,
            target: 0,
        },
    );
    go(&mut s, Action::Pass);
    assert_eq!(s.quiet_plies, 0, "damage reset it");
}

/// §7: the stalemate threshold is config-driven.
#[test]
fn rule_7_the_stalemate_threshold_is_configurable() {
    let mut config = GameConfig::split_deck();
    config.stalemate_quiet_plies = 4;

    let mut p = Position::new(config);
    p.unlock();
    for lane in 0..3 {
        p.face_down(lane, P0, Rank::EIGHT);
        p.face_down(lane, P1, Rank::EIGHT);
    }
    let mut s = p.build();

    for _ in 0..3 {
        go(&mut s, Action::Pass);
        assert_eq!(s.outcome, Outcome::Ongoing);
    }
    go(&mut s, Action::Pass);
    assert_eq!(s.outcome, Outcome::Draw(DrawReason::Stalemate));
}

/// A decisive result beats the stalemate rule: if the last quiet ply also completes a lane
/// win, the win is what gets recorded.
#[test]
fn rule_7_a_lane_win_takes_precedence_over_the_quiet_ply_limit() {
    let mut p = Position::empty();
    p.unlock();
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(1, P0, Rank::FOUR);
    p.face_up(2, P1, Rank::FOUR);
    p.face_up(2, P0, Rank::FOUR);
    p.quiet_plies(19);
    let mut s = p.build();

    go(&mut s, Action::Pass);
    assert_eq!(s.outcome, Outcome::Win(P0));
}

/// §7: an ongoing game and a draw both evaluate to 0.5; a win is 1.0/0.0.
#[test]
fn rule_7_outcome_values_are_zero_sum() {
    assert_eq!(Outcome::Win(P0).value_for(P0), 1.0);
    assert_eq!(Outcome::Win(P0).value_for(P1), 0.0);
    assert_eq!(Outcome::Draw(DrawReason::Stalemate).value_for(P0), 0.5);
    assert_eq!(Outcome::Draw(DrawReason::Stalemate).value_for(P1), 0.5);
}

// ================================================== the base-unlock latch ==

/// §3 + §9: "Base cards unlock, and lane wins become possible, when **both** draw piles are
/// empty." One empty pile is not enough.
#[test]
fn rule_9_base_unlocks_only_when_both_piles_are_empty() {
    let mut p = Position::empty();
    p.pile(P0, &[]);
    p.pile(P1, &[Rank::KING]);
    p.hand(P0, &[Rank::TWO]);
    let mut s = p.build();

    assert!(!s.base_unlocked);
    go(&mut s, Action::Pass); // P0's turn ends; P1 draws their last card
    assert_eq!(s.to_move, P1);
    assert!(s.base_unlocked, "now both piles are empty");
}

/// §3: the flag is a latch — "set once every draw pile is empty, and **never cleared**".
///
/// Nothing in real play refills an emptied pile: the house 2 bottoms into your own pile but
/// fizzles entirely when that pile is empty (§9), which is precisely so it "cannot be used
/// to refill an empty pile". So this test constructs the refill directly. It is pinning
/// defensive behaviour rather than a reachable line, which is worth doing because the
/// alternative — recomputing the flag from live pile sizes — would silently re-lock base
/// cards and un-win a won lane.
#[test]
fn rule_3_the_unlock_latch_is_never_cleared() {
    let mut p = Position::empty();
    p.pile(P0, &[]);
    p.pile(P1, &[]);
    // Both players hold a card and hold every lane, so nothing here ends the game.
    p.hand(P0, &[Rank::ACE]);
    p.hand(P1, &[Rank::ACE]);
    for lane in 0..3 {
        p.face_down(lane, P0, Rank::EIGHT);
        p.face_down(lane, P1, Rank::EIGHT);
    }
    let mut s = p.build();

    go(&mut s, Action::Pass);
    assert!(s.base_unlocked, "both piles empty, so the latch set");

    s.piles[0] = duel52_engine::Pile::from_ranks(vec![Rank::KING]);
    go(&mut s, Action::Pass);
    assert!(
        s.base_unlocked,
        "a non-empty pile must not re-lock base cards"
    );
}

/// §10a: "There is **no last-card stall**: firing a 2 when one card remains draws it and
/// bottoms one back, leaving the pile exactly where it would have been. The engine needs no
/// special case for an empty-after-draw pile."
#[test]
fn rule_10a_a_two_on_a_one_card_pile_does_not_trip_the_unlock() {
    let mut p = Position::empty();
    p.face_down(0, P0, Rank::TWO);
    p.pile(P0, &[Rank::SEVEN]);
    p.pile(P1, &[]);
    p.hand(P0, &[Rank::ACE]);
    let mut s = p.build();

    assert!(!s.base_unlocked, "P0 still has a card to draw");
    go(&mut s, Action::Flip { lane: 0, slot: 0 });
    // Mid-resolution the pile is momentarily empty. The flag must not latch here.
    assert!(
        !s.base_unlocked,
        "the unlock is only evaluated at action boundaries"
    );
    go(&mut s, Action::GiveBack { rank: Rank::ACE });
    assert_eq!(s.pile(P0).len(), 1, "pile-neutral");
    assert!(!s.base_unlocked, "and the base is still locked");
}

// ============================================================ full games ==

/// Every variant plays to a legal conclusion under random play, with no illegal action ever
/// offered and no state invariant violated. The debug assertions in
/// `GameState::debug_check_invariants` do the heavy lifting; this drives them.
#[test]
fn random_games_terminate_legally_in_every_variant() {
    for variant in Variant::ALL {
        let config = GameConfig::preset(variant);
        for seed in 0..40u64 {
            let summary = duel52_engine::stats::play_random_game(config, seed);
            assert!(
                summary.outcome.is_over(),
                "{variant} seed {seed} did not terminate"
            );
            assert_ne!(
                summary.outcome,
                Outcome::Draw(DrawReason::PlyLimit),
                "{variant} seed {seed} hit the safety cap, which means a rules bug"
            );
        }
    }
}

/// Both settings of the 2 produce legal games, so the §10a comparison is measurable.
#[test]
fn rule_10a_both_two_power_settings_play_to_a_conclusion() {
    for two_power in [
        duel52_engine::TwoPower::Bottom,
        duel52_engine::TwoPower::Discard,
    ] {
        for variant in Variant::ALL {
            let mut config = GameConfig::preset(variant);
            config.two_power = two_power;
            for seed in 0..15u64 {
                let summary = duel52_engine::stats::play_random_game(config, seed);
                assert!(summary.outcome.is_over(), "{variant}/{two_power:?} seed {seed}");
            }
        }
    }
}

use duel52_engine::GameState;
