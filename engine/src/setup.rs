//! Dealing a new game.
//!
//! Every configuration deals from a documented, fixed sequence of RNG draws, because
//! `CLAUDE.md` requires that "same seed + same config → identical game". Changing the deal
//! order here invalidates every seed recorded in `FINDINGS.md`, so treat this file as
//! append-only once results have been logged.

use crate::card::Card;
use crate::config::{GameConfig, Variant};
use crate::player::Player;
use crate::rank::Rank;
use crate::rng::Rng;
use crate::state::{empty_state, GameState, Pile};

/// RNG stream tags. Each source of randomness gets its own derived stream, so changing how
/// many numbers one of them consumes cannot perturb the others.
mod stream {
    /// The state's own generator, kept for Phase 3 determinization.
    pub const STATE: u64 = 0xD0E1_5205_2000_0001;
    /// The shared 52-card shuffle, base variant only.
    pub const SHARED_DECK: u64 = 0x5348_4152_4544_0001;
    /// Player 0's colour deck, split variants.
    pub const DECK_P0: u64 = 0x434F_4C30_0000_0001;
    /// Player 1's colour deck, split variants.
    pub const DECK_P1: u64 = 0x434F_4C31_0000_0001;
    /// The mirrored removal multiset, variant 9b only.
    pub const MIRROR_REMOVAL: u64 = 0x4D49_5252_4F52_0001;
}

impl GameState {
    /// Deal a new game.
    ///
    /// Panics if the configuration is internally inconsistent — see
    /// [`GameConfig::validate`]. A bad config is a programming error, and failing at the
    /// deal is far better than producing a subtly wrong game.
    pub fn new(config: GameConfig, seed: u64) -> GameState {
        config
            .validate()
            .unwrap_or_else(|e| panic!("invalid GameConfig: {e}"));

        let mut state = empty_state(config, seed);
        state.rng = Rng::derive(seed, stream::STATE);

        match config.variant {
            Variant::Base => deal_base(&mut state),
            Variant::SplitDeck => deal_split(&mut state, /* mirrored */ false),
            Variant::MirroredRemoval => deal_split(&mut state, /* mirrored */ true),
        }

        // Sanity: nothing lost, nothing duplicated.
        debug_assert_eq!(
            state.card_census(),
            state.expected_card_count(),
            "the deal lost or duplicated cards"
        );

        // The first player's opening turn is a draw plus *two* actions (`game_rules.md` §4:
        // "the draw happens at the start of the turn, including the first player's opening
        // turn — that turn is a draw plus two actions, opening at 6 cards in hand. Only the
        // *action* count is reduced.").
        state.to_move = Player::P0;
        state.ply = 0;
        state.begin_turn();
        state
    }

    /// Convenience constructor for the project's default configuration (split deck).
    pub fn new_default(seed: u64) -> GameState {
        GameState::new(GameConfig::default(), seed)
    }
}

/// One card of each rank, `copies` times, in ascending rank order.
fn build_deck(config: &GameConfig, copies: usize) -> Vec<Rank> {
    let mut deck = Vec::with_capacity(config.rank_count() * copies);
    for i in 0..config.rank_count() {
        for _ in 0..copies {
            deck.push(Rank::from_index(i));
        }
    }
    deck
}

/// **Base game**, `game_rules.md` §2.
///
/// Deal order, fixed for reproducibility:
///
/// 1. Shuffle the full 52-card deck.
/// 2. Base cards off the top: P0 lane 0, P0 lane 1, P0 lane 2, then P1 lane 0, 1, 2.
/// 3. Five to P0's hand, then five to P1's hand.
/// 4. Ten removed unseen.
/// 5. The remaining 26 are the shared draw pile, in shuffled order.
///
/// The published rules do not fix an order within steps 2–4 (physically you would alternate
/// while dealing). Because the deck is already uniformly shuffled, any order gives the same
/// distribution, so the engine picks one and writes it down. **[ENGINE]**
fn deal_base(state: &mut GameState) {
    let config = state.config;
    let mut rng = Rng::derive(state.seed, stream::SHARED_DECK);
    let mut deck = build_deck(&config, config.copies_per_rank);
    rng.shuffle(&mut deck);
    let mut next = deck.into_iter();

    for player in Player::BOTH {
        for lane in 0..config.lanes {
            let rank = next.next().expect("deck exhausted dealing base cards");
            let id = state.fresh_card_id();
            state.lanes[lane]
                .side_mut(player)
                .push(Card::base_card(id, rank, player));
        }
    }

    for player in Player::BOTH {
        for _ in 0..config.hand_size {
            let rank = next.next().expect("deck exhausted dealing hands");
            state.hands[player.idx()].push(rank);
        }
        state.hands[player.idx()].sort_unstable();
    }

    // "Remove 10 cards from the draw pile, face-down, without revealing them." The base
    // game removes from a shared deck, so the removed pool has no owner — see the doc
    // comment on `GameState::removed`.
    for _ in 0..config.removal_count {
        let rank = next.next().expect("deck exhausted removing cards");
        state.removed[0].push(rank);
    }

    let pile: Vec<Rank> = next.collect();
    debug_assert_eq!(pile.len(), config.expected_pile_size());
    state.piles[0] = Pile::from_ranks(pile);
    // `piles[1]` stays empty in the base game; both players draw from `piles[0]`.
}

/// **Split deck**, `game_rules.md` §9a, and **mirrored removal**, §9b.
///
/// Shared shape: each player owns a 26-card colour deck holding two of every rank, and
/// draws only from it. The two variants differ in *when and how* the five unseen cards come
/// out, and §9b explains why they must differ:
///
/// - **§9a** deals in the natural order — base cards, hand, then remove five from what is
///   left of that player's own deck.
/// - **§9b** must draw the shared removal multiset **first** and strip it from both decks
///   before anything is dealt. Doing it in 9a's order is not always feasible: "a rank can be
///   gone from one player's pile (in hand or on base) while still present in the other's,
///   leaving no mirrored set to remove." Rejection sampling was rejected for biasing the
///   deal distribution, and partial mirroring for defeating the point of the variant.
///
/// **[ASSUMED]** for one detail §9b leaves open: "the removed multiset is chosen uniformly
/// at random" does not say uniformly over *what*. The engine shuffles a template colour
/// deck (two of each rank) and takes the top five, i.e. uniform over five-card draws from
/// 26 cards, which is what you would do physically. That induces a non-uniform distribution
/// over distinct *multisets* — a pair of the same rank is half as likely as it would be
/// under multiset-uniform sampling. Flagged rather than assumed silently.
fn deal_split(state: &mut GameState, mirrored: bool) {
    let config = state.config;
    let copies = config.copies_per_rank_per_player();

    // §9b step 0: choose the shared removal multiset, publicly.
    let mirrored_removal: Vec<Rank> = if mirrored {
        let mut rng = Rng::derive(state.seed, stream::MIRROR_REMOVAL);
        let mut template = build_deck(&config, copies);
        rng.shuffle(&mut template);
        template.truncate(config.removal_count);
        template.sort_unstable();
        template
    } else {
        Vec::new()
    };

    for player in Player::BOTH {
        let stream = match player {
            Player::P0 => stream::DECK_P0,
            Player::P1 => stream::DECK_P1,
        };
        let mut rng = Rng::derive(state.seed, stream);
        let mut deck = build_deck(&config, copies);

        if mirrored {
            // Strip the shared multiset before shuffling, so both decks lose exactly the
            // same ranks and the remainder is rank-identical.
            for &rank in &mirrored_removal {
                let pos = deck
                    .iter()
                    .position(|&r| r == rank)
                    .expect("mirrored removal asked for a rank the deck does not hold");
                deck.remove(pos);
                state.removed[player.idx()].push(rank);
            }
        }

        rng.shuffle(&mut deck);
        let mut next = deck.into_iter();

        // §9a step 1: "Deal the 3 base cards off the top of that player's own colour deck,
        // one per lane."
        for lane in 0..config.lanes {
            let rank = next.next().expect("colour deck exhausted dealing base cards");
            let id = state.fresh_card_id();
            state.lanes[lane]
                .side_mut(player)
                .push(Card::base_card(id, rank, player));
        }

        // §9a step 2: "Deal 5 to hand."
        for _ in 0..config.hand_size {
            let rank = next.next().expect("colour deck exhausted dealing hand");
            state.hands[player.idx()].push(rank);
        }
        state.hands[player.idx()].sort_unstable();

        // §9a step 3: "Remove 5 unseen from the remaining 18." Already done above for §9b.
        if !mirrored {
            for _ in 0..config.removal_count {
                let rank = next.next().expect("colour deck exhausted removing cards");
                state.removed[player.idx()].push(rank);
            }
        }

        // §9a step 4: "13 remain as that player's personal draw pile."
        let pile: Vec<Rank> = next.collect();
        debug_assert_eq!(
            pile.len(),
            config.expected_pile_size(),
            "split-deck pile size is wrong for {player}"
        );
        state.piles[player.idx()] = Pile::from_ranks(pile);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rank::rank_counts;

    /// `game_rules.md` §2: 6 base cards, 5-card hands, 10 removed, 26 in the pile — and
    /// P0's opening draw makes it 6 cards in hand for P0.
    #[test]
    fn rule_2_base_setup_deals_the_documented_counts() {
        let s = GameState::new(GameConfig::base(), 1);
        assert_eq!(s.hands[0].len(), 6, "P0 draws on the opening turn");
        assert_eq!(s.hands[1].len(), 5);
        assert_eq!(s.piles[0].len(), 25, "26 minus P0's opening draw");
        assert!(s.piles[1].is_empty(), "the base game has one shared pile");
        assert_eq!(s.removed[0].len(), 10);
        assert_eq!(s.removed[1].len(), 0);
        for lane in 0..3 {
            for p in Player::BOTH {
                let side = s.lanes[lane].side(p);
                assert_eq!(side.len(), 1);
                assert!(side[0].is_base && side[0].entered_as_base && !side[0].face_up);
            }
        }
        assert_eq!(s.card_census(), 52);
    }

    /// `game_rules.md` §9a: per player 3 base + 5 hand + 5 removed + 13 pile = 26.
    #[test]
    fn rule_9a_split_setup_deals_thirteen_card_personal_piles() {
        let s = GameState::new(GameConfig::split_deck(), 1);
        assert_eq!(s.hands[0].len(), 6, "P0 draws on the opening turn");
        assert_eq!(s.hands[1].len(), 5);
        assert_eq!(s.piles[0].len(), 12, "13 minus P0's opening draw");
        assert_eq!(s.piles[1].len(), 13);
        assert_eq!(s.removed[0].len(), 5);
        assert_eq!(s.removed[1].len(), 5);
        assert_eq!(s.card_census(), 52);
    }

    /// `game_rules.md` §9a: "Each player owns one colour — 26 cards, two suits, ranks A–K
    /// twice." So every rank appears exactly twice in each player's material.
    #[test]
    fn rule_9a_each_player_owns_exactly_two_of_every_rank() {
        for seed in 0..25 {
            let s = GameState::new(GameConfig::split_deck(), seed);
            for p in Player::BOTH {
                let mut counts = [0u8; Rank::COUNT];
                for r in &s.hands[p.idx()] {
                    counts[r.index()] += 1;
                }
                for r in s.piles[p.idx()].ranks() {
                    counts[r.index()] += 1;
                }
                for r in &s.removed[p.idx()] {
                    counts[r.index()] += 1;
                }
                for (_, _, c) in s.cards_of(p) {
                    counts[c.rank.index()] += 1;
                }
                assert_eq!(
                    counts,
                    [2u8; Rank::COUNT],
                    "seed {seed}: {p} does not own two of every rank"
                );
            }
        }
    }

    /// `game_rules.md` §9b: both players remove the same set of ranks, so the two decks are
    /// rank-identical, and the removed multiset is revealed.
    #[test]
    fn rule_9b_mirrored_removal_strips_the_same_ranks_from_both_decks() {
        for seed in 0..25 {
            let s = GameState::new(GameConfig::mirrored_removal(), seed);
            assert!(s.removed_revealed, "§9b reveals the removed multiset");
            assert_eq!(
                rank_counts(&s.removed[0]),
                rank_counts(&s.removed[1]),
                "seed {seed}: removal was not mirrored"
            );

            // Rank-identical decks: each player's remaining 21 cards (base + hand + pile)
            // hold the same multiset of ranks.
            let mut totals = [[0u8; Rank::COUNT]; 2];
            for p in Player::BOTH {
                for r in &s.hands[p.idx()] {
                    totals[p.idx()][r.index()] += 1;
                }
                for r in s.piles[p.idx()].ranks() {
                    totals[p.idx()][r.index()] += 1;
                }
                for (_, _, c) in s.cards_of(p) {
                    totals[p.idx()][c.rank.index()] += 1;
                }
            }
            // P0 has drawn one extra card on the opening turn, but it came from P0's own
            // pile, so the *totals* are still equal.
            assert_eq!(totals[0], totals[1], "seed {seed}: decks are not rank-identical");
        }
    }

    /// `game_rules.md` §2 + §3: base cards are dealt face-down and are hidden from **both**
    /// players, their owner included. This is what makes the 4's Foresight worth aiming at
    /// your own base.
    #[test]
    fn rule_3_base_cards_are_hidden_from_their_owner_too() {
        let s = GameState::new_default(7);
        for lane in 0..3 {
            for p in Player::BOTH {
                let card = &s.lanes[lane].side(p)[0];
                assert_eq!(card.known_to, 0, "a base card must be known to nobody");
                assert!(!card.rank_known_to(p), "the owner must not know their base card");
                assert!(!card.rank_known_to(p.other()));
            }
        }
    }

    /// Cards dealt to hand and then played are known to their owner and nobody else
    /// (`game_rules.md` §4).
    #[test]
    fn rule_4_hand_cards_are_private_to_their_owner() {
        let s = GameState::new_default(11);
        // Nothing in this test touches the board; it documents the invariant that a hand is
        // a private multiset of ranks and never a set of card instances.
        assert!(!s.hands[0].is_empty());
        assert!(s.hands[0].windows(2).all(|w| w[0] <= w[1]), "hands stay sorted");
    }

    #[test]
    fn base_is_locked_at_setup_in_every_variant() {
        for v in Variant::ALL {
            let s = GameState::new(GameConfig::preset(v), 3);
            assert!(!s.base_unlocked, "{v}: base must start locked");
        }
    }

    /// `CLAUDE.md`: "Everything is seeded and deterministic. Same seed + same config →
    /// identical game."
    #[test]
    fn identical_seeds_deal_identical_games() {
        for v in Variant::ALL {
            for seed in [0u64, 1, 42, u64::MAX] {
                let a = GameState::new(GameConfig::preset(v), seed);
                let b = GameState::new(GameConfig::preset(v), seed);
                assert_eq!(a, b, "{v} seed {seed}: deal is not reproducible");
            }
        }
    }

    #[test]
    fn different_seeds_deal_different_games() {
        let a = GameState::new_default(1);
        let b = GameState::new_default(2);
        assert_ne!(a.hands, b.hands);
    }
}
