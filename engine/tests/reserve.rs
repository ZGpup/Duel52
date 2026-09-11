//! The encoder reserve — `MODULAR_RULES.md` §7.
//!
//! The reserve widens the observation and the policy head so that a whole category of rules
//! change stops being a layout break. It is **opt-in**: [`GameConfig::extended_encoder`] is
//! derived from the installed powers, so the canonical ruleset encodes byte-identically to
//! the pre-reserve build and every checkpoint in `models/` still loads.
//!
//! That promise is the thing this file exists to hold up, and it has two halves that fail in
//! opposite ways:
//!
//! - **Base rulesets must not move.** This one fails *loudly* if it breaks — a checkpoint
//!   refuses to load — but it breaks the repository's entire trained lineage, so it is pinned
//!   here against literal hashes rather than left to be noticed.
//! - **Reserve features must be declared.** This one fails *silently*: a power that writes a
//!   status flag while its `needs_extended_encoder` says `false` writes into a tensor that has
//!   no room for it, and the symptom is an agent that is merely bad. So the declaration is
//!   checked against what each power actually does, from three directions.

mod common;
use common::*;

use duel52_engine::card::{STATUS_FLAG_COUNT, STATUS_SHIELDED};
use duel52_engine::encode::{
    action_dim, action_layout_hash, encode_action, encode_observation, lane_permutations,
    lane_structure, legal_mask, obs_dim, obs_layout_hash, phase_count, reserve_embedding,
    slot_features, BASE_PHASE_COUNT, EXTENDED_PHASE_COUNT, OPTION_COUNT,
};
use duel52_engine::powers::PowerId;
use duel52_engine::testkit::*;
use duel52_engine::{Action, GameConfig, GameState, Phase, Player::P0, Player::P1, Rank, Side};

/// A canonical config with one card's power swapped — the same helper `rules_mods.rs` uses.
fn with_power(power: PowerId) -> GameConfig {
    let mut cfg = GameConfig::default();
    cfg.powers[power.rank().index()] = power;
    cfg
}

/// Every reserve power, one config each.
fn reserve_configs() -> Vec<(PowerId, GameConfig)> {
    PowerId::ALL
        .iter()
        .copied()
        .filter(|p| p.needs_extended_encoder())
        .map(|p| (p, with_power(p)))
        .collect()
}

// ================================================ the base layout does not move ==

/// **The promise the whole reserve rests on.** The canonical ruleset encodes exactly as it
/// did before the reserve existed.
///
/// Pinned against literal hashes rather than derived, because a derived comparison would move
/// with the code it is checking. `b1355a841a1fdc4a` is independently corroborated: it is the
/// value written into the header of all six checkpoints in `models/`, including
/// `duel52-split-lane-gen032` and the two from the 32-core run, and those files were produced
/// before the reserve was written.
///
/// If this fails, **every trained agent in the repository is dead** and no `runs/` directory
/// can be resumed. It is not a test to update to match new behaviour; it is a test to revert
/// a change against.
#[test]
fn reserve_the_canonical_layout_is_unmoved() {
    let mut cfg = GameConfig::default();
    assert!(
        !cfg.extended_encoder(),
        "the canonical ruleset must never need the reserve — some canonical power now \
         declares `needs_extended_encoder`"
    );

    // The default, 16 slots.
    assert_eq!(obs_dim(&cfg), 3300);
    assert_eq!(action_dim(&cfg), 1324);
    assert_eq!(obs_layout_hash(&cfg), 0x4ab4_99a4_1e73_f9ff);
    assert_eq!(action_layout_hash(&cfg), 0x8016_71b9_2e72_10bf);

    // 21 slots — what every training run and every shipped checkpoint uses.
    cfg.encoding_slots = 21;
    assert_eq!(obs_dim(&cfg), 4290);
    assert_eq!(action_dim(&cfg), 2194);
    assert_eq!(
        obs_layout_hash(&cfg),
        0xb135_5a84_1a1f_dc4a,
        "this is the hash in every shipped checkpoint's header"
    );
    assert_eq!(action_layout_hash(&cfg), 0x5169_f946_1d62_7b39);
    assert_eq!(phase_count(&cfg), BASE_PHASE_COUNT);
    assert_eq!(slot_features(&cfg), 33);
}

/// Every ruleset built only from non-reserve powers shares the canonical layout, whatever
/// else it changes.
///
/// `rulesets.rs::no_ruleset_moves_the_encoder_layout` says this for the files on disk; this
/// says it for **every power variant in the enum**, including ones no config file installs
/// yet. A new Tier 2 variant that accidentally tripped the gate would be caught here on the
/// day it was written rather than when somebody wrote a config for it.
#[test]
fn reserve_non_reserve_powers_never_move_the_layout() {
    let canonical = GameConfig::default();
    for power in PowerId::ALL.iter().copied() {
        if power.needs_extended_encoder() {
            continue;
        }
        let cfg = with_power(power);
        assert!(!cfg.extended_encoder(), "{power} tripped the reserve gate");
        assert_eq!(
            obs_layout_hash(&cfg),
            obs_layout_hash(&canonical),
            "{power} moved the observation layout without declaring the reserve"
        );
        assert_eq!(
            action_layout_hash(&cfg),
            action_layout_hash(&canonical),
            "{power} moved the action layout without declaring the reserve"
        );
    }
}

// ==================================================== the declaration is honest ==

/// A power needs the extended encoder **exactly when** it uses something the extended encoder
/// provides.
///
/// This is the check that makes the silent failure unrepresentable. Forgetting
/// `needs_extended_encoder` on a power that opens a reserve phase or writes a status flag
/// would otherwise produce a ruleset whose tensor has no room for the thing it does, and
/// nothing would crash.
///
/// It is an `==`, not an `implies`: a power that declares the reserve and uses none of it
/// costs a layout break for nothing, which is the other way to get this wrong.
#[test]
fn reserve_declaration_matches_what_each_power_uses() {
    for power in PowerId::ALL.iter().copied() {
        let opens_reserve_phase = power.opens_phases().iter().any(|p| p.needs_extended_encoder());
        let uses_a_flag = !power.status_flags_used().is_empty();
        assert_eq!(
            power.needs_extended_encoder(),
            opens_reserve_phase || uses_a_flag,
            "{power}: needs_extended_encoder() is {}, but it opens {:?} and uses flags {:?}",
            power.needs_extended_encoder(),
            power.opens_phases(),
            power.status_flags_used(),
        );
        for &flag in power.status_flags_used() {
            assert!(
                (flag as usize) < STATUS_FLAG_COUNT,
                "{power} claims status flag {flag}, past the reserve's {STATUS_FLAG_COUNT}"
            );
        }
    }
}

/// Every phase a power can open has a one-hot position in the layout that power selects.
///
/// The failure this prevents is an out-of-range one-hot write: `phase_index` returns 7 for
/// `Phase::ChooseLane`, and under a base ruleset the `phase_onehot` field is only 7 wide, so
/// the write would land in `actions_remaining_onehot` instead. It cannot happen, because a
/// power that opens the phase turns the reserve on — and that is what this asserts.
#[test]
fn reserve_phases_need_the_extended_encoder() {
    for power in PowerId::ALL.iter().copied() {
        for phase in power.opens_phases() {
            if phase.needs_extended_encoder() {
                assert!(
                    power.needs_extended_encoder(),
                    "{power} opens {phase:?}, which has no one-hot position in the base layout"
                );
            }
        }
    }
    // …and the two halves of that agree about which phases are which.
    assert_eq!(
        Phase::ALL.iter().filter(|p| p.needs_extended_encoder()).count(),
        EXTENDED_PHASE_COUNT - BASE_PHASE_COUNT - 3,
        "two of the reserve's five spare phase positions are used and three are spare; \
         update this count deliberately when one is claimed"
    );
}

// ============================================================= what it costs ==

/// The extended layout adds the reserve and **nothing else**.
///
/// Stated as arithmetic against the base layout rather than as literals, so it reads as "what
/// the reserve is" rather than as a number to re-copy when the board shape changes.
#[test]
fn reserve_the_extended_layout_adds_exactly_the_reserve() {
    let base = GameConfig::default();
    let ext = with_power(PowerId::SevenShieldAll);
    let (l, s) = (base.lanes, base.encoding_slots);

    assert_eq!(slot_features(&ext), slot_features(&base) + STATUS_FLAG_COUNT);
    assert_eq!(phase_count(&ext), EXTENDED_PHASE_COUNT);
    assert_eq!(
        obs_dim(&ext),
        obs_dim(&base) + l * 2 * s * STATUS_FLAG_COUNT + (EXTENDED_PHASE_COUNT - BASE_PHASE_COUNT),
        "the observation grows by the status flags on every slot, plus the spare phase slots"
    );
    assert_eq!(
        action_dim(&ext),
        action_dim(&base) + 2 * l + OPTION_COUNT,
        "the policy head grows by CHOOSE_LANE (a lane on either side) and CHOOSE_OPTION"
    );

    // Every reserve ruleset lands on the *same* extended layout — §7's batching argument. A
    // reserve that fragmented per feature would mean one break per mechanic, which is the
    // thing the reserve exists to avoid paying.
    for (power, cfg) in reserve_configs() {
        assert_eq!(
            (obs_layout_hash(&cfg), action_layout_hash(&cfg)),
            (obs_layout_hash(&ext), action_layout_hash(&ext)),
            "{power} landed on a different extended layout"
        );
    }
}

/// The reserve is **inert until something uses it**, which is the claim §7 priced the whole
/// decision on: the search path's cost tracks the observation's *non-zeros*, not its width.
///
/// Measured on a reserve ruleset that claims **no status flag** — `king-any-lane` needs the
/// extended layout for its phase and its `CHOOSE_LANE` block, and touches no bit. So it
/// carries the full status block and every float in it is zero: at 21 slots that is 1,008
/// extra features costing exactly nothing at the input layer, which walks non-zeros.
///
/// (A ruleset that *does* claim a flag obviously sets it — `seven-shield` shields a whole
/// side at once. What is true there is that seven of the eight flags stay zero, and that is
/// a measurement for `FINDINGS.md` rather than an invariant.)
#[test]
fn reserve_status_flags_add_no_non_zeros_until_a_power_sets_one() {
    let cfg = with_power(PowerId::KingEmpowerAnyLane);
    assert!(cfg.extended_encoder());
    assert!(
        PowerId::KingEmpowerAnyLane.status_flags_used().is_empty(),
        "this test needs a reserve power that claims no flag"
    );
    let mut buf = vec![0f32; obs_dim(&cfg)];
    let mut positions = 0;
    for seed in 0..40u64 {
        for depth in [1usize, 9, 30, 75] {
            let Some(state) = position_after(cfg, seed, depth) else {
                continue;
            };
            positions += 1;
            for observer in [P0, P1] {
                encode_observation(&state, observer, &mut buf);
                // The status block of every slot is the tail of that slot's features.
                let f = slot_features(&cfg);
                for chunk in 0..(cfg.lanes * 2 * cfg.encoding_slots) {
                    let flags = &buf[chunk * f + f - STATUS_FLAG_COUNT..chunk * f + f];
                    assert!(
                        flags.iter().all(|v| *v == 0.0),
                        "a status flag is set in a game where no 7 has been flipped"
                    );
                }
            }
        }
    }
    assert!(positions > 50, "the position sample is too thin to prove anything");
}

// ================================================ the reserve actions encode ==

/// `CHOOSE_LANE` and `CHOOSE_OPTION` round-trip through the policy head, and are legal only
/// in their own phase.
///
/// The phase conditioning is what lets `CHOOSE_SLOT` serve four sub-decisions, and the two
/// reserve blocks inherit it: an index that names a lane decodes to nothing outside
/// `Phase::ChooseLane`, so a policy that proposed one in the main phase proposes an illegal
/// move rather than a silently different one.
#[test]
fn reserve_actions_round_trip_and_are_phase_conditioned() {
    let cfg = with_power(PowerId::KingEmpowerAnyLane);
    let mut state = king_choosing_a_lane(cfg);
    assert_eq!(state.phase(), Phase::ChooseLane);

    let legal = state.legal_actions();
    assert!(!legal.is_empty(), "a ChooseLane node is never offered without an answer");
    for action in &legal {
        let i = encode_action(action, &state);
        assert_eq!(
            duel52_engine::decode_action(i, &state).as_ref(),
            Some(action),
            "{action} did not round-trip through index {i}"
        );
    }

    // Outside the phase, the same indices mean nothing.
    let indices: Vec<usize> = legal.iter().map(|a| encode_action(a, &state)).collect();
    state.pending.clear();
    assert_eq!(state.phase(), Phase::Main);
    for i in indices {
        assert_eq!(
            duel52_engine::decode_action(i, &state),
            None,
            "a CHOOSE_LANE index decoded to an action outside Phase::ChooseLane"
        );
    }
}

/// A power that may only name its own side gets exactly `lanes` legal logits out of the
/// block's `2 · lanes`, and the rest are masked.
///
/// This is the answer to "can one `CHOOSE_LANE` block be restricted to your lanes in some
/// cases and theirs in others": yes, and the legality mask is what does it. The block is
/// `2·L` wide precisely so both shapes fit in it.
#[test]
fn reserve_choose_lane_is_restricted_by_the_legality_mask() {
    let cfg = with_power(PowerId::KingEmpowerAnyLane);
    let state = king_choosing_a_lane(cfg);

    let mut mask = vec![false; action_dim(&cfg)];
    legal_mask(&state, &mut mask);

    for action in state.legal_actions() {
        let Action::ChooseLane { side, .. } = action else {
            panic!("a ChooseLane node offered {action}");
        };
        assert_eq!(
            side,
            Side::Mine,
            "a King reactivates its owner's cards, so it may never name the opponent's lane"
        );
    }

    // Every `Theirs` logit of the block is masked off. Built by asking the encoder for the
    // index rather than by arithmetic, so this cannot drift from the layout.
    for lane in 0..cfg.lanes as u8 {
        let theirs = encode_action(
            &Action::ChooseLane {
                side: Side::Theirs,
                lane,
            },
            &state,
        );
        assert!(!mask[theirs], "the opponent's lane {lane} is legal to a King");
    }
}

// ======================================== the lane equivariance still holds ==

/// The reserve's lane-indexed block is relabelled by a lane permutation.
///
/// ⚠️ **The most dangerous line in this change.** `CHOOSE_LANE` is lane-owned and sits past
/// `CHOOSE_RANK`, which is global. A permutation table that left it fixed would still be a
/// bijection and would still compose as S₃, so the existing structural tests would pass —
/// and the lane-equivariant network would route one lane's logit through another's weights,
/// producing an agent that is merely bad with the training run as the natural suspect.
///
/// So this checks the table against [`encode_action`] itself, and separately asserts the
/// block actually *moves*, which is the part a fixed table would fail.
#[test]
fn reserve_lane_permutations_relabel_choose_lane() {
    let cfg = with_power(PowerId::KingEmpowerAnyLane);
    let state = king_choosing_a_lane(cfg);
    let perms = lane_permutations(&cfg);
    assert_eq!(perms.len(), 6);

    let mut moved = false;
    for sigma in &perms {
        let permuted = permute_lanes(&state, &sigma.lanes);
        for lane in 0..cfg.lanes as u8 {
            for side in [Side::Mine, Side::Theirs] {
                let action = Action::ChooseLane { side, lane };
                let from = encode_action(&action, &state);
                let to = encode_action(&permute_action(&action, &sigma.lanes), &permuted);
                assert_eq!(
                    sigma.action[from] as usize, to,
                    "{:?} maps CHOOSE_LANE({side}, {lane}) to the wrong logit",
                    sigma.lanes
                );
                moved |= from != to;
            }
        }
    }
    assert!(
        moved,
        "no permutation moved a CHOOSE_LANE logit — the table is the identity on the block, \
         which is exactly the silent failure this test exists for"
    );
}

/// The lane/global partition still covers the policy head exactly, with `CHOOSE_LANE`
/// lane-owned and `CHOOSE_OPTION` global.
///
/// `lane_structure` asserts its own partition internally, so the useful thing here is the
/// *widths*: the reserve makes `global_action` non-contiguous for the first time, and a
/// filter that dropped or double-counted a logit would show up as a width mismatch before it
/// showed up as a bad agent.
#[test]
fn reserve_lane_structure_partitions_the_wider_head() {
    let cfg = with_power(PowerId::TwoViewChoose);
    let st = lane_structure(&cfg);

    assert_eq!(st.lane_action.len(), cfg.lanes);
    for lane in &st.lane_action {
        assert_eq!(lane.len(), duel52_engine::encode::lane_action_len(&cfg));
    }
    assert_eq!(
        st.global_action.len(),
        duel52_engine::encode::global_action_len(&cfg)
    );
    assert_eq!(
        cfg.lanes * duel52_engine::encode::lane_action_len(&cfg) + st.global_action.len(),
        action_dim(&cfg),
        "the lane and global logit counts no longer add up to the policy head"
    );
    assert_eq!(
        cfg.lanes * duel52_engine::encode::lane_obs_len(&cfg)
            + duel52_engine::encode::global_obs_len(&cfg),
        obs_dim(&cfg)
    );

    // `global_action` is a filter now, not a range. If it were still written as
    // `choose_rank..total` it would swallow the six lane-owned CHOOSE_LANE logits and
    // `assert_partitions` would fire — but assert the shape here too, so the reason is
    // recorded rather than inferred from a panic.
    assert!(
        st.global_action.windows(2).any(|w| w[1] != w[0] + 1),
        "global_action should be non-contiguous in the extended layout"
    );
}

// ==================================================== the checkpoint bridge ==

/// A base-layout checkpoint can be widened into the extended layout **exactly**.
///
/// `MODULAR_RULES.md` §7. This is what keeps the reserve affordable: without it the first
/// ruleset to claim a flag pays a 24-hour from-scratch run.
///
/// The property is that the base layout embeds in the extended one with every feature keeping
/// its meaning, and that the features with no preimage are exactly the reserve's — which are
/// zero in any position a base ruleset could produce. Checked against
/// [`encode_observation`] itself: encode the same position under both layouts and require the
/// embedded tensor to match float for float.
#[test]
fn reserve_embedding_preserves_the_encoding() {
    let base = GameConfig::default();
    // `SevenShieldAll` never fires in these positions (no 7 is flipped face-up by
    // construction below), so the two configs describe the same *game* at the same seed and
    // the tensors are comparable.
    let ext = with_power(PowerId::SevenShieldAll);
    let embed = reserve_embedding(&ext);

    assert_eq!(embed.obs.len(), obs_dim(&base));
    assert_eq!(embed.action.len(), action_dim(&base));
    assert_eq!(embed.extended_obs_dim, obs_dim(&ext));
    assert_eq!(embed.extended_action_dim, action_dim(&ext));

    // Monotone and injective. Monotonicity is what makes a widened checkpoint's forward pass
    // *bit*-identical and not merely mathematically equal: the input layer accumulates over
    // non-zeros in index order, and a reordering would change the floating-point
    // associativity, which the determinism contract forbids.
    assert!(
        embed.obs.windows(2).all(|w| w[0] < w[1]),
        "the observation embedding is not monotone"
    );
    assert!(embed.action.windows(2).all(|w| w[0] < w[1]));

    let mut a = vec![0f32; obs_dim(&base)];
    let mut b = vec![0f32; obs_dim(&ext)];
    let mut checked = 0;
    for seed in 0..25u64 {
        for depth in [1usize, 11, 40, 90] {
            let Some(sa) = position_after(base, seed, depth) else {
                continue;
            };
            // **The same position**, encoded under the other layout — not the same seed
            // played out under the other ruleset. A reserve ruleset is a different *game*
            // (a shield changes what survives), so two playouts diverge and comparing them
            // would prove nothing. Swapping the config on one state isolates the layout,
            // which is the only thing the embedding is about. Every normalisation the
            // encoder reads — `max_plies`, `copies_per_rank`, `hand_size` — is identical
            // between the two, because `ext` is `base` with one power swapped.
            let mut sb = sa.clone();
            sb.config = ext;
            checked += 1;
            for observer in [P0, P1] {
                encode_observation(&sa, observer, &mut a);
                encode_observation(&sb, observer, &mut b);
                let mut claimed = vec![false; b.len()];
                for (i, &v) in a.iter().enumerate() {
                    let j = embed.obs[i] as usize;
                    claimed[j] = true;
                    assert_eq!(
                        v, b[j],
                        "base float {i} does not survive the embedding to extended float {j}"
                    );
                }
                // Everything with no preimage is a reserve feature, and is zero.
                for (j, &v) in b.iter().enumerate() {
                    if !claimed[j] {
                        assert_eq!(v, 0.0, "unclaimed extended float {j} is not zero");
                    }
                }
            }
        }
    }
    assert!(checked > 40, "the position sample is too thin to prove anything");
}

/// The lane-partition half of the bridge: the same embedding, in the coordinates the
/// lane-equivariant network actually holds its weights in.
///
/// `arch = "lane"` — every checkpoint in `models/` worth widening — has no `[width, obs_dim]`
/// matrix to gather. It has `[width, lane_obs_len]` and `[width, global_obs_len]`, so the
/// bridge has to be expressed per partition. The property checked here is that following the
/// flat embedding and then locating the result in the extended partition gives back exactly
/// these maps, i.e. that the two statements of the bridge agree.
#[test]
fn reserve_embedding_agrees_with_the_lane_partition() {
    let base = GameConfig::default();
    let ext = with_power(PowerId::KingEmpowerAnyLane);
    let embed = reserve_embedding(&ext);
    let (bs, es) = (lane_structure(&base), lane_structure(&ext));

    assert_eq!(embed.lane_obs.len(), duel52_engine::encode::lane_obs_len(&base));
    assert_eq!(embed.global_obs.len(), duel52_engine::encode::global_obs_len(&base));
    assert_eq!(embed.lane_action.len(), duel52_engine::encode::lane_action_len(&base));
    assert_eq!(
        embed.global_action.len(),
        duel52_engine::encode::global_action_len(&base)
    );

    // Every lane, not just the one `reserve_embedding` builds from.
    for lane in 0..base.lanes {
        for (k, &pos) in embed.lane_obs.iter().enumerate() {
            assert_eq!(
                embed.obs[bs.lane_obs[lane][k] as usize],
                es.lane_obs[lane][pos as usize],
                "lane {lane} observation position {k} disagrees with the flat embedding"
            );
        }
        for (k, &pos) in embed.lane_action.iter().enumerate() {
            assert_eq!(
                embed.action[bs.lane_action[lane][k] as usize],
                es.lane_action[lane][pos as usize],
                "lane {lane} policy position {k} disagrees with the flat embedding"
            );
        }
    }
    for (k, &pos) in embed.global_obs.iter().enumerate() {
        assert_eq!(embed.obs[bs.global_obs[k] as usize], es.global_obs[pos as usize]);
    }
    for (k, &pos) in embed.global_action.iter().enumerate() {
        assert_eq!(
            embed.action[bs.global_action[k] as usize],
            es.global_action[pos as usize]
        );
    }

    // All four are injective, or a widened weight matrix would have two rows written onto
    // one and a row left at its random initialisation.
    for (name, map, width) in [
        ("lane_obs", &embed.lane_obs, duel52_engine::encode::lane_obs_len(&ext)),
        ("global_obs", &embed.global_obs, duel52_engine::encode::global_obs_len(&ext)),
        ("lane_action", &embed.lane_action, duel52_engine::encode::lane_action_len(&ext)),
        (
            "global_action",
            &embed.global_action,
            duel52_engine::encode::global_action_len(&ext),
        ),
    ] {
        let mut hit = vec![false; width];
        for &i in map.iter() {
            assert!(!hit[i as usize], "{name} maps two positions onto {i}");
            hit[i as usize] = true;
        }
    }
}

/// The action half of the bridge: the base policy head is an exact prefix of the extended
/// one, so a widened checkpoint's existing logits do not move.
#[test]
fn reserve_embedding_leaves_the_base_policy_head_in_place() {
    let ext = with_power(PowerId::KingEmpowerAnyLane);
    let embed = reserve_embedding(&ext);
    for (i, &j) in embed.action.iter().enumerate() {
        assert_eq!(
            i, j as usize,
            "base logit {i} moved to {j}; the reserve blocks must be appended, not inserted"
        );
    }
}

// ============================================ the three variants, as rules ==

/// **`seven-shield`.** A shielded card ignores one hit, and the shield is spent doing it.
///
/// Driven through a real attack rather than by poking the damage counter, because the whole
/// question is whether the shield sits on the path every hit takes.
#[test]
fn mod_seven_shield_absorbs_exactly_one_hit() {
    let cfg = with_power(PowerId::SevenShieldAll);
    let mut p = Position::new(cfg);
    // Two attackers, so both hits fit in one turn and nothing depends on turn machinery.
    p.face_up(0, P0, Rank::FOUR); // 1 damage, no combat power
    p.face_up(0, P0, Rank::FIVE); // likewise, once it is already face-up
    p.face_up(0, P1, Rank::JACK); // 3 HP face-up, so it survives both hits
    p.state_mut().lanes[0].side_mut(P1)[0].set_status(STATUS_SHIELDED);
    let mut s = p.build();

    let hit = |attacker: u8| Action::Attack {
        lane: 0,
        attacker,
        target: 0,
    };

    s.apply(hit(0)).expect("attack the shielded card");
    let card = &s.lanes[0].side(P1)[0];
    assert_eq!(card.damage, 0, "the shield should have absorbed the whole hit");
    assert!(!card.has_status(STATUS_SHIELDED), "and been spent doing it");

    s.apply(hit(1)).expect("attack again");
    assert_eq!(
        s.lanes[0].side(P1)[0].damage,
        1,
        "the shield is spent, so the second hit lands normally"
    );
}

/// **`seven-shield`.** Flipping the 7 shields the whole side, in every lane — the same reach
/// as Heal All, so the two are comparable as an experiment.
#[test]
fn mod_seven_shield_covers_every_lane_on_the_flip() {
    let cfg = with_power(PowerId::SevenShieldAll);
    let mut p = Position::new(cfg);
    p.face_down(0, P0, Rank::SEVEN);
    p.face_up(1, P0, Rank::FOUR);
    p.face_up(2, P0, Rank::FIVE);
    p.face_up(1, P1, Rank::SIX);
    let mut s = p.build();

    s.apply(Action::Flip { lane: 0, slot: 0 }).expect("flip the 7");

    for lane in 0..3 {
        for card in s.lanes[lane].side(P0) {
            assert!(
                card.has_status(STATUS_SHIELDED),
                "lane {lane}: the 7 should shield every card its owner has"
            );
        }
    }
    for card in s.lanes[1].side(P1) {
        assert!(!card.has_status(STATUS_SHIELDED), "it shields the owner only");
    }
}

/// **`king-any-lane`.** The King reactivates the lane the player picks, not its own.
#[test]
fn mod_king_any_lane_reactivates_the_chosen_lane() {
    let cfg = with_power(PowerId::KingEmpowerAnyLane);
    let mut p = Position::new(cfg);
    p.face_down(0, P0, Rank::KING);
    // A reactivatable power in a *different* lane from the King.
    p.face_up(2, P0, Rank::ACE);
    let mut s = p.build();

    s.apply(Action::Flip { lane: 0, slot: 0 }).expect("flip the King");
    assert_eq!(s.phase(), Phase::ChooseLane, "the King asks which lane");

    // Only lane 2 has anything to reactivate, so §8's fizzle rule means it is the only offer.
    let legal = s.legal_actions();
    assert_eq!(
        legal,
        vec![Action::ChooseLane {
            side: Side::Mine,
            lane: 2
        }],
        "a lane with nothing to reactivate is not offered"
    );

    s.apply(legal[0]).expect("choose lane 2");
    // The Ace's reactivation grants an action, which is how we can tell it fired at all.
    assert!(
        s.actions_remaining >= 2,
        "the Ace in lane 2 should have refired and granted an action"
    );
}

/// **`king-any-lane`.** A King with nothing to reactivate anywhere fizzles rather than asking
/// a question with no answer (`game_rules.md` §8).
#[test]
fn mod_king_any_lane_fizzles_with_no_targets() {
    let cfg = with_power(PowerId::KingEmpowerAnyLane);
    let mut p = Position::new(cfg);
    p.face_down(0, P0, Rank::KING);
    p.face_up(1, P0, Rank::EIGHT); // constant: not reactivatable
    let mut s = p.build();

    s.apply(Action::Flip { lane: 0, slot: 0 }).expect("flip the King");
    assert_ne!(s.phase(), Phase::ChooseLane, "there was nothing to choose between");
}

/// **`two-choose`.** The player picks where the card goes, and both options work.
#[test]
fn mod_two_choose_offers_both_destinations() {
    let cfg = with_power(PowerId::TwoViewChoose);
    for (option, expect_discard) in [(0u8, false), (1u8, true)] {
        let mut p = Position::new(cfg);
        p.face_down(0, P0, Rank::TWO);
        p.hand(P0, &[Rank::NINE]);
        // §6: a 2 whose pile is empty does nothing at all — no draw and no give-back.
        p.pile(P0, &[Rank::SIX]);
        let mut s = p.build();
        let discards_before = s.discards[P0.idx()].len();

        s.apply(Action::Flip { lane: 0, slot: 0 }).expect("flip the 2");
        assert_eq!(s.phase(), Phase::GiveBack, "the rank is chosen first");
        let give = s.legal_actions()[0];
        s.apply(give).expect("give a card back");

        assert_eq!(s.phase(), Phase::ChooseOption, "then the destination");
        assert_eq!(
            s.legal_actions().len(),
            2,
            "two of the block's {OPTION_COUNT} logits are legal; the rest are masked"
        );
        s.apply(Action::ChooseOption { option }).expect("choose a destination");

        let discarded = s.discards[P0.idx()].len() > discards_before;
        assert_eq!(
            discarded, expect_discard,
            "option {option} sent the card to the wrong place"
        );
    }
}

/// A game plays end to end under each reserve ruleset, and every action it chose encodes.
///
/// The cross-ruleset suite already plays these configs, but it does not touch the encoder.
/// This is the combination that matters: a reserve power firing *and* its action going
/// through the policy head, which is where a mis-sized block would show up.
#[test]
fn reserve_rulesets_play_and_encode_end_to_end() {
    for (power, cfg) in reserve_configs() {
        let mut reached = 0;
        for seed in 0..60u64 {
            let mut state = GameState::new(cfg, seed);
            let mut rng = duel52_engine::Rng::derive(seed, 0x5E5E_2026_0907_0001);
            let mut mask = vec![false; action_dim(&cfg)];
            while !state.outcome.is_over() {
                let legal = state.legal_actions();
                // Every legal action has a logit, and the mask agrees with the list.
                legal_mask(&state, &mut mask);
                assert_eq!(
                    mask.iter().filter(|m| **m).count(),
                    legal.len(),
                    "{power}: the mask and the legal list disagree in phase {}",
                    state.phase()
                );
                if state.phase().needs_extended_encoder() {
                    reached += 1;
                }
                let action = *rng.choose(&legal).expect("a running game has actions");
                state.apply_trusted(action);
            }
        }
        // The `SevenShieldAll` ruleset opens no reserve *phase* — it uses a status flag — so
        // only the two phase-opening powers are expected to get here.
        if !power.opens_phases().is_empty() {
            assert!(
                reached > 0,
                "{power} never reached its own phase in 60 games, so this proved nothing"
            );
        }
    }
}

// =================================================================== helpers ==

/// A position with a `KingEmpowerAnyLane` flipped and waiting for its lane.
fn king_choosing_a_lane(cfg: GameConfig) -> GameState {
    let mut p = Position::new(cfg);
    p.face_down(0, P0, Rank::KING);
    // One reactivatable power in every lane, so every lane is a legal answer and the test
    // sees the whole block rather than a single offer.
    p.face_up(0, P0, Rank::FOUR);
    p.face_up(1, P0, Rank::FIVE);
    p.face_up(2, P0, Rank::ACE);
    let mut s = p.build();
    s.apply(Action::Flip { lane: 0, slot: 0 }).expect("flip the King");
    assert_eq!(s.phase(), Phase::ChooseLane);
    s
}
