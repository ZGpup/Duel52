# `configs/rules/` — the ruleset registry, and how rule changes work

**The rules of Duel 52 are configuration.** Every card's power, every combat number and every
structural constant is a field on `GameConfig`; a ruleset is a `.toml` file in this directory;
and the engine reads what it is given rather than what the rulebook says. This file is the map:
where a change lives, what it costs, and what stops it going wrong.

Read [the three tiers](#classify-the-change-before-you-cost-it) before pricing any rule change,
and [provenance](#provenance) before trusting any number that came out of one.

> Source comments cite `MODULAR_RULES.md §N`. That document is no longer tracked; what still
> binds is here and in `CLAUDE.md`, and [Old section numbers](#old-section-numbers) resolves a
> citation.

## The registry

**This directory is the registry.** A ruleset is a `.toml` file here and nothing else — there
is no list to add it to. `engine/tests/rulesets.rs` enumerates the directory at test time, so
a new file is covered by every cross-ruleset invariant the moment it exists. If registering a
ruleset required also registering its test, at twenty rulesets you would forget, and the one
you forgot would be the one that quietly fails to terminate inside a 24-hour run.

## Anatomy of a ruleset

```toml
include = "canonical.toml"            # resolved against THIS file's directory
rules_name = "three-vengeance-1"      # a label; not part of rules_hash
powers.three = "trap_vengeance_one_damage"
```

- **`include` is spliced in where it appears**, depth first. Later keys win, so anything the
  file writes itself overrides what it included. A cycle is an error, not a truncation.
- ⚠️ **Put `include` first.** Last-wins is uniform and applies to `rules_name` too, so an
  `include` written *below* the name silently replaces it with the base's. This happened: every
  ruleset reported itself as `canonical-2026-09` while its `rules_hash` stayed correct, so
  nothing looked wrong except the label. `duel52 config <file>` prints the resolved name; check
  it once when you add a file.
- **`rules_name` is provenance, not a rule.** Two files that play the same game have the same
  `rules_hash` whatever they are called.
- **Explicit is better than clever.** `duel52 config configs/rules/<file>` prints the fully
  resolved form, which is exactly what gets stamped into every shard and game record.

## What is here

| File | Tier | What changes |
|---|---|---|
| `canonical.toml` | — | The rules as written plus the house 2. The baseline every other file is a diff against. |
| `three-vengeance-1.toml` | 2 | The 3 also deals 1 damage to the card that killed it. The worked example the mod system was designed around. |
| `three-vengeance-2.toml` | 2 | The same, for 2 damage. Exists to show that a number inside a shape is a **second name**, not a config knob. |
| `three-none.toml` | 2 | Ablation: the 3 has no Trap at all. What is the Trap worth? |
| `eight-on-survival.toml` | 2 | The 8 retaliates **only if it survives** the attack — the exact inverse of the rules-as-written ruling that it "fires even if that damage killed the 8". |
| `eight-none.toml` | 2 | Ablation: the 8 does not hit back. |
| `jack-2hp.toml` | 1 | The Jack still taunts but has 2 HP, not 3. A pure config number — no code, no new power. |
| `two-blast-four-bomb.toml` | 2 | The 2 and 4 become face-down traps, like the 3. A 2 killed face-down deals 1 to every enemy card in its lane; a 4 killed face-down kills its killer. Both replace View and Foresight. `configs/train-mod-traps-3h.toml` is its warm start. |
| `seven-shield.toml` | 3 ⚠️ | The 7 **shields** instead of healing: each of your cards ignores the next damage it takes. Claims reserve status flag 0. |
| `king-any-lane.toml` | 3 ⚠️ | The King reactivates a lane **you choose**, not its own. Claims the reserve's `CHOOSE_LANE` block and a reserve phase. |
| `two-choose.toml` | 3 ⚠️ | The 2 lets you pick bottom **or** discard, per use. Claims the reserve's `CHOOSE_OPTION` block — `game_rules.md` §10a's ruling becomes an in-game decision. |

⚠️ marks the rulesets that claim the [encoder reserve](#the-encoder-reserve) and therefore play
on the extended layout. No shipped checkpoint plays them without `nn widen` first.

## Classify the change before you cost it

The question that separates the tiers is **not** how big the rule sounds; it is *what the
engine and the tensors need in order to express it*.

| Tier | What changes | Cost | Warm start |
|---|---|---|---|
| **1 — a number** | A quantity in a rule that already exists | One line of TOML. No code, no rebuild | From the current champion |
| **2 — a power shape** | What a card does, built from parts the engine already has | A named variant in the rank's module: ~150 lines the first time, ~20 after | From the current champion |
| **3 — a new mechanic** | Something the tensors cannot say | Covered by the reserve: Tier 2 plus a layout break already paid for. Uncovered: a layout revision and a from-scratch run | After `nn widen`, if covered |

**Tier 1.** Eleven fields, each named for what it is and defaulted to the rules-as-written
value: `default_hp`, `jack_hp`, `single_attack_damage`, `pair_attack_damage`,
`nimble_vs_taunt_multiplier`, `twinstrike_split_damage`, `eight_retaliate_damage`,
`ace_bonus_actions`, `ace_attack_allowance`, `six_freeze_turns`, `seven_heal_amount`.

**Tier 2.** The parts are damage, healing, flipping, freezing, drawing, moving, and a
sub-decision that picks a card. In the tree: `ThreeTrapVengeance1`/`2`,
`EightRetaliateOnSurvival`, `TwoBlast1`, `FourBomb`, and the `ThreeNone` / `EightNone`
ablations.

**Tier 3.** A per-card status nothing else models, a new *kind* of sub-decision, or a target
that is not a card.

**The decision procedure:**

1. Does an existing `GameConfig` field already say it? → **Tier 1.**
2. Can it be written with damage, healing, flips, freezes, draws, moves, and a sub-decision
   that picks a card? → **Tier 2.**
3. Does it need a per-card status, a new kind of decision, or a lane/option target? →
   **Tier 3, covered by the reserve.** One layout, shared with every other reserve ruleset.
4. Anything else → **Tier 3, uncovered.** Price it as a layout revision, and batch it with
   every other uncovered change you know about.

### Why Tier 2 is cheap

Two properties, neither designed on purpose.

**The encoder is rank-agnostic.** A card contributes a rank one-hot plus generic mechanical
state — damage, hit points, frozen, paired, attacks used — and what its rank *means* is
something the network learns. So changing what a card does moves no feature, a checkpoint
trained under one ruleset has exactly the right shape for another, and a rules experiment is a
**3-hour warm start from the current champion** instead of 24 hours from noise.
`rulesets.rs::no_ruleset_moves_the_encoder_layout` asserts it over every file here. This is the
single most valuable accident in the project, and the reserve exists for the one category of
change that cannot have it.

**`CHOOSE_SLOT` is multiplexed by phase.** One block of `2·L·S` logits serves four
sub-decisions — a 4's Foresight, a 5 or King's resolution order, a Queen's source, a 10's
second target — because their phases are mutually exclusive, so the legality mask
disambiguates and no two collide on one logit. A new power that picks **a card** therefore
needs no new logits, only a new phase. Sharing `FLIP` or `PAIR` across same-rank cards would
*not* be safe: those collide inside a single phase.

## How a rule lives in the engine

**`config.powers: [PowerId; 13]`**, one variant per rank, validated so a config cannot put the
King's Empower on the 4. It is an enum and not a trait object because `GameConfig` is
`Copy + PartialEq` and is serialised into every shard and game record; `[PowerId; 13]` keeps all
three properties for free and `[&'static dyn CardRules; 13]` keeps none. Dispatch is a match on
a small C-like enum, so there is no dynamic dispatch in the search hot path, and `GameState`
stays `Clone + PartialEq`, which matters because determinization clones states constantly.
Adding a variant is a compile error at **every** match that does not handle it. That is the
point, so nothing in `powers/` carries a `_ =>` arm.

**One module per rank**, `engine/src/powers/<rank>.rs`. Changing what the 3 does touches
`three.rs` and nothing else; `mod.rs` holds `PowerId`, the hooks and the dispatch. Nothing
outside `powers/` may branch on a `Rank` (`CLAUDE.md`, Architecture). The same goes for anything
that *describes* the rules: `duel52 powers` reads the config, because a teaching screen that
described the defaults under a modded ruleset would lead the reader to check the engine against
it and conclude the engine is wrong.

⚠️ **A variant carries no data. A number inside a shape gets its own name.**
`ThreeTrapVengeance1` and `ThreeTrapVengeance2`, not one variant plus a
`three_vengeance_damage` key. A key that applies only under some *other* key's value is one
`from_config_str` cannot reject when it is inapplicable, and an inert-but-hashed key gives two
descriptions of the same game different `rules_hash`es. The named form cannot express the
problem. Tier 1's fields are a different thing: they parameterise powers that are present in
the canonical ruleset. The rule is only that a **new variant** gets a name rather than a knob.

**What this deliberately is not.** Not an engine copy per ruleset: twenty forks of `apply.rs`
means every bug fix is twenty patches and every invariant is asserted twenty times or not at
all. And not a data-driven effect DSL, which is the less obvious one. The interesting rules in
this game are not effects but exceptions written *around* effects — the 8 fires even when the
damage killed it, the 9 is immune to that retaliation, both members of a pair pay it, the 3
springs only while face-down, a Queen breaks a pair but not a freeze. A DSL either cannot
express those or grows until it is Rust with worse tooling and no type checker.

## Damage is a queue, not recursion

Every "on death, do X" power rests on this. `enqueue_damage` adds a `Hit` and `drain_damage`
applies them in order, so a death trigger that enqueues more damage lands it *after* everything
already in flight. `drain_damage` is re-entrant and returns at once if a drain is already
running, which is what keeps the order one FIFO rather than a call stack. Under recursion, a 10
that twinstrikes two face-down vengeance 3s drops the second 3's vengeance, because the loop
that spawned it has moved on (`mod_three_vengeance_from_two_traps_in_one_twinstrike_both_land`).
`engine/src/damage.rs`'s module header has the longer version.

**Two arguments keep a cascade finite, and a new power has to fit one of them.**

- **Retaliation cannot chain.** `DamageSource::attackers()` is empty for non-attack damage, so
  vengeance damage cannot be retaliated against and cannot start a second cascade.
- **A death trigger fires at most once per card.** `TwoBlast1` *does* chain — it hits face-down
  cards, so it can spring a 3, set off a 4's Bomb, or kill a face-down enemy 2 whose blast comes
  back — and `attackers()` does not stop that, because the blast is not aimed at an attacker.
  What ends it is a count: every death trigger in the tree fires only on a **face-down** card,
  and afterwards the card has left play (2, 4) or turned face-up (3). Nothing turns a card
  face-down again (`game_rules.md` §7), so a cascade is bounded by the cards on the board.

⚠️ A death trigger that fires **face-up**, or any power that turns a card face-down, breaks the
second argument. `DamageQueue::MAX_CASCADE = 512` asserts as a backstop and nothing relies on
it; what actually catches the break is `no_ruleset_reaches_the_ply_cap` ([Testing](#testing)).

The 8 is the case for modules at all. Retaliate is the one canonical back-edge in the damage
graph, with three exceptions written around it: it is read **before** damage lands, so it fires
even when the attack killed the 8; the 9 is immune; and both members of a pair pay it.
`EightRetaliateOnSurvival` changes *when* it is read without disturbing the other two, as a mode
inside `eight.rs` rather than a rewrite.

## Provenance

`GameConfig::rules_hash()` is a 64-bit fingerprint of **the game these rules describe** — every
power, every number, the deck and the turn structure. Not `rules_name`, which is a label, and
not `encoding_slots`, which sizes a tensor rather than changing the game.

| | moves when | checked by |
|---|---|---|
| `rules_hash` | the *game* changes | `Shard::read`, `ladder`, `match`, `probe`, `card-value`, `analyze`, the Python replay buffer |
| `obs_layout_hash` / `action_layout_hash` | the *tensors* change | `Weights::load`, `Shard::read` |

Almost every ruleset moves the first and not the second. That is what makes warm starts work,
and it is also why a shard or checkpoint from another ruleset is **indistinguishable on shape
alone**. Before `rules_hash` existed, `lane-gen032` would play `--variant base` at full speed
with no warning, and the score looked exactly like a measurement.

- ⚠️ **It is not checked in `Weights::load`**, because generation 1 of every warm-started run is
  legitimately cross-ruleset. It is checked wherever a cross-ruleset number would be read *as a
  result*.
- ⚠️ **The gate and panel go through `duel52 match`**, and a warm-started incumbent is a
  checkpoint trained on other rules until a candidate replaces it. `match --warm-start-gate`
  downgrades that refusal to a warning, and the loop passes it for warm-started runs only.
  Re-stamping the copied checkpoint was rejected: a run that never promoted a candidate would
  leave an incumbent claiming rules it was never trained on. Until 2026-09-14 every modded warm
  start died scoring its baseline, and `runs/mod-three-vengeance` is what that leaves — an
  incumbent byte-identical to lane-gen032, no shards, no log.
- ⚠️ **An unstamped checkpoint cannot be judged.** None of the six in `models/` carries a stamp.
  Under a modded ruleset they are **refused**; under canonical they are accepted with a loud
  warning, because "probably canonical" is an inference and the mechanism exists not to make
  one.
- **Cross-ruleset play is a hazard by default, and an experiment only when named.** `analyze`
  and `card-value` take `--allow-cross-ruleset`, for a rules experiment's **control column**:
  the checkpoint the run warm-started from, playing the new rules beside the fine-tuned agent,
  which is how you tell "learned the mod" from "got stronger". `meta.json` records `trained_on`
  and `cross_ruleset`, the analysis document flags the column in its header and provenance
  table, and a `card-value` run on that corpus inherits the permission rather than needing a
  flag someone has to remember. `ladder`, `match` and `probe` have no such flag, because a score
  between two agents on different footings is exactly the number the refusal exists to stop.

## The encoder reserve

Tier 3 used to cost a layout break and a from-scratch run. The reserve is capacity added once,
so that most of the category no longer does.

| Reserve | Width | Unlocks |
|---|---:|---|
| 5 spare `phase_onehot` positions (7 → 12) | 5 floats | Any new *kind* of sub-decision — per [Why Tier 2 is cheap](#why-tier-2-is-cheap), the constraint that actually bites |
| `CHOOSE_LANE` (`2·L`) + `CHOOSE_OPTION` (4) policy blocks | 10 logits | A target that is a lane, or a nameless modal choice |
| 8 per-slot `status_flags` | 8 per slot | Per-card statuses: shielded, poisoned, marked, stunned |

At `encoding_slots = 21`, `obs_dim` goes 4290 → **5303** and `action_dim` 2194 → **2204**.

- **`CHOOSE_LANE` is a lane on either side**, `2·L` rather than two blocks of `L`, and the
  legality mask is what restricts it. A power acting on your own lanes emits only `Side::Mine`,
  so the three `Theirs` logits are never legal. `king-any-lane` is the worked example.
- **`CHOOSE_OPTION`'s four options are unnamed.** What an index means belongs to the power that
  opened the node, so one block serves every modal power without the encoder learning anything
  about any of them.
- **A status is public**, like damage and unlike a peeked rank. The observation has to be a
  function of the information set, and a flag only one player could see would need encoding
  per observer, which the slot block has no room for. Model a status as a token sitting on the
  card.

**It is opt-in, and derived rather than declared.** `GameConfig::extended_encoder()` is
`powers.iter().any(|p| p.needs_extended_encoder())` and every width in `encode.rs` keys off it,
so the canonical layout is byte-identical to the pre-reserve build and there are exactly two
layouts, ever. `CLAUDE.md` (Architecture) has why it must never become a config key.

**What it costs: nothing measurable in self-play.** The input layer walks only the
observation's non-zeros (`FINDINGS.md` F3.3), the trunk is what self-play pays for and its width
is unchanged, and status flags are zero by nature —
`reserve_status_flags_add_no_non_zeros_until_a_power_sets_one` shows all 1,008 extra floats stay
zero until a power sets one. Measured with `duel52 selfplay --games 48 --sims 256
--encoding-slots 21` on the 8-core laptop, `lane 128×3`, two repeats each: canonical 0.795 and
0.668 games/sec, `seven-shield` 0.725 and 0.720. Canonical's own repeats differ by more than the
two rulesets do, so read it as no cost resolvable above scheduling noise, not as a ratio. What
you do pay is parameters and memory in the input projection.

What follows from a ruleset claiming it, in the order you will hit it:

- **No shipped checkpoint plays it.** `models/*.d52nn` are all base-layout, and `--init-from`
  refuses them by name and number rather than quietly mis-reading them.
- **One `nn widen` fixes that, exactly:**

  ```bash
  .venv/bin/python -m duel52.nn widen \
      --in models/duel52-split-lane-gen032.d52nn --out models/lane-gen032-wide.d52nn \
      --rules-file configs/rules/seven-shield.toml --encoding-slots 21
  ```

  Every reserve feature is **appended** — status flags after the base slot features, spare
  phases after the seven real ones, the two blocks after `CHOOSE_RANK` — so the base layout
  embeds in the extended one monotonically. Every trained weight keeps its meaning, and the new
  rows are exactly the reserve's, which are zero in any position a base ruleset can produce, so
  the widened net plays identically until the new rules fire. `encode::reserve_embedding`
  computes the map, in Rust like every other reading of the layout; Python only scatters with
  it. Then `--init-from` the result as usual: a reserve ruleset is still a **3-hour warm
  start**.
- **Every reserve ruleset shares one layout**, whatever part of the reserve it claims. The
  tenth flag-using ruleset costs no more than the first.
- `duel52 config <file>` prints `reserve status_flags=8 phases=5` in the resolved layout for
  these and nothing for the others, which is the quickest way to tell which kind you have.

### Adding a power that uses the reserve

1. Add the variant to `PowerId`, with its token, rank, display name and text.
2. Declare what it uses: `needs_extended_encoder()`, plus `opens_phases()` and/or
   `status_flags_used()`. ⚠️ **This is the one silent failure in the reserve**: a power that
   writes a flag while declaring `false` writes into a tensor with no room for it. It is checked
   from three directions rather than trusted — `reserve_declaration_matches_what_each_power_uses`
   requires the declaration to be an `==` with what the power actually uses, the slot writer
   asserts its own width, and `Phase::needs_extended_encoder` has to agree.
3. Claim a status flag by name in `card.rs` (`STATUS_SHIELDED` is flag 0), never by a bare
   index.
4. Implement it in the rank's module, and give it a named test.

⚠️ `CHOOSE_LANE` is lane-indexed, so `encode::action_permutation` must relabel it. A table that
left it fixed would still be a bijection and still compose as S₃, so the structural tests would
pass and only the *meaning* would be wrong: the lane-equivariant network would route one lane's
logit through another's weights. `reserve_lane_permutations_relabel_choose_lane` asserts the
block actually moves.

## Testing

- **`rules_*.rs` pin the canonical ruleset** — mechanically, because `Position::empty()` builds
  `GameConfig::default()`.
- **`rules_mods.rs` and `reserve.rs` hold the named tests for each variant.** A mod is not in
  `game_rules.md`, so its tests are named `mod_<ruleset>_<what it asserts>` rather than for a
  rule section, and each states the canonical behaviour it departs from.
- **`rulesets.rs` runs 11 invariants over every file here**: valid and distinct; round-trips
  through its config string; canonical is the shipped default; deterministic; never reaches the
  ply cap; always offers a move until the game ends; conserves cards; leaves no dead card
  standing; the observation is a function of the information set, including any status a power
  sets; no agent reads hidden information, so a mod cannot open a leak; and exactly two encoder
  layouts, with the base one canonical.

⚠️ **A `PlyLimit` draw is a bug report, not a result.** `game_rules.md` §7 proves the game
finite from *specific rules*: powers fire on flip, a King reactivates once, nothing ever turns a
card face-down again. A variant that turns a card face-down, or lets a King reactivate a King,
breaks that proof. `max_plies` would log it as a draw, so the suite fails it instead.

`rulesets.rs` is a cross product, so it is the suite that slows as the registry grows. Keep its
per-ruleset game counts low: these are structural properties, and a leak or a non-termination
shows in tens of games, not thousands. If it does get slow, cut games per ruleset before
cutting rulesets.

## Adding a ruleset

1. Write the file. Keep it a **diff**: include a base and change what you mean to change.
2. `./target/release/duel52 config configs/rules/<file>` — validates it, prints the resolved
   form, and tells you its `rules_hash` and which cards differ from canonical.
3. `cargo test --test rulesets` — every structural invariant, over every file here.
4. `./target/release/duel52 screen` — the cheap behavioural read, no training involved.

**Only then is it worth a training run**, because measuring a ruleset is the part that costs
money. A Tier 2 variant is ~150 lines the first time and ~20 thereafter; measuring it is

```
edit a card module → warm-start from the current champion → probe + card-value → FINDINGS → decide
```

and twenty rulesets at a 3-hour warm start each is 60 hours of box time. `screen` is the tier
below that. `greedy` and `ismcts` take no checkpoint, so they play any ruleset the day the file
is written, and it reports draw rate, game length, lane concentration and per-rank flip rate in
minutes. It will not tell you a ruleset is *good*, because search agents are not the meta. It
reliably tells you one is **broken** — degenerate, drawish, or holding a card nobody plays —
which at twenty candidates is most of the value. Keep `random` in the roster for scale.

For the run itself, `configs/train-mod-3h.toml` and `configs/train-mod-traps-3h.toml` are the
templates, and `CLAUDE.md`'s Commands section has the warm-start and control-column
invocations.

## Priced but not built: peek as a main action

`[PowerId; 13]` assumes a rule change is a *card* change, and not all of them are. The worked
example: spend one of your three actions to look at any face-down card, and give the 4 some
other power. That is a fifth entry in `game_rules.md` §4's action list rather than a power, so
it needs its own config axis (`peek_action = off | any_face_down | unknown_only`) and its own
branch in `legal_main_actions`. **It is not implemented.** It prices at **Tier 2**, which is not
the intuition:

- **No new logits, and no reserve.** `decode_action`'s `CHOOSE_SLOT` branch dispatches on
  `state.phase()`, and `Phase::Main` is the one phase that does not claim the block — it falls
  through to `_ => None`. A `Main` arm fills the last free slot in the multiplexer.
- **No new phase**, because the decision happens *in* Main rather than on a pending node, so the
  phase one-hot does not move. ⚠️ When the 4 loses Foresight, **do not delete
  `Phase::Foresight`**: a dead variant costs nothing, and deleting it renumbers `phase_index`
  and breaks the observation hash for no benefit.
- **No new state.** `do_peek` is `known_to |= me.bit()`, which the observation already reads
  through `card.rank_known_to(observer)`, and `legal_peeks` reads nothing off the Foresight node,
  so `legal_main_actions` can call it unchanged.

Two mechanical wrinkles. `do_peek` ends with `pending.pop()` and must not in Main. And
`Action::costs_an_action()` is a pure function of the variant, so it cannot say "free as the 4's
power, one action in Main". But `legal_actions` returns `legal_main_actions()` exactly when
`pending` is empty, so replacing the match with `pending.is_empty()` in `dispatch` makes the
rule what §4 actually says — a sub-decision is free, a main action costs one — and lets one
ruleset carry both Foresight and the peek action.

## What a card is worth

A mod needs a before-and-after, and training for three hours to watch a win rate against a
frozen reference move is confounded and slow: it cannot tell "the 3 got better" from "the meta
shifted around it". `duel52 card-value` reads a card against the other twelve instead. Read
`CLAUDE.md`'s Commands section before running it — the CONTROL line, and why the column that
counts is `in hand`.

The canonical baseline, measured on `lane-gen032`, 400 positions, in win-probability points:

| rank | power | kind | in hand | ± | on board | gap |
|---|---|---|---:|---:|---:|---:|
| A | Action | one-shot | **+3.50** | 0.14 | −1.50 | +5.00 |
| J | Taunt | constant | **+2.46** | 0.20 | +4.15 | −1.69 |
| Q | Move | one-shot | **+2.24** | 0.13 | −1.97 | +4.21 |
| 7 | Heal All | one-shot | **+1.97** | 0.15 | −0.25 | +2.22 |
| 5 | Flip | one-shot | **+1.21** | 0.17 | −2.49 | +3.70 |
| 8 | Retaliate | constant | **+0.67** | 0.15 | +4.01 | −3.34 |
| 9 | Nimble | constant | **−0.12** | 0.10 | +0.45 | −0.57 |
| 6 | Freeze | one-shot | **−0.97** | 0.15 | −0.05 | −0.92 |
| 10 | Twinstrike | constant | **−1.19** | 0.12 | +1.90 | −3.09 |
| K | Empower | one-shot | **−1.31** | 0.12 | −2.24 | +0.93 |
| 3 | Trap | condition | **−1.62** | 0.16 | +0.54 | −2.16 |
| 2 | View | one-shot | **−2.99** | 0.13 | −1.35 | −1.64 |
| 4 | Foresight | one-shot | **−3.83** | 0.13 | −1.20 | −2.64 |

A spread of **7.33 points**, and the 2 and the 4 at the bottom are why `two-blast-four-bomb`
exists. The sign that the method works is that `in hand` interleaves the power kinds while
`on board` still sorts all four constants to the top. `gap` says where a card's value lives:
large and positive (the Ace, the Queen) means it is in the flip and the card is a spent body
afterwards; negative (the 8, the 10) means it is the body standing in the lane. The control,
each rank substituted into the unseen opponent hand, reads a spread of `0.00000`.

⚠️ Three limits. It measures **gen032's value head, not Duel 52**. The sample leans toward
**early, high-uncertainty positions**, because asking "what if I held an `R`" needs an unseen
copy of every rank, and only 8.3% of sampled `split` positions have one. And **`mirrored` is out
of reach**: 0.06% survive, since §9b publishes the removed multiset, so the tool refuses to
print.

## Old section numbers

Where each section of the untracked `MODULAR_RULES.md` went, for the source comments that cite
one.

| `MODULAR_RULES.md` | Now |
|---|---|
| §1a | Not kept: history. Rank-naming production code was ~30 lines in one match, which is why this was a refactor rather than a rewrite |
| §1b, §1c | [Why Tier 2 is cheap](#why-tier-2-is-cheap) |
| §2 | [Classify the change before you cost it](#classify-the-change-before-you-cost-it) |
| §3, §3a, §3b | [Damage is a queue, not recursion](#damage-is-a-queue-not-recursion) |
| §4, §5, §5a, §5b | [How a rule lives in the engine](#how-a-rule-lives-in-the-engine) |
| §5c | The damage queue landed before any power variant, with zero behaviour change and the then-354 tests as proof — `engine/src/damage.rs`'s header |
| §5d | [Anatomy of a ruleset](#anatomy-of-a-ruleset) and [The registry](#the-registry) |
| §5e | [Priced but not built: peek as a main action](#priced-but-not-built-peek-as-a-main-action) |
| §6 | [Provenance](#provenance) |
| §7 | [The encoder reserve](#the-encoder-reserve) |
| §8 | [Testing](#testing) |
| §9 | [Adding a ruleset](#adding-a-ruleset) |
| §10 | [What a card is worth](#what-a-card-is-worth) |
| §11 | The build order, on branch `ruleset-configs` over 2026-09-09/11. Step 1 `rules_hash` · 2 the eleven Tier 1 fields, each defaulted to the constant it named · 3 `DamageSource` and the queue · 4 `PowerId` and `powers/` · 5 dotted keys, `include`, this directory as the registry · 6 `rulesets.rs` · 7 per-card probe counters, generalised from `three_fates` · 8 `duel52 screen` · 9 the first seven mods · 10 `duel52 card-value` · 11 the encoder reserve and `nn widen` |
