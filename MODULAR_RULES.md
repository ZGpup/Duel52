# Modular rules — how rule changes work in this codebase

**The rules of Duel 52 are configuration.** Every card's power, every combat number and every
structural constant is a field on `GameConfig`; a ruleset is a `.toml` file in
`configs/rules/`; and the engine reads what it is given rather than what the rulebook says.
This document is the map: where a change lives, what it costs, and what stops it going wrong.

Read **§2 before pricing any rule change** and **§6 before trusting any number that came out
of one.**

> **History, and why the section numbers look like a plan's.** This began on 2026-09-09 as an
> assessment — *can this codebase absorb rule changes, and what should the mechanism look
> like?* — and was implemented over 2026-09-09/11 on branch `ruleset-configs`. It is now a
> description of what exists rather than a proposal. The **section numbering is deliberately
> unchanged**, because roughly fifty source comments cite it; the recommendations and the open
> questions are gone, because building the thing answered them. §11 is kept as a record of
> what was built in what order, for the comments that cite a step number.

---

## 1. Where the codebase stands

### 1a. Rules logic is concentrated

`engine/src` is ~21,000 lines, but production code that named a specific rank was about **30
lines**, almost all inside one 13-arm match in `apply.rs`. The rest of the engine already spoke
in terms of *properties* — "does this card taunt", "is it nimble" — rather than "is it a Jack".
Turning that into `config.power(rank)` moved the 30 lines and touched almost nothing else.

That is why a rules system was a refactor rather than a rewrite, and it is worth knowing when
estimating the next one: the number to look at is not the size of the engine, it is how much of
it branches on identity.

### 1b. The encoder is rank-agnostic, and that is worth more than everything else here

Nothing in `encode.rs` has ever had a per-power feature. A card contributes a rank one-hot plus
generic mechanical state — damage, hit points, frozen, paired, attacks used — and *what its
rank means* is something the network learns. So changing what a card does moves **no feature at
all**:

> **A ruleset that changes what cards do does not move the observation or action layout.** A
> checkpoint trained under one ruleset has exactly the right shape for another, which makes a
> rules experiment a **3-hour warm start from the current champion** instead of a 24-hour run
> from scratch. `engine/tests/rulesets.rs::no_ruleset_moves_the_encoder_layout` asserts it over
> every registered ruleset.

Nobody designed that on purpose. It is the single most valuable accident in the project, and §7
is about the one category of change that cannot have it.

### 1c. The action space is more general than the powers that use it

`CHOOSE_SLOT` is one block of `2·L·S` logits shared by four different sub-decisions — a 4's
Foresight, a 5 or King's resolution order, a Queen's source, a 10's second target — because
their **phases are mutually exclusive**, so the legality mask disambiguates and no two can
collide on one logit. Sharing `FLIP` or `PAIR` across same-rank cards would *not* have been
safe, because those collide inside a single phase.

The consequence for rule changes: a new power that picks **a card** needs no new logits, only a
new phase. That is why §7's item 1 — spare `phase_onehot` positions — was the cheapest and
highest-value part of the reserve, and why a "peek" action that costs one of your three actions
is Tier 2 rather than Tier 3 (`Phase::Main` is the one free slot in the `CHOOSE_SLOT`
multiplexer).

---

## 2. Classify the change before you cost it — the three tiers

The question that separates the tiers is **not** how big the rule sounds; it is *what the
engine and the tensors need in order to express it*.

### Tier 1 — numbers. A config field, one line at one use site.

The rule already exists; a quantity in it changes. `jack_hp = 2`, `six_freeze_turns = 2`,
`eight_retaliate_damage = 2`.

**Cost:** a line in a `.toml`. No code, no rebuild, no layout change. Warm-starts from the
current champion.

Eleven such fields exist, each named for what it is and defaulted to the rules-as-written
value: `default_hp`, `jack_hp`, `single_attack_damage`, `pair_attack_damage`,
`nimble_vs_taunt_multiplier`, `twinstrike_split_damage`, `eight_retaliate_damage`,
`ace_bonus_actions`, `ace_attack_allowance`, `six_freeze_turns`, `seven_heal_amount`.

### Tier 2 — swappable power shapes. A named variant in Rust, selected by config.

The card does something structurally different, but out of parts the engine already has:
damage, healing, flipping, freezing, drawing, moving, or a sub-decision that picks a card.

**Cost:** ~150 lines the first time in a rank's module, ~20 thereafter, plus a `.toml`. No
layout change. Warm-starts from the current champion.

In the tree: `ThreeTrapVengeance1` (the Trap also damages its killer),
`EightRetaliateOnSurvival`, and the `ThreeNone` / `EightNone` ablations.

**A number inside a shape gets its own name, not a config knob** — see §5a.

### Tier 3 — new mechanics. Needs the encoder reserve.

The card needs something the tensors cannot say: a per-card status nothing else models, a new
*kind* of sub-decision, or a target that is not a card.

**Cost:** it depends entirely on whether the **reserve** covers it (§7). Since 2026-09-11 most
of this category is reserved for, which makes it behave like Tier 2 plus one already-paid
layout break. What is *not* covered is still a genuine Tier 3: a real layout revision and a
from-scratch run.

### The decision procedure

1. Does an existing `GameConfig` field already say it? → **Tier 1.**
2. Can it be written with damage, healing, flips, freezes, draws, moves, and a sub-decision
   that picks a card? → **Tier 2.**
3. Does it need a per-card status, a new kind of decision, or a lane/option target? →
   **Tier 3, covered by the reserve.** One layout, shared with every other reserve ruleset.
4. Anything else → **Tier 3, uncovered.** Price it as a layout revision, and batch it with
   every other uncovered change you know about.

---

## 3. Damage is a queue, not recursion

The worked example that drove this design: *the 3 damages the card that killed it.* It broke
the engine's model of damage, which flowed strictly one way per action, and it is the gate on
every "on death, do X" power.

`apply.rs` used to recurse. Now `enqueue_damage` adds a `Hit` and `drain_damage` applies them
in order, so a death trigger that enqueues more damage lands it *after* everything already in
flight. `drain_damage` is re-entrant and returns immediately if a drain is already running,
which is what keeps the order one FIFO rather than a call stack.

### 3a. What vengeance actually breaks — order, then depth

**Order.** A 10 twinstrikes two face-down 3s. Under recursion the first 3's vengeance fires
inside the loop that is still iterating the second target, and the second 3's vengeance is
dropped — the loop that spawned it has moved on. The queue makes the order a single, explicit
FIFO, so both land and both land in a defined sequence.

**Depth.** Vengeance could in principle retaliate into a retaliation. What keeps a cascade
finite is `DamageSource::attackers()` returning **empty** for non-attack damage: vengeance
damage is `Vengeance`, not `Attack`, so it cannot be retaliated against and cannot start a
second cascade. `MAX_CASCADE = 512` is a backstop that asserts, not a mechanism anyone relies
on.

### 3b. The 8 is the better argument for card modules than the 3 is

Retaliate is the one canonical power that is a genuine back-edge in the damage graph, and it
has three separate exceptions written around it: it is read **before** damage lands (so it
fires even when the attack killed the 8), the 9 is immune to it, and both members of a pair
pay it. Changing *when* it is read, without disturbing the other two, is exactly the thing a
modular system has to be able to do — and `EightRetaliateOnSurvival` is that change, as a
**mode** inside `powers/eight.rs` rather than a rewrite.

---

## 4. What this deliberately is not

**Not an engine copy per ruleset.** Twenty forks of `apply.rs` means every bug fix is twenty
patches and every invariant is asserted twenty times or not at all. The cross-ruleset suite in
§8 exists precisely because one engine can be checked against every ruleset at once.

**Not a data-driven effect DSL.** This is the less obvious one. A DSL looks like the general
answer, and it fails in a specific way: the interesting rules in this game are not effects,
they are *interactions between effects* — the 8 fires even when the damage killed it, the 9 is
immune to that retaliation, both members of a pair pay it, the 3 springs only while face-down,
a Queen breaks a pair but not a freeze. Every one of those is an exception written *around* an
effect rather than an effect. A DSL either cannot express them, or grows a general-purpose
language until it is Rust with worse tooling and no type checker.

Rust with one module per rank gives exhaustiveness checking, a debugger, and a compile error
for every match that forgot the new variant. That is what a rules system actually needs.

---

## 5. The design

### 5a. Rules live in `GameConfig`, as data, selecting code

`config.powers: [PowerId; 13]` — one power variant per rank, validated so a config cannot put
the King's Empower on the 4.

**Why an enum and not a trait object.** `GameConfig` is `Copy + PartialEq` and is serialised
into every shard and every game record. `[PowerId; 13]` keeps all three properties for free;
`[&'static dyn CardRules; 13]` keeps none. Dispatch is a match on a small C-like enum, so there
is no dynamic dispatch in the search hot path, and `GameState` stays `Clone + PartialEq` —
which matters because determinization clones states constantly.

The other benefit is the one you feel while working: adding a variant produces a compile error
at **every** match that does not handle it. That exhaustiveness is the whole point, so nothing
in `powers/` carries a `_ =>` arm.

**A variant carries no data.** A number that varies within a shape gets its own **name** —
`ThreeTrapVengeance1` and `ThreeTrapVengeance2`, not one variant plus a
`three_vengeance_damage` key. The reason is provenance: a key that only applies under some
*other* key's value is a key `from_config_str` cannot reject when it is inapplicable, and an
inert-but-hashed key means two configs describing the same game get different `rules_hash`es.
The named form cannot express the problem.

Tier 1's numeric fields are a different thing and still exist — they parameterise powers that
are *present* in the canonical ruleset. The rule is only that a **new variant** gets a name
rather than a knob.

### 5b. One module per rank

`engine/src/powers/<rank>.rs`. Changing what the 3 does touches `three.rs` and nothing else;
`mod.rs` holds `PowerId`, the hooks and the dispatch.

⚠️ **Nothing outside `powers/` may branch on a `Rank` to decide behaviour.** Read
`card.live_power(&config)` and ask the `PowerId` — `taunts()`, `is_nimble()`,
`retaliate_mode()`. `live_power` returns `None` for a face-down card, which makes
`game_rules.md` §6's "powers are inert while face-down" structural rather than remembered. A
`rank == Rank::EIGHT` in `state.rs` or `apply.rs` is a bug: it silently ignores the ruleset.

The same applies to anything that *describes* the rules. `duel52 powers` reads the config, not
the rulebook — a teaching screen that quietly described the defaults under a modded ruleset is
worse than none, because the reader checks the engine against it and concludes the engine is
wrong.

### 5c. The damage refactor came first, as a no-op

§3's queue landed before any power variant existed, with **zero behaviour change** and the
then-354 tests as the proof. That ordering is the reason the vengeance variants were small:
the structural work and the rules work were never in the same change, so a test failure could
only mean one of them.

### 5d. Config format and ruleset composition

```toml
include = "canonical.toml"            # resolved against THIS file's directory
rules_name = "three-vengeance-1"      # a label; not part of rules_hash
powers.three = "trap_vengeance_one_damage"
```

Includes are spliced in where they appear, depth first; later keys win; a cycle is an error
rather than a truncation.

**`configs/rules/` is the registry.** A ruleset is a file there and nothing else — there is no
list to add it to, because `engine/tests/rulesets.rs` enumerates the directory at test time. At
twenty rulesets a registration step is a step you forget, and the one you forget is the one
that quietly fails to terminate inside a 24-hour run.

⚠️ **Put `include` first.** Last-wins is uniform and applies to `rules_name` too, so an
`include` written *below* the name silently replaces it with the base's. This actually
happened: every ruleset reported itself as `canonical-2026-09` while its `rules_hash` stayed
correct, so nothing looked wrong except the label. `duel52 config <file>` prints the resolved
name — check it once when you add a file.

---

### 5e. Turn-level actions are a second axis, and `[PowerId; 13]` does not cover them

Everything above assumes a rule change is a *card* change. Not all of them are, and this is the
one gap in the design worth knowing about. The worked example: **make peek a §4 action** —
spend one of your three actions to look at any face-down card — and give the 4 some other power
entirely.

That is not a power on a card. It is a fifth entry in `game_rules.md` §4's action list, and it
needs its own config axis (`peek_action = off | any_face_down | unknown_only`) and its own
branch in `legal_main_actions`. `config.powers` has nowhere to put it. **It is not
implemented.**

It is worth pricing because it comes out **Tier 2 — the cheapest structural change available**,
which is not the intuition:

- **The action space already has it, and `Main` is the one free slot.** `decode_action`'s
  `CHOOSE_SLOT` branch dispatches on `state.phase()` — the same logits mean `Peek` under
  Foresight, `ResolveNext` under ResolveOrder, `MoveHere` under QueenSource, `SplitTarget` under
  SplitTarget. `Phase::Main` is the only phase that does not claim the block; it falls through
  to `_ => None`. Adding a `Phase::Main` arm fills the last empty multiplexer slot. Zero new
  logits, and **no reserve needed**.
- **No new phase, so the phase one-hot does not move** — the decision happens *in* Main rather
  than on a new pending node, which is exactly the §1c trap and it misses. ⚠️ When the 4 loses
  Foresight, **do not delete `Phase::Foresight` from the enum**: a dead variant costs nothing,
  and deleting it renumbers `phase_index` and breaks the observation hash for no benefit.
- **No new state.** `do_peek` is `known_to |= me.bit()`, and the observation already reads
  exactly that through `card.rank_known_to(observer)`. A peeked-but-face-down card already
  encodes.
- **The enumerator is already phase-independent.** `legal_peeks` reads `face_down_cards()` and
  `to_move` and nothing off the Foresight node, so `legal_main_actions` can call it unchanged.

Two mechanical wrinkles, one of which is a simplification. `do_peek` ends with `pending.pop()`
and must not, in Main. And `Action::costs_an_action()` is a pure function of the variant, so it
cannot say "free as the 4's power, one action in Main" — but `legal_actions` returns
`legal_main_actions()` **iff** `pending.last()` is `None`, so `pending.is_empty()` ⟺ main phase
⟺ costs an action, exactly. Replacing the four-variant match with that check in `dispatch`
makes the rule what §4 actually says — a sub-decision is free, a main action costs one —
rather than a hand-maintained list, and lets a ruleset carry both Foresight and the peek action
with no ambiguity.

---

## 6. Provenance — `rules_hash` is not a layout hash

This is the failure the system was built around, and it is worth stating plainly because it
produces numbers that look exactly like results.

**Before this work, nothing recorded which ruleset produced a number.** `lane-gen032`, trained
on the split deck, would load and play `--variant base` at full speed with no warning at all,
and the score it produced looked like a measurement. With three variants that was a live
hazard. With twenty rulesets it would have been the dominant source of wrong conclusions.

So `GameConfig::rules_hash()` is a 64-bit fingerprint of **the game these rules describe** —
every power, every number, the deck and the turn structure. Not `rules_name`, which is a label,
and not `encoding_slots`, which sizes a tensor rather than changing the game.

### The two hashes are independent, and that is the point

| | moves when | checked by |
|---|---|---|
| `rules_hash` | the *game* changes | `Shard::read`, `ladder`, `match`, `probe`, `card-value`, the Python replay buffer |
| `obs_layout_hash` / `action_layout_hash` | the *tensors* change | `Weights::load`, `Shard::read` |

Almost every ruleset moves the first and not the second. That is exactly what makes warm
starting work — and it also means **a shard or checkpoint from another ruleset is
indistinguishable on shape alone**, which is why `rules_hash` has to exist as a separate thing.

⚠️ `rules_hash` is deliberately **not** checked in `Weights::load`, because generation 1 of
every warm-started run is legitimately cross-ruleset. It is checked wherever a cross-ruleset
number would be read *as a result*.

⚠️ **An unstamped checkpoint cannot be judged.** The four pre-2026-09-09 checkpoints carry no
rules stamp. Under a modded ruleset they are **refused**; under canonical they are accepted
with a loud warning, because "probably canonical" is an inference and the whole point of the
mechanism is not to make one.

**Cross-ruleset play is a hazard, not an experiment.** There is no case in this project where
playing an agent under rules it was not trained on answers a question, so every check above is
a refusal or a warning rather than a mode.

---

## 7. The encoder reserve

Tier 3's cost used to be a layout break and a from-scratch run. The reserve is capacity that
was added, once, so that most of the category no longer is.

### What it adds

| # | Reserve | Width | Unlocks |
|---|---|---:|---|
| 1 | 5 spare `phase_onehot` positions (7 → 12) | 5 floats | *Any* new kind of sub-decision — per §1c, the constraint that actually bites |
| 2 | `CHOOSE_LANE (2·L)` + `CHOOSE_OPTION (k=4)` policy blocks | 10 logits | Targets that are a lane, or a nameless modal choice |
| 3 | 8 per-slot `status_flags` | 8 per slot | Per-card statuses: shielded, poisoned, marked, stunned |

At `encoding_slots = 21`: `obs_dim` 4290 → **5303**, `action_dim` 2194 → **2204**.

`CHOOSE_LANE` is `2·L` — a lane on **either** side — rather than two blocks of `L`, because the
**legality mask** is what restricts it. A power acting on one of your own lanes emits only
`Side::Mine`, so the three `Theirs` logits are never legal; a power targeting an enemy lane
would use the same block from the other end. `king-any-lane` is the worked example.

`CHOOSE_OPTION`'s four options are **unnamed**. What an option index means belongs to the power
that opened the node, which lets one block serve every modal power without the encoder learning
anything about any of them.

### It is opt-in, and that is the whole design

`GameConfig::extended_encoder()` is `powers.iter().any(|p| p.needs_extended_encoder())`. Every
width in `encode.rs` keys off that one predicate, so:

- **The canonical layout is byte-identical to the pre-reserve build.** `obs_layout_hash` is
  still `b1355a841a1fdc4a` at 21 slots — the value in the header of all six checkpoints in
  `models/` — so every shipped agent still loads, every `runs/` directory still resumes, and
  every `.d52sp` shard still replays.
- **There are exactly two layouts, ever.** Every reserve ruleset shares the second one,
  whatever part of the reserve it claims. That is the batching argument made mechanical: one
  break buys the whole reserve, and the tenth flag-using ruleset costs no more than the first.

It is **derived from the powers rather than declared by a config key** on purpose. A key would
be a third thing to keep in step, and it fails silently: a ruleset that installs a flag-using
power but forgets the key writes a status nobody encodes, and the network simply never learns
the mechanic. Deriving it makes that state unrepresentable.

A status is **public**, like damage and unlike a peeked rank. The observation has to be a
function of the information set, so a flag only one player could see would have to be encoded
per-observer, which the slot block has no room for. Model a status as a token sitting on the
card.

### What it costs

The original assessment asked for a measurement rather than an argument, on the grounds that
`obs_dim` growth is the wrong number to weigh. It is: the input layer walks only the
observation's **non-zeros** (`FINDINGS.md` F3.3 — 205 of 4,290), the trunk is what self-play
actually pays for, and the trunk width is unchanged. Status flags are zero by nature.

Measured, `duel52 selfplay --games 48 --sims 256 --encoding-slots 21` on the 8-core laptop,
`lane 128×3`, two repeats each:

| ruleset | `obs_dim` | games/sec |
|---|---:|---|
| canonical | 4290 | 0.795, 0.668 |
| `seven-shield` | 5303 (+23.6%) | 0.725, 0.720 |

**The prediction holds, though not in a way that makes a clean table:** canonical's own two
repeats differ by 19%, which is more than the gap between the two rulesets. On this machine the
difference is not resolvable above scheduling noise. Read it as "no cost measurable at this
sample", not as a ratio; re-measure on a quiet box if it ever matters.

`reserve_status_flags_add_no_non_zeros_until_a_power_sets_one` is the structural half: under a
reserve ruleset that claims no flag, all 1,008 extra floats are zero and the input layer skips
every one of them.

What you *do* pay is parameters and memory in the input projection, which F3.3 puts at 58% of a
flat checkpoint.

### Crossing the break: `nn widen`

A reserve ruleset's layout is not the one any shipped checkpoint was trained against, so
`--init-from` refuses one — loudly, by name and number. The bridge is exact:

```bash
.venv/bin/python -m duel52.nn widen \
    --in models/duel52-split-lane-gen032.d52nn --out models/lane-gen032-wide.d52nn \
    --rules-file configs/rules/seven-shield.toml --encoding-slots 21
```

Every reserve feature is **appended** — status flags after the base slot features, spare phase
positions after the seven real ones, the two blocks after `CHOOSE_RANK` — so the base layout
embeds in the extended one **monotonically**, every weight keeps its meaning, and the rows with
no preimage are exactly the reserve's, which are zero in any position a base ruleset could
produce. `encode::reserve_embedding` computes the map (in Rust, like every other reading of the
layout); Python only scatters with it.

So **a reserve ruleset is still a 3-hour warm start**, not a 24-hour run. That is what turns
the reserve from a capacity into something worth having.

### Adding a power that uses the reserve

1. Add the variant to `PowerId`, with its token, rank, display name and text.
2. Declare what it uses: `needs_extended_encoder()`, plus `opens_phases()` and/or
   `status_flags_used()`. ⚠️ **This is the one silent failure in the reserve** — a power that
   writes a flag while declaring `false` writes into a tensor with no room for it. It is
   checked from three directions rather than trusted:
   `reserve_declaration_matches_what_each_power_uses` walks every variant and requires the
   declaration to be an `==` with what the power actually uses; the slot writer asserts its own
   width; and `Phase::needs_extended_encoder` has to agree.
3. Claim a status flag by name in `card.rs` (`STATUS_SHIELDED` is flag 0), never by a bare
   index.
4. Implement it in the rank's module, and give it a named test.

⚠️ One more place a mistake would be silent: `CHOOSE_LANE` is lane-indexed, so
`encode::action_permutation` must relabel it. A table that left it fixed is still a bijection
and still composes as S₃, so the structural tests would pass and only the *meaning* would be
wrong — the lane-equivariant network would route one lane's logit through another's weights.
`reserve_lane_permutations_relabel_choose_lane` asserts the block actually moves.

### The three worked examples

| Ruleset | Claims | Why it is the interesting shape |
|---|---|---|
| `seven-shield` | status flag 0 | The 7 shields rather than heals — a *prevention* effect, invisible until spent, which the base observation had nowhere to put. |
| `king-any-lane` | `CHOOSE_LANE` + a phase | Separates *having* a King from *having put it in the right lane* — a question the canonical rules cannot ask. |
| `two-choose` | `CHOOSE_OPTION` + a phase | Makes `game_rules.md` §10a's contested ruling an in-game decision, so a strong agent can answer it by how often it discards. |

---

## 8. Testing

`engine/tests/` holds ~150 rules tests named for their rule section, plus three suites that
exist specifically for the mod system.

**`rules_*.rs` pin the canonical ruleset.** They stopped being "the rules" and became "the
canonical ruleset's rules" — mechanically a no-op, since `Position::empty()` builds
`GameConfig::default()`.

**`rules_mods.rs` and `reserve.rs` hold the named tests for each variant**, per `CLAUDE.md`'s
rule that every ruling gets one. A mod is not in `game_rules.md`, so they are named
`mod_<ruleset>_<what it asserts>` and each states the canonical behaviour it departs from.

**`rulesets.rs` — 11 invariants × every registered file**, enumerated from `configs/rules/`.

| Invariant | Why it is structural |
|---|---|
| Determinism: same seed + config → identical game | |
| No agent reads hidden information | a mod must not open an information leak |
| Observation is a function of the information set | including any status a power sets |
| `legal_actions()` empty iff the game is over | a sub-decision with no answer is a hang |
| **Every game terminates without reaching `max_plies`** | see below |
| Card census: nothing lost or duplicated | |
| A player holding a card is never stuck | |
| Exactly two encoder layouts, and the base one is canonical | §7 |

At twenty rulesets this suite is the one thing that gets meaningfully slower, since it is a
cross product. Keep per-ruleset game counts low — these are structural properties, and a leak
or a non-termination shows up in tens of games, not thousands. If it does become slow, cut
games per ruleset before cutting rulesets.

⚠️ **A `PlyLimit` draw is a bug report, not a result.** `game_rules.md` §7 proves the game
finite from *specific rules* — powers fire on flip, a King reactivates once, nothing ever turns
a card face-down again. A variant that turns a card face-down, or lets a King reactivate a
King, **breaks that proof**. `max_plies` catches it as a logged draw, so the suite treats it as
a failure.

---

## 9. The part that actually costs money

A Tier 2 rule change is ~150 lines the first time and ~20 thereafter. Measuring one is a
training run. Optimise for that, not for lines of code.

```
edit a card module → warm-start from the current champion → probe + card value → FINDINGS → decide
```

Two things make it affordable, and both are already true:

- **Warm starting works across rulesets** (§1b), including across the reserve break (§7).
- **`--eval-batch` gave 3.26× on self-play for free** (`FINDINGS.md` F4.7), and self-play is
  53–91% of a generation.

**At twenty rulesets the binding constraint stops being expressiveness and becomes
measurement.** Twenty × a 3 h warm start is 60 hours of box time. So there is a screening tier
below "train an agent":

> **`duel52 screen`.** `greedy` and `ismcts` take no checkpoint, so they play any ruleset the
> day the file is written — no training, no warm start. Minutes, not hours. It reports draw
> rate, game length, lane concentration and per-rank flip *rate*.

It will not tell you a ruleset is *good* — search agents are not the meta. It reliably tells
you one is **broken**: degenerate, drawish, or containing a card nobody plays. At twenty
candidates that is most of the value, and it is the difference between 60 hours and a handful.
Keep `random` in the roster for scale.

---

## 10. What a card is worth — the other half of the instrument

Modularity lets you *make* a change. It does not tell you whether the change did what you
wanted. For that you need a way to say what a card is worth, which is `duel52 card-value`: hold
a `testkit` position fixed, vary one card's rank, and read the value head's delta in
win-probability points.

Without it the balance loop is "change the 3, train for three hours, see whether the win rate
against a frozen reference moved" — confounded, slow, and unable to distinguish "the 3 got
better" from "the meta shifted around it". With it, the loop reads the 3's value against the
twelve other cards, before and after.

⚠️ **Read the CONTROL line first, and read `in hand`.** An early version varied the card
**face-up in a lane** and produced a ranking whose top four were 8, J, 10, 9 — exactly
`PowerId::is_constant()`, in a block. That was not a fact about Duel 52; it was the method
reading its own selection criterion back out:

| power kind | ranks | what "face-up in a lane" measures |
|---|---|---|
| constant | 8, 9, 10, J | the power live and working — **100% of its value** |
| one-shot | A, 2, 4, 5, 6, 7, Q, K | a **spent** card; it fired on the flip and the effect is past |
| conditional | 3 | **nothing** — the Trap works only face-down |

A card **in hand** has its whole future ahead of it whatever kind of power it carries, so every
rank is measured at the same point in its life. `on board` is kept as the contrast, and `gap`
says whether a card's value is in the flip or in the body.

### The table

Measured on `lane-gen032`, 400 positions, canonical rules, in win-probability points.

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

**Spread: 7.33 points**, best to worst, every gap many standard errors wide.

**The check that the fix worked** is that `in hand` is *interleaved* by power kind while
`on board` is sorted by it. One-shots span the whole in-hand range with the four constants
scattered through it at positions 2, 6, 7 and 9. The board column still has all four constants
on top and nothing else above +0.54 — the original artifact, now isolated in a column labelled
as not comparable across kinds.

**The `gap` column is the useful by-product.** It says where a card's value lives: large and
positive (the Ace at +5.00, the Queen at +4.21) means the value is in the flip and the card is
a spent body afterwards; negative (the 8 at −3.34, the 10 at −3.09) means the value is the body
standing in the lane.

**The control.** Each rank is also substituted into the **opponent's** hand, which the observer
cannot see, so all thirteen tensors are bit-identical and the value head must return one
number. It does: spread `0.00000`. `engine/tests/cardvalue.rs` checks this against a *hash* of
the observation rather than against a network, which is strictly stronger — a real value head
could return the same number for two different tensors by luck.

⚠️ **Three limits, all real.**

1. **This measures gen032's value head, not Duel 52.**
2. **The sample is biased toward early, high-uncertainty positions.** Asking "what if I held an
   `R`" needs a copy of `R` somewhere the observer cannot see, so positions where all thirteen
   ranks are still holdable are ones where little has been revealed. 8.3% of sampled `split`
   positions survive.
3. **`mirrored` is out of reach entirely** — 0.06% survive, because §9b publishes the removed
   multiset and the observer then accounts for nearly their whole deck. `card-value` detects a
   thin sample and refuses to print rather than showing a table nobody can interpret.

⚠️ The flip-timing curve corroborates this at both ends — the 8 earliest-flipped and the Queen
latest — but that belongs to the **board** column, not to card value: the 8 is +4.01 on board
and the Queen −1.97, while *in hand* the ordering is nearly reversed. Flip timing is about when
a card is worth turning over, which is a board question.

---

## 11. What was built, and in what order

Kept because source comments cite step numbers. All of it landed on `ruleset-configs` over
2026-09-09/11.

| # | Step | Where it lives |
|---|---|---|
| 0 | Correct `PLAN.md` §5's claim about per-variant layouts | `PLAN.md` §5 |
| 1 | `rules_hash`: computed, stamped into `.d52nn` and `.d52sp`, checked everywhere a cross-ruleset number would read as a result | `config.rs`, `Shard::read`, `ladder`/`match`/`probe`/`card-value` |
| 2 | Tier 1 numeric fields — every magic constant in the powers, defaulted to today's value | eleven fields on `GameConfig` |
| 3 | `DamageSource` + the damage queue + `on_lethal_damage`. **Zero behaviour change** | `engine/src/damage.rs` |
| 4 | `PowerId` + one module per rank; `fire_power`'s 13-arm match gone | `engine/src/powers/` |
| 5 | Dotted keys, `include`, `configs/rules/` as the registry, `rules_file` on the Python side | `config.rs`, `train/config.py` |
| 6 | Cross-ruleset invariant suite | `engine/tests/rulesets.rs` |
| 7 | Per-card probe counters, generalised from `three_fates` | `probe.rs` |
| 8 | Screening harness | `duel52 screen` |
| 9 | First real mods — seven rulesets | `configs/rules/`, `engine/tests/rules_mods.rs` |
| 10 | The card value table | `engine/src/cardvalue.rs`, `duel52 card-value` |
| 11 | **The encoder reserve**, opt-in, plus the `nn widen` bridge | `encode.rs`, `engine/tests/reserve.rs`, `py/duel52/nn/__main__.py` |

Step 3 is the one that unlocked the category. Step 8 is the one that makes twenty rulesets
affordable at all. Step 11 was held back deliberately until §7's shape was decided, because a
reserve taken piecemeal buys nothing a reserve is for.
