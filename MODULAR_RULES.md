# Modular rules — an assessment

**Written 2026-09-09**, against `main` at `dbe34b4`, with the 32-core and 128-core runs still
in flight. Nothing here has been implemented. This is where the codebase stands with respect
to changing rules, what a per-card rules system would actually cost, and what I think you
should do.

**§12's four open questions were answered the same day and are now a decisions record.** The
four choices — named variants over parameters, ~20 rulesets, take the encoder reserve, and
cross-ruleset play as a hazard rather than an experiment — are folded into §5a, §5d, §6, §7, §9
and §11, each marked ✅ **DECIDED** at the point it applies.

**The question.** The project's endgame is not a strong bot — it is *understand the meta, then
rebalance the game*. Rebalancing means editing individual card powers and re-measuring. Can
this codebase absorb that, and what should the mechanism look like?

**The short answer.** The engine is in much better shape for this than the question implies —
rules logic is far more concentrated than 20k lines of Rust would suggest, and the neural
encoding is accidentally *rank-agnostic*, which is the single most valuable property you have
for this work and which nobody designed on purpose. The hard part is not the code. It is
three other things:

1. **Provenance.** Nothing in this project records which ruleset produced a number. Verified
   below: `lane-gen032`, trained on `split`, loads and plays under `--variant base` with no
   warning at all. That is already a live hazard with three variants. With twenty rulesets it
   is the dominant source of wrong conclusions, and it should be fixed *before* any modularity
   work, not after.
2. **One structural assumption.** Your own worked example — the 3 damages the card that killed
   it — breaks the engine's current model of damage, which flows strictly one way per action.
   That is a real refactor, and it is the gate on every "on death, do X" power.
3. **Cost per ruleset is a training run, not a diff.** A rule change is 30 lines of Rust. Knowing
   whether it helped needs an agent trained on it. Optimise for that, not for lines of code.

I recommend **against** engine copies, and **against** a data-driven effect DSL. I recommend a
three-tier scheme where most balance edits are config *numbers*, structural edits are
*swappable named power variants in Rust with one module per rank*, and genuinely new mechanics
are budgeted honestly as a layout break.

---

## 1. Where the codebase actually stands

### 1a. Rules logic is far more concentrated than it looks

`engine/src` is 20,353 lines. Production code that names a specific rank is **~30 lines across
six files**. Everything else is generic.

| File | Rank-specific production sites | What lives there |
|---|---:|---|
| [apply.rs](engine/src/apply.rs) | 15 | `fire_power`'s 13-arm match ([apply.rs:371](engine/src/apply.rs#L371)), the 10's split in `do_attack` ([apply.rs:173](engine/src/apply.rs#L173)), the 8/9 retaliate logic in `resolve_attack` ([apply.rs:242](engine/src/apply.rs#L242)), the 3's Trap in `damage_card` ([apply.rs:287](engine/src/apply.rs#L287)) |
| [state.rs](engine/src/state.rs) | 4 | Jack taunt in `legal_attack_targets` ([state.rs:462](engine/src/state.rs#L462)), the 9/J split blocks in `twinstrike_split_candidates` ([state.rs:498](engine/src/state.rs#L498)), the 9-vs-Jack multiplier in `attack_damage` ([state.rs:531](engine/src/state.rs#L531)) |
| [card.rs](engine/src/card.rs) | 1 | `max_hp` → `rank.face_up_max_hp()` ([card.rs:154](engine/src/card.rs#L154)) |
| [rank.rs](engine/src/rank.rs) | 5 tables | `face_up_max_hp`, `is_constant_power`, `is_king_reactivatable`, `power_name`, `power_text` |
| [display.rs](engine/src/display.rs) | 6 | `combat_notes` ([display.rs:568-582](engine/src/display.rs#L568-L582)), `power_reference` ([display.rs:816](engine/src/display.rs#L816)) — presentation only |
| [probe.rs](engine/src/probe.rs) | 4 | Trap instrumentation: `traps_sprung`, `three_fates`, `threes_face_down_at_end` |
| [menu.rs](engine/src/menu.rs) | **0** | Uses `rank.power_name()`/`power_text()` for labels; no rank constants |
| [legal.rs](engine/src/legal.rs) | **0** | One predicate call (`is_king_reactivatable`); otherwise written entirely against `state.rs` query helpers |

`legal.rs` having zero rank constants is the strongest single signal in the codebase. Legality
was built as *"ask `state.rs` what is targetable"*, so a rule that changes targeting is a change
to one helper, not to the enumerator. The module header says so explicitly and it turns out to
be true.

`damage_card` has exactly **two call sites**, both inside `resolve_attack`
([apply.rs:263](engine/src/apply.rs#L263), [apply.rs:270](engine/src/apply.rs#L270)). That is a
clean chokepoint, which matters a great deal for §3 below.

### 1b. The encoding is rank-agnostic, and that is worth more than everything else here

[encode.rs](engine/src/encode.rs)'s per-slot features are `occupied`, `rank_onehot`,
`rank_unknown`, `face_up`, `is_base`, `entered_as_base`, `damage_onehot`, `max_hp_onehot`,
`frozen`, `allowance_onehot`, `attacks_used_frac`, `can_attack_now`, `paired`, `is_mine`.

**There is not a single feature that says what a power does.** No `retaliates`, no `taunts`, no
`is_nimble`. The network is shown the rank one-hot and generic mechanical state, and it learned
the semantics of each rank in its weights.

Three consequences, all of them good:

- **Changing what a rank does never moves the observation layout.** The obs hash, the action
  hash, `obs_dim` and `action_dim` are all unchanged.
- **Therefore `--init-from` works across rulesets.** A rules-variant training run can warm-start
  from `lane-gen032` instead of starting from scratch. The network does not have to relearn the
  game; it has to relearn one rank. That is plausibly the difference between a 24-hour run and a
  3-hour one, and it is what makes an *iterative* balance loop affordable at all.
- **The clamped overflow buckets were built for exactly this.** `DAMAGE_BUCKETS = 4` when the
  live range is 0–2, with a comment saying it "exists so a future rule that raised hit points
  would degrade into a saturated feature rather than an out-of-bounds write."
  `ALLOWANCE_BUCKETS = 4` when nothing reaches 3. Someone already thought about this.

### 1c. The action space is more general than the powers that use it

The 1324/2194-wide policy head has six blocks, and two of them are already generic:

- `CHOOSE_SLOT (2·L·S)` — any slot, either side, any lane — serves **four** distinct
  sub-decisions: `Peek`, `ResolveNext`, `MoveHere`, `SplitTarget`.
- `CHOOSE_RANK (R)` — serves `GiveBack`.

**The Queen is the instructive case, because she looks like a lane choice and is not.**
`Action::MoveHere { lane, slot }` encodes as `choose_slot(Side::Mine, lane, slot)`
([encode.rs:679](engine/src/encode.rs#L679)) — the *destination* is fixed (the Queen's own lane,
carried on the `Pending::QueenSource` node), and the *source* is addressed as a **card**. The
lane index is part of that card's address, not a separate decision. "Move an allied card from
another lane" is "pick a card, excluding these" — the exclusion is a mask, and masks are free.

So the line is not "does a lane appear in the action". It is:

> **A lane needs its own block only when it is chosen with no card standing in for it** — a
> *destination* lane, or a lane that might be empty.

"Move an allied card into a lane of your choice" is the Queen inverted and it *does* need
`CHOOSE_LANE`, because the destination is not a card. "Freeze a lane of your choice" needs it
for the same reason. "Move an allied card from another lane" does not.

⚠️ **But the action space is the wrong thing to watch, and this is the correction that matters.**
Each sub-decision has its own `Phase`, `Phase` is a scalar one-hot in the *observation*, and
`PHASE_COUNT = 7` appears in `scalar_fields` and in `obs_layout_string`
([encode.rs:99](engine/src/encode.rs#L99), [encode.rs:128](engine/src/encode.rs#L128)). So:

| Change | action hash | **obs hash** |
|---|---|---|
| A power reusing an existing sub-decision (a King that also peeks; a 5 that also moves a card) | unchanged | **unchanged** |
| A genuinely new *kind* of sub-decision, even reusing `CHOOSE_SLOT` logits entirely | unchanged | **moves** — `PHASE_COUNT` 7 → 8 |
| A new decision needing `CHOOSE_LANE` / `CHOOSE_OPTION` | moves | moves |

**Any new sub-decision kind is a layout break, whether or not it needs new logits.** That is a
much tighter constraint than the action blocks, and it is the one to design around. It also has
a much cheaper fix — see §7.

### 1d. The config system is close, and its shape is right

[GameConfig](engine/src/config.rs) is `Copy + PartialEq`, is cloned into every `GameState`, is
serialized by `to_config_string()`, and that string is written verbatim into **both** the
`.d52sp` shard header and the `.jsonl` game record. `from_config_str` **rejects unknown keys**,
so a typo cannot silently change what was measured.

That last property is the reason the config is the right home for rules. A record whose config
string fully determines the ruleset replays honestly forever; `record.rs`'s `walk` *verifies*
rather than decodes, so a record that stops reproducing is refused rather than silently
reinterpreted. Put rules anywhere else — a compile-time feature, an environment variable, a
separate file — and that guarantee evaporates.

Backward compatibility is favourable: adding keys means old artifacts (missing the key) still
parse under a new build with defaults filled in, while new artifacts fail loudly on an old
build. One-directional, and in the safe direction.

The parser is 40 lines of hand-rolled `key = value` with `[section]` headers ignored. It handles
dotted keys with no change at all, since it just lowercases the left side of the `=`.

### 1e. What is already broken — the provenance hole

**Verified, not asserted.** All three variants produce identical layout hashes:

```
$ .venv/bin/python -c "from duel52 import _engine; ..."
base     obs_dim 4290  action_dim 2194  obs b1355a841a1fdc4a  action 5169f9461d627b39
split    obs_dim 4290  action_dim 2194  obs b1355a841a1fdc4a  action 5169f9461d627b39
mirrored obs_dim 4290  action_dim 2194  obs b1355a841a1fdc4a  action 5169f9461d627b39
```

And the consequence, run on this laptop today:

```
$ ./target/release/duel52 match --a netmcts:models/duel52-split-lane-gen032.d52nn@16 \
    --b random --games 4 --seed 1 --encoding-slots 21 --variant base
netmcts:models/duel52-split-lane-gen032.d52nn@16 vs random — 4 games
  config: variant=base two_power=bottom stalemate=20plies
  score for ...gen032...: 1.0000 +/- 0.0000 (95% CI) — W4 L0 D0
```

A checkpoint trained on the split deck played the rules-as-written game, at full speed, with no
warning. The printed `config:` line describes the *runtime* config, not the checkpoint's
training config, so nothing on screen says anything is wrong.

Two things follow.

**⚠️ `PLAN.md` §5 states something false.** It says: *"The observation layout is per variant, so a
checkpoint cannot be loaded against a variant it was not trained on"*, and uses that as the
reason the variant comparison costs a training run per variant. The **conclusion** survives — a
fair comparison does need a run per variant, because the agent must be trained on the game it is
judged in — but the stated **mechanism** does not exist. That matters, because the wrong
mechanism describes a guard rail that isn't there. Worth correcting in `PLAN.md` regardless of
what you decide about modularity.

**The `.d52sp` shard is the one artifact that gets this right**, and only half-right. It embeds
the full config text and replays each game under *that* config, so a shard from ruleset A is
internally honest. But nothing checks the shard's config against the *run's* config, so two
rulesets can be mixed into one replay buffer and trained as if they were one game. The
`Shard::read` layout check compares against the shard's own config, which is always self-
consistent and therefore cannot catch this.

The `.d52nn` checkpoint gets it wrong outright: its header carries `obs_dim, action_dim, width,
blocks, value_hidden, obs_layout_hash, action_layout_hash, param_order` and **nothing about the
game it was trained on**.

---

## 2. Classify the change before you cost it — the three tiers

The single most useful thing I can give you is not an architecture, it is a way to price an idea
in about ten seconds. Three tiers, differing in cost by roughly two orders of magnitude.

### Tier 1 — numbers. A config field, one line at one use site.

No structural change, no new hooks, no layout break, warm start works. Roughly half of the
plausible balance questions live here, and **none of these fields exist today** — they are all
magic constants inside `apply.rs`, `state.rs` and `rank.rs`.

| Question | Field | Today's site |
|---|---|---|
| Is the Jack's 3 HP right? | `jack_hp` | [rank.rs:93](engine/src/rank.rs#L93) |
| Is retaliate 1 damage enough? | `eight_retaliate_damage` | [apply.rs:270](engine/src/apply.rs#L270) |
| Should the 9's Jack bonus be ×2? | `nine_vs_jack_multiplier` | [state.rs:531](engine/src/state.rs#L531) |
| Should the Ace give one action or two? | `ace_bonus_actions` | [apply.rs:391](engine/src/apply.rs#L391) |
| Should a fresh Ace attack twice or three times? | `ace_attack_allowance` | [apply.rs:394](engine/src/apply.rs#L394) |
| Should a 6 freeze for one turn or two? | `six_freeze_turns` | [apply.rs:472](engine/src/apply.rs#L472) |
| Should a 7 heal 2 or fully? | `seven_heal_amount` | [apply.rs:488](engine/src/apply.rs#L488) |
| Is 2 HP the right baseline? | `default_hp` | [card.rs:154](engine/src/card.rs#L154) |
| Should a pair deal 2 or 3? | `pair_damage` | [state.rs:532](engine/src/state.rs#L532), [apply.rs:199](engine/src/apply.rs#L199) |
| Should a 4 peek once or twice? | `four_peeks` | [apply.rs:424](engine/src/apply.rs#L424) |

⚠️ Two of these are only Tier 1 if the encoder's clamped buckets absorb them. `jack_hp = 4` or
`default_hp = 3` pushes `damage_onehot` past its live range of 0–2 — which the 4th bucket
absorbs *without moving the layout*, but at the cost of the network seeing "3 or more damage"
as one undifferentiated state. `jack_hp = 4` also needs a third `MAX_HP_BUCKETS` slot, which
**does** move the layout. HP changes are Tier 1 downward and Tier 3 upward; check the buckets
before assuming.

### Tier 2 — swappable power shapes. A named variant in Rust, selected by config.

Structural change to what a power *does*, but reusing existing state fields, existing
sub-decision kinds, and the existing action blocks. Warm start still works, no layout break.

- The 3 damages its killer, or returns face-down, or returns to hand, or does nothing.
- The 6 can freeze 9s. The 6 freezes the whole lane including allies.
- The Queen moves an **enemy** card — `CHOOSE_SLOT` already spans both sides, and
  `Phase::QueenSource` already exists, so this is a field on `Action::MoveHere` and nothing more.
- The King reactivates constant powers too, or reactivates other Kings once.
- The 10 splits **across lanes** — same reason: the logits for every lane already exist, and
  `Phase::SplitTarget` already exists.
- The Jack taunts only face-up attackers.
- The 2 bottoms two cards. (You already have `two_power: bottom | discard` — this tier is that
  idea, generalised to all thirteen ranks.)

This is the tier the modular system is *for*, and it is the tier your worked example sits in.

⚠️ **The boundary is subtle, and two things I first put in this list belong in Tier 3.** A Queen
that moves a card **out** to a lane of your choice needs `CHOOSE_LANE`, because the destination
is not a card (§1c). A Jack that absorbs **one attack per turn** needs a new per-card counter —
`attacks_used` counts attacks *made*, not absorbed. Both read like small variations on an
existing power and both are layout breaks. Check the phase and the state fields before pricing
anything.

### Tier 3 — new mechanics. A layout break: everything regenerates, no warm start.

Anything needing a new per-card state field (a shield counter, a poison marker, a "cannot be
healed" flag) or a new **kind** of sub-decision — which means a new `Phase`, and therefore a
layout break *even if it reuses existing action logits*. See §1c; this is easy to get wrong,
because the action space looks like the constraint and isn't.

Cost: `obs_layout_hash` and/or `action_layout_hash` move → every `.d52nn` refused, every
`.d52sp` refused, `--init-from` impossible, and the ruleset needs a **from-scratch** run to be
measured. On rented cores that is your 24-hour budget, spent.

**Do not do Tier 3 casually, and do not do it one change at a time.** Batch every Tier 3 idea
you want and take one layout break for all of them — see §6.

---

## 3. Your worked example, priced honestly

> *"Suppose I want to make the 3 not only flip when killed face down but also deal a damage to
> the attacker that killed it."*

This is Tier 2, and it is a good example precisely because it does not fit today.

**What the engine assumes now.** Not "damage flows one way" — the 8 is a back-edge and it works
fine. What the engine has is a **hardcoded two-step pipeline of fixed depth**:

```
do_attack  →  collect targets  →  resolve_attack
                                    ├─ read the set of retaliating 8s  ← before any damage
                                    ├─ damage each target              ← step 1
                                    └─ pay out retaliate to attackers  ← step 2
                                  done. Nothing reads the board again.
```

**Why depth 2 is provably enough today**, and it takes two independent facts:

1. **Retaliate cannot trigger retaliate.** §6's trigger is *"any card that **attacks** this 8"*,
   and retaliate damage is not an attack. The rule is self-limiting; an 8 hit by retaliate does
   not retaliate back.
2. **Retaliate cannot trigger the Trap.** The 3 springs only while **face-down**, and only
   **face-up** cards attack ([card.rs:192](engine/src/card.rs#L192)). Retaliate damage lands
   only on `attackers`, so it can never reach a face-down card. Nothing ever turns a card
   face-down again (§7), so this holds for the whole game.

Put together: **retaliate damage can kill but can never *trigger* anything**
(`rule_6_retaliate_can_kill_the_attacker`). Step 2 provably cannot generate a step 3, so
hardcoding the pipeline at depth 2 is correct — not by construction, but by an accident of
which two triggers happen to exist.

`damage_card(id, amount)` **does not know who dealt the damage**. It cannot: it is called from
the target loop and from the retaliate loop, and neither passes a source. The 3's Trap fires
inside it ([apply.rs:303-318](engine/src/apply.rs#L303-L318)) with no idea what killed the 3.

**What the change needs.**

1. `damage_card(id, amount)` → `damage_card(id, amount, source: DamageSource)`, where
   `DamageSource` is `Attack { attackers: Vec<CardId> }`, `Retaliate { from: CardId }`,
   `Power { rank, card }`, or `Vengeance { from: CardId }`. Two call sites.
2. Death becomes a hook with a payload: `on_lethal_damage(card, source) -> DeathResponse`,
   where today's Trap is `DeathResponse::Survive { face_up: true, clear_damage: true }` and the
   new one is that plus `Then::Damage(source.attackers, 1)`.
3. **Damage becomes a queue, not a call stack.** This is the real work, and the reason is
   *order*, not depth — see §3a.

   Recommendation: a `VecDeque<PendingDamage>` drained to fixpoint inside one action, with a
   hard depth/iteration cap that panics in debug and logs in release — the same shape as the
   existing `pending` sub-decision stack and the same shape as `max_plies`, which exists
   precisely so a rules bug degrades into a logged artifact rather than a hang.

**What does *not* change:** `legal.rs`, `encode.rs`, the action space, the observation layout,
the checkpoint format, `menu.rs`, `display.rs`'s board rendering. And because the layout is
untouched, `--init-from models/duel52-split-lane-gen032.d52nn` works.

**Total:** roughly 150 lines of engine change (most of it the damage queue, which is a
one-time cost paid on behalf of every future death trigger), plus named tests, plus one
warm-started training run to find out whether the new 3 is any good.

**The thing to take from this:** the first mod you named is not the cheap one. It is the one
that pays for the infrastructure. Every subsequent "on death, do X" is then nearly free.

### 3a. What vengeance actually breaks — order, then depth

Vengeance hits the **attacker**, and attackers are always face-up, so a face-up 3 has no Trap
to spring. By the same two facts that bound retaliate, *this specific mod also stays
depth-bounded.* It is worth being precise about that rather than telling a scare story.

**What it breaks is ordering, and the engine has already been here.** [apply.rs:186](engine/src/apply.rs#L186)
collects both halves of a 10's twinstrike before any damage lands, with the comment:

> *"Both targets are collected before any damage lands, so the two halves of the split are
> simultaneous and **retaliate has no ambiguous ordering**."*

A death trigger reintroduces exactly the ambiguity that comment was written to remove. Take a
10 on 1 damage, twinstriking two face-down 3s that are each on 1 damage:

| | today | with vengeance |
|---|---|---|
| `damage_card(3a, 1)` | 3a springs face-up | 3a springs, **deals 1 to the 10 → the 10 dies** |
| `damage_card(3b, 1)` | 3b springs face-up | 3b springs, its vengeance finds no card and is **silently lost** |

Whether 3b's vengeance should land is a real ruling with no answer in `game_rules.md`, and
today the engine would answer it as a byproduct of `for &(id, amount) in hits` iterating in
order. §5 already insists the two halves are simultaneous; a `VecDeque` drained to fixpoint is
what makes the *consequences* simultaneous too, instead of sequenced by a loop.

**Depth breaks on the second death-trigger power, not the first.** Each of these reaches a
face-down card and chains:

- A Trap variant that returns the 3 **face-down** — now it can re-trigger, and the bound is gone.
- A death effect that hits the **lane** rather than the killer — face-down 3s are in range.
- A death trigger on any other rank (a Jack that novas the lane on death) — same reason.

So the queue is not paranoia about your example. It is the thing that lets the example be
*specified* rather than emergent, and the thing that stops the second one being a rewrite.

### 3b. The 8 is the better argument for card modules than the 3 is

Look at what [`resolve_attack`](engine/src/apply.rs#L242) actually holds. **Four rulings, from
three different cards, in thirty lines, none of them named:**

| Ruling | How it is expressed |
|---|---|
| §6 — an 8 retaliates for 1 | `retaliations` counted from `hits` |
| §6 — an 8 that *dies* to the attack still retaliates | the count is read **before** the damage loop |
| §5 — a 9 takes no retaliate damage, pair or not | `attacker_rank != Some(Rank::NINE)`, in the same `if` |
| §5 — a pair takes retaliate on **both** members | implicit in `for &id in &attackers` |
| §6 **[ASSUMED]** — a 10 hitting two 8s takes **2** | *emergent from `retaliations` being a `.count()` rather than a `bool`* |

That last row is the one to sit with. A ruling the rules do not address, flagged `[ASSUMED]` in
two places and covered by `rule_6_assumed_twinstrike_into_two_eights_takes_two_retaliate`,
exists in the engine only because someone wrote `.count()` instead of `.any()`. It is correct,
it is tested, and it is **stated nowhere in the code**.

Under §5b's design, `eight.rs` owns `after_attacked` and returns how much it retaliates,
`nine.rs` owns `veto_status_damage`, and the multi-8 stacking rule becomes an explicit decision
in one place with a name. That is the actual payoff of card modules — not "changing the 3 is
easier", but **rulings stop being emergent from control flow.**

---

## 4. What I recommend against, and why

### Engine copies per ruleset — no

You raised this and dismissed it, correctly, but the reasons are worth stating because they are
sharper than "it seems far from ideal":

- **It destroys the comparison, which is the entire deliverable.** `ladder`, `match`, `probe`
  and the Elo fit all assume one engine. Two agents from two engine copies cannot meet. You
  would be able to build variants and unable to compare them, which is the opposite of what the
  balance goal needs.
- **354 tests × N copies, with no shared guarantee.** The invariants that make this engine
  trustworthy — determinism, no-hidden-information (`phase2_no_agent_reads_hidden_information`),
  `legal_actions()` empty iff over, card census — would be asserted N times independently and
  drift N ways.
- **`record.rs`'s central promise dies.** A record "verifies rather than decodes"; a record from
  copy A replayed by copy B is refused, or worse, silently reinterpreted.
- **The divergence is ~30 lines.** Forking 20,000 lines to vary 30 is a bad trade by three
  orders of magnitude.

There *is* a legitimate worry underneath it: rules churn while `lane-gen032` is the reference
and two runs are in flight. The answer to that is not a fork — it is a **named, frozen default
ruleset** and an explicit opt-in for everything else. That is what §5 buys.

### A data-driven effect DSL — also no, and this one is less obvious

The instinct with a card game is to build an effect language: keywords, triggers, targets,
`{"trigger": "on_death", "effect": "damage", "target": "killer", "amount": 1}`. It is the right
call for Hearthstone or Magic, where hundreds of cards share a few dozen keywords.

Duel 52 has **thirteen cards and essentially thirteen distinct mechanics.** Deriving the hook
surface from the existing powers:

| Hook | Used by |
|---|---|
| `on_flip` | A, 2, 4, 5, 6, 7, Q, K |
| `on_reactivate` | same set (the King) |
| `on_lethal_damage` | 3 |
| `modify_outgoing_damage` | 9 (vs Jack) |
| `attack_spread` | 10 |
| `veto_spread_target` | 9 (personal), J (confinement) — **and §8 explicitly forbids unifying these** |
| `after_attacked` | 8 |
| `restrict_lane_targets` | J |
| `max_hp` | J |
| `veto_status` | 9 (cannot be frozen) |
| `veto_status_damage` | 9 (no retaliate) |

Eleven hooks for thirteen cards. The reuse ratio is close to 1. Building an interpreter with
eleven opcodes and thirteen programs is strictly worse than writing thirteen functions: you pay
for a parser, a validator, an evaluator and a debugging story, and you get back exactly the
flexibility you already had in Rust — minus the exhaustive match, minus type safety, minus the
compiler telling you which card you forgot when you add a hook.

`game_rules.md` §8 makes the point better than I can: *"Do not unify these into one 'blocker'
concept in the engine; they are different mechanics that happen to share a symptom in the
one-card case."* That warning generalises. The powers in this game resist abstraction, and the
document already knows it.

---

## 5. What I recommend — the design

### 5a. Rules live in `GameConfig`, as data, selecting code

```rust
pub struct GameConfig {
    // ... existing fields ...

    /// One power variant per rank. Index by `Rank::index()`.
    pub powers: [PowerId; Rank::COUNT],

    /// Turn-level actions — §5e. Not card powers, so `powers` has nowhere to put them.
    pub peek_action: PeekMode,

    /// Tier-1 numeric knobs, named individually.
    pub jack_hp: u8,
    pub default_hp: u8,
    pub eight_retaliate_damage: u8,
    pub nine_vs_jack_multiplier: u8,
    pub ace_bonus_actions: u32,
    pub ace_attack_allowance: u8,
    pub six_freeze_turns: u32,
    pub seven_heal_amount: u8,
    pub pair_damage: u8,
}

/// Every implemented power. Adding a variant is a code change — deliberately.
///
/// **No variant carries data.** A number that varies *within* a shape gets its own name,
/// never a config field. See "Named variants carry their own numbers" below.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PowerId {
    AceAction,
    TwoView,
    ThreeTrap,            // the current rule
    ThreeTrapVengeance1,  // your example, 1 damage
    ThreeTrapVengeance2,  // the same shape, 2 damage
    ThreeNone,            // ablation: what is the Trap actually worth?
    // ...
}
```

**Why an enum and not a trait object.** `GameConfig` is `Copy + PartialEq` and is serialized into
every shard and every game record. `[PowerId; 13]` preserves all three properties for free;
`[&'static dyn CardRules; 13]` preserves none of them. Dispatch is a `match` on a small enum,
so there is no dynamic dispatch in the search hot path, and `GameState` stays
`Clone + PartialEq`, which matters because determinization clones states constantly.

**Why adding a variant is a code change and that is correct.** You are not building
user-generated content; you are running experiments. `CLAUDE.md` already requires every ruling
to get a named test, so every power variant is a code change regardless. In exchange you get
the compiler's exhaustive match: add a hook and it tells you which of the thirteen cards you
have not handled. That is the safety property a DSL cannot give you.

**Named variants carry their own numbers — no per-variant config fields.** ✅ **DECIDED** (§12
Q1). `powers.three = "trap_vengeance_two_damage"` rather than `powers.three = "trap_vengeance"`
plus `three_vengeance_damage = 2`. That keeps `PowerId` a plain C-like enum with no associated data,
which is what preserves `Copy + PartialEq` for free, keeps `to_config_string()` at one name per
rank, and reduces `rules_hash` to a hash of thirteen strings plus the Tier-1 numbers. Nothing in
the hand-rolled parser has to grow.

The better argument is not simplicity, though. **It makes an inapplicable setting
unrepresentable.** `from_config_str` rejects unknown keys, but it has no way to reject a key that
is *known and irrelevant*: `three_vengeance_damage = 2` sitting in a config where
`powers.three = "trap"` parses clean, hashes into `rules_hash`, and does nothing. That is a silent
provenance bug of exactly the kind §6 exists to prevent, and the named form cannot express it.

Two consequences to hold on to:

- **Keep the count additive.** A named variant encodes a *shape*. Two orthogonal knobs on one
  power multiply — vengeance damage `1|2` × fires-face-up-too `yes|no` is four names, not three.
  A genuine cross-product is the signal to promote one axis to a Tier-1 numeric field, not to
  keep enumerating.
- **The Tier-1 numeric fields above still stand.** They parameterise powers that exist in the
  *canonical* ruleset, which is what Tier 1 is for. The rule is only that a **new Tier-2 variant
  gets a name, not a knob**. The heuristic is not airtight — `eight_retaliate_damage` does go
  inert under `eight = "none"` — so see §6 for the one nuisance that causes.

### 5b. One module per rank

```
engine/src/powers/
    mod.rs      — the hook traits, the dispatch match, the default no-ops
    ace.rs      — every Ace variant, with its tests
    two.rs
    three.rs    — ThreeTrap, ThreeTrapVengeance1, ThreeTrapVengeance2, ThreeNone, + tests
    ...
```

`apply.rs`'s `fire_power` becomes a dispatch to `powers::on_flip(config.powers[rank], ctx)`,
and the 13-arm match moves out of the turn machinery into the place where it belongs. This is
what delivers the property you asked for: *changing the 3 touches `three.rs` and nothing else.*

`rank.rs`'s five tables (`power_name`, `power_text`, `face_up_max_hp`, `is_constant_power`,
`is_king_reactivatable`) move onto `PowerId`, which is also what makes `duel52 powers` and the
CLI's teaching text correct under a modified ruleset instead of quietly describing the default.

### 5c. Do the damage refactor first, as a no-op

`DamageSource` + the damage queue + `on_lethal_damage` as a hook, with the current 3 as the only
implementation and **zero behaviour change**. The 354 existing tests are the proof it landed
correctly. Then, and only then, add `ThreeTrapVengeance1` — which becomes a 20-line diff in one
file with three new tests.

Doing it in the other order means debugging a new rule and a refactor at the same time, against
a test suite that cannot tell you which one broke.

### 5d. Config format and ruleset composition

✅ **DECIDED: build this properly** (§12 Q2 — you expect up to twenty rulesets). At three or four
this machinery would not have earned its keep; at twenty it does.

Flat dotted keys, which the existing parser handles unchanged:

```toml
# configs/rules/three-vengeance.toml
powers.three = "trap_vengeance_one_damage"
```

For "variants are collections of card configs", add a single `include` directive with
last-wins ordering and a cycle guard:

```toml
# configs/rules/experiment-04.toml
variant = "split"
include = "rules/three-vengeance.toml"
include = "rules/six-freezes-nines.toml"
jack_hp = 4                              # explicit keys win over includes
```

**A ruleset is a file in `configs/rules/`, and that directory is the registry.** This is the
part that only matters at twenty. Nothing should have to be *added to a list* for a ruleset to
count as registered — §8's cross-ruleset invariant suite enumerates the directory, so a new
ruleset is covered by every structural invariant the moment the file exists. If registering a
ruleset requires remembering to also register its test, at twenty you will forget, and the one
you forget is the one that quietly fails to terminate inside a 24-hour run.

Keep `to_config_string()` round-tripping **everything**, including resolved includes — it should
emit the fully-resolved flat form, not the include lines, because that string is what goes into
the shard and the record and it has to be self-contained. `config_files_round_trip` already
tests this and will catch a missed field.

### 5e. Turn-level actions are a second axis, and `[PowerId; 13]` does not cover them

Everything above assumes a rule change is a *card* change. Not all of them are. The worked
example that exposed this: **make peek a §4 action** — spend one of your three actions to look at
any face-down card — and give the 4 some other power entirely.

That is not a power on a card. It is a fifth entry in `game_rules.md` §4's action list, and it
needs its own config axis (`peek_action = off | any_face_down | unknown_only`) and its own branch
in `legal_main_actions`. `config.powers` has nowhere to put it.

It is worth pricing because it comes out **Tier 2 — the cheapest structural change available**,
which is not the intuition:

- **The action space already has it, and Main is the one free slot.** `decode_action`'s
  `CHOOSE_SLOT` branch dispatches on `state.phase()`
  ([encode.rs:750-772](engine/src/encode.rs#L750-L772)) — the same logits mean `Peek` under
  Foresight, `ResolveNext` under ResolveOrder, `MoveHere` under QueenSource, `SplitTarget` under
  SplitTarget. `Phase::Main` is the only phase that does not claim the block; it falls through to
  `_ => None`. Adding a `Phase::Main` arm fills the last empty multiplexer slot. Zero new logits.
- **No new phase, so `PHASE_COUNT` does not move** — the decision happens *in* Main rather than on
  a new pending node, which is exactly the §1c trap and it misses. ⚠️ When the 4 loses Foresight,
  **do not delete `Phase::Foresight` from the enum.** Leaving a dead variant costs nothing;
  deleting it takes `PHASE_COUNT` 7 → 6 and breaks the observation hash for no benefit.
- **No new state.** `do_peek` is `known_to |= me.bit()` ([apply.rs:577](engine/src/apply.rs#L577)),
  and the observation already reads exactly that — `card.rank_known_to(observer)`
  ([encode.rs:359](engine/src/encode.rs#L359)). A peeked-but-face-down card already encodes.
- **The enumerator is already phase-independent.** `legal_peeks`
  ([legal.rs:174-188](engine/src/legal.rs#L174-L188)) reads `face_down_cards()` and `to_move` and
  nothing off the Foresight node, so `legal_main_actions` can call it unchanged.

Two mechanical wrinkles, one of which is a simplification. `do_peek` ends with `self.pending.pop()`
([apply.rs:578](engine/src/apply.rs#L578)) and must not, in Main. And `costs_an_action()`
([action.rs:113](engine/src/action.rs#L113)) is a pure function of the variant, so it cannot say
"free as the 4's power, one action in Main" — but `legal_actions` returns `legal_main_actions()`
**iff** `pending.last()` is `None` ([legal.rs:35-37](engine/src/legal.rs#L35-L37)), so
`pending.is_empty()` ⟺ main phase ⟺ costs an action, exactly. Replacing the four-variant match
with that check in `dispatch` makes the rule what §4 actually says — a sub-decision is free, a
main action costs one — rather than a hand-maintained list, and lets a ruleset carry both
Foresight and the peek action with no ambiguity.

⚠️ **The real hazard is that this reintroduces the pass.** `face_down_cards()` does not filter on
`known_to` ([state.rs:545](engine/src/state.rs#L545)). Harmless today, since Foresight is free and
mandatory so a redundant peek merely wastes a power. As a *main action* it is a pass: spend an
action, learn nothing, change nothing. That lands on something load-bearing — there is no `PASS`
block in the policy head, deliberately, and F2.4b's 0.7–1.7% → **0 stalemates in 4,000 games per
variant** came from making actions mandatory and removing the standoff. One predicate closes it
(restrict to cards not already `known_to` the actor), which is why the config axis above has
`unknown_only` as a distinct setting rather than a bug fix: whether an action may be wasted is
itself worth measuring.

**So the design needs a `TurnActions` axis alongside `[PowerId; 13]`.** It is small — a handful of
config fields and their branches in `legal_main_actions` — but it is a category the per-card table
does not contain, and the peek idea will not be the only member.

⚠️ **One Python change is required and is easy to miss.**
[py/duel52/train/config.py](py/duel52/train/config.py)'s `GameSettings.cli_flags()` emits
`--variant`, and the CLI errors on `--config` and `--variant` together
([duel52.rs:442](engine/src/bin/duel52.rs#L442)). `GameSettings` needs an optional
`rules_file` that emits `--config` *instead of* `--variant`. Until that lands, no training run
can use a custom ruleset, however good the engine side is.

---

## 6. Provenance — do this before anything else

This is the recommendation I feel most strongly about, and it is independent of every design
choice above. It is worth doing even if you never build the modular system.

**Add `rules_hash` to `GameConfig`**: an FNV of the fully-resolved config string restricted to
the fields that change the *game* (variant, powers, the Tier-1 numbers, deck composition, turn
structure, `two_power`) and excluding the ones that do not (`stalemate_value`, which is a
learning weight; `encoding_slots`, which is already covered by the layout hash). Compute it in
one place, the way `encode.rs` owns the layout hashes and for the same reason.

Then stamp and check it:

| Artifact | Change |
|---|---|
| `.d52nn` header | Add `rules_hash`. ⚠️ **`Weights::load` must NOT refuse on it** — see below. |
| `.d52sp` header | Already carries the config text. Add the hash, and have the training loop refuse to mix rulesets in one replay buffer. |
| `.jsonl` record | Already carries the full config. Nothing to do — this one is already correct. |
| `ladder` / `match` / `probe` output | Print the ruleset name and hash in the header, next to the existing `config:` line. |
| `FINDINGS.md` | Every entry names its ruleset. An unreproducible finding is not a finding, and "which rules" is now part of reproducing it. |

**No escape flag — but the check is three checks, not one.** ✅ **DECIDED** (§12 Q4): cross-ruleset
play is purely a hazard, never an experiment you want to run deliberately. That removes
`--allow-cross-rules`, which is simpler and safer. But a *blanket* hard error would be wrong,
because there are three different operations here and only two of them are hazards:

| Operation | Cross-ruleset? | Behaviour |
|---|---|---|
| A `.d52sp` shard into a replay buffer | Silent corruption, no human present | **Hard error, no escape** |
| `ladder` / `match` / `probe` | A human reads the number as a result | **Hard error, no escape** |
| Generation 1 of a warm-started run | **Structural and intended** | Print loudly, continue |

The third row is the one that cannot be an error, and §9's entire affordability argument rests
on it. Warm-starting `lane-gen032` into a new ruleset *is* a net trained under ruleset A playing
under ruleset B — including inside that run's own gate, where at generation 1 the incumbent is
the foreign net by construction. Refuse it and the loop costs 24 h per ruleset instead of 3 h,
twenty times over.

So: **`rules_hash` must not be checked in `Weights::load` the way the layout hashes are.** Put the
hard error on `Shard::read`, which already has exactly this shape for the layout hashes, and on
the three measurement CLIs. The training loop prints the mismatch rather than refusing it.
`--init-from` is unaffected either way — it is a Python-side header read, already a separate path
from `Weights::load`.

⚠️ **One nuisance from §5a's decision.** A Tier-1 numeric field that goes inert under a variant
swap — `eight_retaliate_damage` when `eight = "none"` — still contributes to `rules_hash`, so two
rulesets that are behaviourally identical can hash differently. That is a labelling annoyance,
not a correctness bug: it can make you think you have two rulesets when you have one, never the
reverse. Live with it, or normalise inert fields to their default before hashing. Do **not** fix
it by dropping fields from the hash on a judgment call about relevance — that direction turns a
nuisance into the silent-collision bug this whole section exists to prevent.

**Name the current ruleset and freeze it.** Something like `rules = "canonical-2026-09"`, so
every existing number in `FINDINGS.md` retroactively acquires a ruleset label without anybody
re-deriving what it was measured under.

---

## 7. The encoder reserve — decided, with an ordering constraint

Tier 3 costs a layout break, and a break is only cheap at a from-scratch run. ✅ **DECIDED**
(§12 Q3): from-scratch runs are acceptable, so **take the reserve.**

That settles *what*, but it changes *when* into the thing to be careful about. Willingness to do
a from-scratch run makes each break affordable, not free — it is still 24 h of rented cores. So
the move is **not** "break whenever a Tier-3 mod comes up". It is:

> **One deliberate layout revision that takes the full reserve *plus* every concrete Tier-3
> change already on the table.** Then that layout stays canonical for a long time, and the
> twenty rulesets of §5d are all Tier 1 and Tier 2 against it.

Batching is the whole value. Reserving capacity buys many Tier-3 changes for one break instead of
one break per change; taking the break without batching buys nothing the reserve was for.

⚠️ **This is a hard ordering constraint, not a preference.** The reserve has to land *before* the
next from-scratch run starts, or it waits for the one after. That makes step 11 in §11 a
scheduled item with a real predecessor, not a decision to revisit later.

The question is whether to reserve generic capacity in the observation so that common Tier-3
mods become Tier 2. Concretely: a `status_flags` block of 4 unnamed per-slot booleans, and one
generic per-slot counter. Most plausible mechanics are boolean statuses — shielded, poisoned,
marked, cannot-be-healed, cannot-attack, stunned — and a card module could claim a flag without
touching the layout.

The cost is smaller than it looks:

| | slot features | board floats | obs_dim @ S=21 | vs today |
|---|---:|---:|---:|---:|
| today | 33 | 4158 | 4290 | — |
| +4 flags | 37 | 4662 | 4794 | +11.7% |
| +4 flags +4 counter buckets | 41 | 5166 | 5298 | +23.5% |

And **the search-path cost is far below that**, because the input layer walks only the
observation's non-zeros ([mlp.rs:437](engine/src/nn/mlp.rs#L437), `FINDINGS.md` F3.3: 205 of
4290 features). Reserve flags are zero almost always, so they add one branch test each per
evaluation and skip the `width`-long inner loop entirely. What you actually pay is weights and
memory in the input projection, which F3.3 puts at 58% of a flat checkpoint — so +11.7% on the
observation is roughly +7% on parameters.

**But the status flags are the *third* priority, not the first.** §1c is the reason: the binding
constraint on new powers is `PHASE_COUNT`, not slot features and not action logits. Ordered by
value for money:

| # | Reserve | Cost | Unlocks |
|---|---|---:|---|
| 1 | **`PHASE_COUNT = 12`**, 5 spare one-hot slots | **5 floats of 4290 — 0.1%** | *Any* new kind of sub-decision. This is the constraint that actually bites. |
| 2 | `CHOOSE_LANE (2·L)` + `CHOOSE_OPTION (k=4)` action blocks | 10 logits of 2194 — 0.5% | Destination lanes, yes/no choices, modal powers |
| 3 | 4 per-slot `status_flags` | 504 floats — **+11.7%** | New per-card statuses (shielded, poisoned, marked) |

Item 1 needs **no enum change at all** — bump the constant and leave `phase_index`'s seven arms
alone; the extra one-hot positions are simply always zero until something claims one. It is the
cheapest structural option in the entire codebase and I had it ranked below the flags, which was
wrong.

**Take all three.** Items 1 and 2 were never in doubt — 0.6% of the observation between them,
and they move an entire category of powers from Tier 3 to Tier 2. Item 3 was the judgment call,
and with a from-scratch run available it resolves in favour of taking it, because **11.7% is the
wrong number to be weighing.** That figure is `obs_dim` growth, which is parameters and memory
(~+7%, per the paragraph above). Self-play does not pay it: the input layer walks non-zeros, the
trunk is what self-play actually costs, and the trunk width is unchanged. Status flags are zero
by nature — a card is not shielded, not poisoned, not marked — so they add almost nothing to the
non-zero count that the search path is actually proportional to.

Per `CLAUDE.md`'s own rule this deserves a measurement rather than an argument: build the wider
encoder, run `duel52 selfplay` at fixed games and sims against today's, and compare games/sec
before committing the layout. The prediction is that the difference is within noise. If it is,
be **generous** with the reserve rather than minimal — 8 status flags cost as little as 4 on the
metric that matters, and a second layout break costs another 24 hours.

---

## 8. Testing

354 tests today, of which ~146 are in `rules_*.rs` and named for their rule section. Under a
mod system:

- **Every existing rules test pins the default ruleset.** They stop being "the rules" and become
  "the canonical ruleset's rules". Mechanically this is just `Position::new(canonical_config())`
  where they currently take the default — `testkit` already threads a `GameConfig`, so this is a
  rename, not a rewrite.
- **Each new power variant gets its own named tests, in its own module**, per `CLAUDE.md`'s
  existing rule. `three.rs` holds `rule_6_three_trap_returns_face_up` and
  `mod_three_vengeance_damages_the_attacker` side by side.
- **Add a cross-ruleset invariant suite.** This is the piece that makes rule modding safe to do
  quickly, and it does not exist in any form today. Every registered ruleset, run through the
  structural properties that must hold *regardless of card rules*:

  | Invariant | Existing test to generalise |
  |---|---|
  | Determinism: same seed + config → identical game | `determinism.rs` |
  | No agent reads hidden information | `phase2_no_agent_reads_hidden_information` |
  | Observation is a function of the information set | `phase3_observation_is_a_function_of_the_information_set` |
  | `legal_actions()` empty iff the game is over | `debug_check_playable` |
  | Every game terminates without hitting `max_plies` | implicit today |
  | Card census: nothing lost or duplicated | `card_census` |
  | A player holding a card is never stuck | `rule_4_a_player_holding_a_card_is_never_stuck` |

  Right now these are asserted over three variants. Under a mod system they should be asserted
  over **every registered ruleset, automatically** — the suite enumerates `configs/rules/`
  (§5d), so registration is the file existing and nothing else. Adding a power that accidentally
  makes the game non-terminating or leaks information then fails in CI rather than in a 24-hour
  run.

  At twenty rulesets this suite is also the one thing that gets meaningfully slower, since it is
  a cross product of rulesets and invariants. Keep the per-ruleset game counts low — these are
  structural properties, and a leak or a non-termination shows up in tens of games, not
  thousands. If it does become slow, cut games per ruleset before cutting rulesets.

  The termination one deserves special attention: `game_rules.md` §7's argument that the game is
  finite depends on specific rules — *"powers fire on flip, a King reactivates once, and nothing
  ever turns a card face-down again, so total power activations are bounded."* A power variant
  that turns a card face-down, or that lets a King reactivate a King, **breaks that proof**. The
  `max_plies` cap catches it, but as a logged draw rather than an error. The invariant suite
  should treat a `PlyLimit` draw in any ruleset as a test failure.

---

## 9. The part that actually costs money

A Tier-2 rule change is ~150 lines the first time and ~20 lines thereafter. Measuring one is a
training run.

The loop is:

```
edit a card module  →  warm-start from the current champion  →  probe + within-ruleset metrics
                    →  FINDINGS entry  →  decide
```

Two things make this affordable, and both are already true:

- **Warm starting works across rulesets** (§1b). A Tier-1 or Tier-2 ruleset can start from
  `lane-gen032` rather than scratch. `train-3h.toml` warm-started from gen022 and produced a
  +82 Elo agent in three hours; a rules experiment is a smaller ask than that.
- **`--eval-batch` gave 3.26× on self-play for free** (F4.7), and self-play is 53–91% of a
  generation. The loop is already about as fast as it gets on a given box.

**At twenty rulesets (§12 Q2) the binding constraint stops being expressiveness and becomes
measurement.** Twenty × a 3 h warm-start is 60 hours of box time before anything is confounded.
So there should be a screening tier below "train an agent", and there is one available today:

> **Screen with the hand-written agents before spending a training run.** `ismcts:800` and
> `greedy` take no checkpoint, so they run under any ruleset the day the file is written —
> no training, no layout, no warm start. `probe --agents ismcts:800,random --games 400` costs
> minutes and already reports draw rate, game length, lane concentration, and per-rank
> play/flip frequency. Only survivors get a run.

This will not tell you a ruleset is *good* — search agents are not the meta. It reliably tells
you one is *broken*: degenerate, drawish, or containing a card nobody ever plays. At twenty
candidates that is most of the value, and it is the difference between 60 hours and a handful.
Keep `random` in the roster for scale, per `CLAUDE.md` — a lane concentration of 0.907 means
nothing without uniform play's 0.777 in the same table.

It also raises the priority of §10. A screen ranks rulesets against each other; only the card
value table says *which card* is responsible, which is what turns one trained net per surviving
ruleset into an answer about thirteen cards instead of one win rate.

**Cross-ruleset Elo is meaningless and you should refuse to compute it.** An agent trained on
ruleset A beating one trained on ruleset B under ruleset A's rules says nothing about whether
ruleset A is a better game. The measurements that *are* meaningful are **within-ruleset**
properties, and `probe.rs` already produces most of them:

| Metric | Already in `probe`? | What it tells you about balance |
|---|---|---|
| Draw / stalemate rate | yes | Did the change make the game degenerate? |
| Mean game length (plies) | yes | Did it make the game drag? |
| Hand at unlock, and the win/loss gap | yes | Is the endgame resource still the deciding one? |
| Lane concentration | yes (needs `random` for scale) | Is the three-lane structure still meaningful? |
| Flip-timing curve per rank | yes | Did the card's role move? |
| Rank play/flip frequency | yes | Is the card now unplayable, or now mandatory? |
| Trap-style per-card outcome rates | 3 only, hardcoded | **Needs generalising** — see below |

`probe.rs`'s `traps_sprung` / `three_fates` / `threes_face_down_at_end` are the template for
what per-card instrumentation looks like, and they are hardcoded to the 3. If a card module owns
its rules it should own its probe counters too, so that a new power arrives with its own
measurement rather than being invisible to the instrument.

---

## 10. The thing the balance goal is actually blocked on

Modularity lets you *make* a change. It does not tell you whether the change did what you
wanted. For that you need a way to say what a card is worth, and **`PLAN.md` §4's card value
table does not exist.** Its own status line says so: *"nothing exists. This is the main balance
deliverable and it is missing."*

What exists is the flip-timing curve — the order the agent turns each rank face-up, spanning
twenty-two turns from the 8 to the Queen, stable across search budgets. `PLAN.md` is careful
that this is *when a power starts paying*, not *what it is worth*, and it is right to be.

Without the value table, the balance loop is: change the 3, train for three hours, and look at
whether the win rate against a frozen reference moved — which is confounded, slow, and cannot
distinguish "the 3 got better" from "the meta shifted around it". With the value table, the loop
is: change the 3, train, and read the 3's value in win-probability units against the twelve
other cards, before and after. That is a direct answer to "is this card now worth what the
others are worth", which is the actual balance question.

**So: build the value table alongside the rules system, not after it.** They are the two halves
of one instrument, and the rules system on its own is a machine for generating unanswerable
questions. `PLAN.md` §4 already describes the route — hold a `testkit` position fixed, vary one
card's rank, read the value head's delta — and notes the one prerequisite, a trustworthy value
head, which is what the two runs in flight are for.

### ✅ Built, 2026-09-10 — `engine/src/cardvalue.rs`, `duel52 card-value`

**Vary the card in *hand*, not the card on the board.** The first implementation put rank `R`
face-up in a lane and read the value head, and its answer was wrong in a way that looked like
a finding — worth recording, because the failure is instructive.

Its top four came out **8, J, 10, 9**: exactly `PowerId::is_constant()`, in a block, above
everything else. Not a fact about Duel 52 — the method reading its own selection criterion
back out. `game_rules.md` §6's three-way split of powers predicts it exactly:

| power kind | ranks | what "face-up in a lane" measures |
|---|---|---|
| constant | 8, 9, 10, J | the power live and working — **100% of its value** |
| one-shot | A, 2, 4, 5, 6, 7, Q, K | a **spent** card; it fired on the flip and the effect is past |
| conditional | 3 | **nothing** — the Trap works only face-down |

So it ranked *how much of a card's value survives being face-up*, which is a property of the
power's type. A card **in hand** has its whole future ahead of it whatever kind of power it
carries, so every rank is measured at the same point in its life.

Measured on `lane-gen032`, 400 positions, canonical rules, win-probability points. `in hand`
is the measurement and is paired — every rank on the identical position, so the position's own
difficulty subtracts out and the ± is the error on the *comparison*.

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
`on board` is sorted by it. One-shots span the whole in-hand range (+3.50 to −3.83) with the
four constants scattered through it at positions 2, 6, 7 and 9. The board column still has all
four constants on top and nothing else above +0.54 — which is the original artifact, now
isolated in a column that is labelled as not comparable across kinds.

**The `gap` column is the useful by-product.** It says where a card's value lives: large and
positive (the Ace at +5.00, the Queen at +4.21) means the value is in the flip, and the card is
a spent body afterwards; negative (the 8 at −3.34, the 10 at −3.09) means the value is the body
standing in the lane.

**The control.** Each rank is also substituted into the **opponent's** hand, which the observer
cannot see — the tensor carries `my_hand_counts` for the observer and only
`opponent_hand_size` for the other side, so all thirteen are bit-identical and the value head
must return one number. It does: spread `0.00000`. `engine/tests/cardvalue.rs` checks this
against a *hash* of the observation rather than a network, which is strictly stronger — a real
value head could return the same number for two different tensors by luck.

⚠️ **Three limits, all of them real.**

1. **This measures gen032's value head, not Duel 52.** The prerequisite `PLAN.md` §4 names.
2. **The sample is biased toward early, high-uncertainty positions.** Asking "what if I held an
   `R`" requires a copy of `R` to be somewhere the observer cannot see, so positions where all
   thirteen ranks are still holdable are ones where little has been revealed. 8.3% of sampled
   `split` positions survive.
3. **`mirrored` is out of reach entirely** — 0.06% survive, because §9b publishes the removed
   multiset and the observer then accounts for nearly their whole deck. `card-value` detects a
   thin sample and refuses to print rather than showing a table nobody can interpret.

⚠️ **A correction to an earlier claim in this file.** The first version of this section said the
table "corroborates the flip-timing curve at both ends — the 8 earliest-flipped *and* highest
valued, the Queen latest *and* lowest". That corroboration is real but belongs to the **board**
column, not to card value: the 8 is +4.01 on board and the Queen −1.97, while in hand the
ordering is nearly reversed. Which makes sense — the flip-timing curve is about the face-up
card, and so is the board column. It is a consistency check on the board measurement, not
evidence for the value table.

---

## 11. Sequencing

Ordered by dependency, with the two runs in flight respected.

⚠️ **Do not rebuild the binary those runs are using.** Do all of this on a branch. Config
extension is backward-compatible for *reading* old artifacts (missing keys take defaults), so
shards from the current runs will replay fine under a new build — but a mid-run binary swap is
its own hazard for unrelated reasons.

✅ **Steps 0–10 were implemented on branch `ruleset-configs`, 2026-09-09/10.** 382 Rust tests
(354 before, so 28 new) and 102 Python tests pass, and the layout hash is unmoved:
`obs b1355a841a1fdc4a` at `encoding_slots = 21`, exactly what it was before the refactor, so
every shipped checkpoint still loads and `--init-from` still crosses rulesets. Step 11 is
deliberately **not** done — see §7.

| # | Step | Depends on | Rough size |
|---|---|---|---|
| 0 | Correct `PLAN.md` §5's claim about per-variant layouts | — | one paragraph |
| 1 | `rules_hash`: compute, stamp into `.d52nn` and `.d52sp`. Hard-error in `Shard::read` and in `ladder`/`match`/`probe`; print-and-continue in the training loop; **not** in `Weights::load` (§6). Name and freeze `canonical-2026-09`. | — | small, do it first |
| 2 | Tier-1 numeric fields: every magic constant in the powers becomes a `GameConfig` field, defaulted to today's value. Round-trip test proves nothing was missed. | 1 | small, high value |
| 3 | `DamageSource` + the damage queue + `on_lethal_damage` as a hook. **Zero behaviour change**; the 354 tests are the proof. | — | the real refactor |
| 4 | `PowerId` + `engine/src/powers/`, one module per rank. `fire_power` becomes dispatch. `rank.rs`'s five tables move onto `PowerId`. Still zero behaviour change. | 3 | medium |
| 5 | Config plumbing: dotted keys, `include`, resolved round-trip, `configs/rules/` as the registry, and `GameSettings.rules_file` on the Python side emitting `--config`. | 2, 4 | small |
| 6 | Cross-ruleset invariant suite over the registry, including "no ruleset may reach `max_plies`". | 4, 5 | small, load-bearing |
| 7 | Per-card probe counters owned by the card module; generalise `three_fates`. | 4 | small |
| 8 | **Screening harness**: `probe --agents ismcts:800,random` over every registered ruleset, one comparison table (§9). No training involved. | 5 | small, saves the most clock |
| 9 | **First real mod**: `ThreeTrapVengeance1`. Warm-start run, probe, `FINDINGS` entry. | 3–8 | one training run |
| 10 | `PLAN.md` §4's card value table. Can start any time after the runs land; independent of 1–9. | a good value head | the other half |
| 11 | **Encoder reserve**: §7's items 1–3 *plus* every known Tier-3 change, batched into one layout revision. ⚠️ **Must land before the next from-scratch run starts.** | a scheduled from-scratch run | one break, taken once |

Steps 1 and 2 are worth doing regardless of whether you ever build the rest. Step 3 is the one
that unlocks the category. Step 8 is the one that makes twenty rulesets affordable at all.
Step 11 is the only item with a hard external deadline — miss the from-scratch run it is
attached to and it waits for the next one.

### What landed, and where it lives

| Step | Where |
|---|---|
| 0 | `PLAN.md` §5, with the counter-evidence and the command that produced it |
| 1 | `GameConfig::rules_hash`, stamped into `.d52nn` and `.d52sp`; hard error in `Shard::read`, `ladder`/`match`/`probe`/`card-value` and the Python replay buffer; **not** in `Weights::load` |
| 2 | Eleven named numeric fields on `GameConfig`, every one defaulted to the rules-as-written value |
| 3 | `engine/src/damage.rs` — `DamageSource`, `Hit`, `DamageQueue`; `apply.rs`'s `drain_damage` |
| 4 | `engine/src/powers/`, one module per rank. `fire_power`'s 13-arm match is gone |
| 5 | Dotted `powers.*` keys, `include`, `configs/rules/` as the registry, `GameSettings.rules_file`, `encoding_spec(rules_file=…)` |
| 6 | `engine/tests/rulesets.rs` — 11 invariants over every registered ruleset |
| 7 | `probe`'s `triggers_sprung_by_rank` / `flipped_by_cascade_by_rank` / `face_down_at_end_by_rank`, and `card_fates(rank)` |
| 8 | `duel52 screen` |
| 9 | `configs/rules/*.toml` (7 rulesets), `engine/tests/rules_mods.rs` (14 tests), `configs/train-mod-3h.toml` |
| 10 | `engine/src/cardvalue.rs` and `duel52 card-value` |

**Two things the build found that this document had wrong.** Neither changes a conclusion, but
both were real:

- **`include` must come first in a ruleset file.** Keys resolve last-wins uniformly, which
  applies to `rules_name` too — so an `include` written below the name silently replaces it
  with the base's. Every ruleset reported itself as `canonical-2026-09` while its `rules_hash`
  stayed correct, so nothing looked wrong except the label. Documented in
  `configs/rules/README.md`.
- **An *unstamped* checkpoint cannot be judged against a fabricated hash.** §6's table implied
  a pre-mod checkpoint could be treated as "canonical for the runtime variant", which would
  have let the original `gen032`-plays-`--variant base` bug straight through. The honest split
  is: unstamped **+ modded runtime** is a definite mismatch and refuses; unstamped **+
  canonical runtime** cannot be checked at all and warns loudly.

---

## 12. Decisions

Answered 2026-09-09. Each one is folded into the section it governs; this is the record of what
was chosen and what it cost.

| Q | Decision | Governs | What it changed |
|---|---|---|---|
| 1 | **Named variants carry their own numbers.** No per-variant config fields; `trap_vengeance_one_damage` and `trap_vengeance_two_damage` are two names. | §5a | `PowerId` stays a data-free enum. Removes the risk of a known-but-inapplicable config key, which `from_config_str` cannot catch. Tier-1 numeric fields survive unchanged. |
| 2 | **Up to twenty rulesets.** | §5d, §8, §9 | Build the `include` machinery properly. `configs/rules/` becomes the registry the invariant suite enumerates. The bottleneck moves from expressiveness to *measurement*, which is what adds the screening tier (§9, step 8). |
| 3 | **From-scratch runs are acceptable — take the reserve.** | §7, step 11 | All three reserve items, not just the two cheap ones. Turns a judgment call into a scheduling constraint: the break must land before the next from-scratch run starts. |
| 4 | **Cross-ruleset play is purely a hazard.** | §6 | `--allow-cross-rules` removed. But one check becomes three severities, because generation 1 of every warm-started run is legitimately cross-ruleset — so `rules_hash` must *not* be enforced in `Weights::load`. |

**What is still open.** Two things, both of which need the runs in flight to report first:

- **When the batched layout revision happens** (step 11). It needs a from-scratch run to attach
  to, and whether one is close depends on whether the 32-core and 128-core runs produce a
  champion. Everything that should ride along with it — §7's three reserve items, the peek action
  of §5e if it is wanted with a `CHOOSE_LANE`-shaped sibling, any other known Tier-3 mod — should
  be listed *before* that run is scheduled, not discovered after.
- **Whether the value head is good enough for the card value table** (§10, step 10). That is the
  one prerequisite `PLAN.md` §4 names, and it is precisely what the two runs are for.
