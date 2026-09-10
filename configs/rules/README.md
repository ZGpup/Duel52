# `configs/rules/` — the ruleset registry

**This directory is the registry.** A ruleset is a `.toml` file here and nothing else — there
is no list to add it to. `engine/tests/rulesets.rs` enumerates the directory at test time, so
a new file is covered by every cross-ruleset invariant the moment it exists
(`MODULAR_RULES.md` §5d, §8). If registering a ruleset required also registering its test,
at twenty rulesets you would forget, and the one you forgot would be the one that quietly
fails to terminate inside a 24-hour run.

## Anatomy of a ruleset

```toml
include = "canonical.toml"            # resolved against THIS file's directory
rules_name = "three-vengeance-1"      # a label; not part of rules_hash
powers.three = "trap_vengeance_one_damage"
```

- **`include` is spliced in where it appears**, depth first. Later keys win, so anything the
  file writes itself overrides what it included.
- ⚠️ **Put `include` first.** Last-wins is uniform and applies to `rules_name` too, so an
  `include` written *below* the name silently replaces it with the base's — every ruleset here
  would report itself as `canonical-2026-09`, and the `rules_hash` would still be correct, so
  nothing would look wrong except the label. `duel52 config <file>` prints the resolved name;
  check it once when you add a file.
- **`rules_name` is provenance, not a rule.** Two files that play the same game have the same
  `rules_hash` whatever they are called.
- **Explicit is better than clever.** `duel52 config configs/rules/<file>` prints the fully
  resolved form, which is exactly what gets stamped into every shard and game record.

## What is here

| File | Tier | What changes |
|---|---|---|
| `canonical.toml` | — | The rules as written plus the house 2. The baseline every other file is a diff against. |
| `three-vengeance-1.toml` | 2 | The 3 also deals 1 damage to the card that killed it. |
| `three-vengeance-2.toml` | 2 | The same, for 2 damage. Exists to show that a number inside a shape is a **second name**, not a config knob (§5a). |
| `three-none.toml` | 2 | Ablation: the 3 has no Trap at all. What is the Trap worth? |
| `eight-on-survival.toml` | 2 | The 8 retaliates **only if it survives** the attack — the exact inverse of the rules-as-written ruling that it "fires even if that damage killed the 8". |
| `eight-none.toml` | 2 | Ablation: the 8 does not hit back. |
| `jack-2hp.toml` | 1 | The Jack still taunts but has 2 HP, not 3. A pure config number — no code, no new power. |

## Adding one

1. Write the file. Keep it a **diff**: include a base and change what you mean to change.
2. `./target/release/duel52 config configs/rules/<file>` — validates it, prints the resolved
   form, and tells you its `rules_hash` and which cards differ from canonical.
3. `cargo test --test rulesets` — every structural invariant, over every file here.
4. `./target/release/duel52 screen` — the cheap behavioural read, no training involved.

Only then is it worth a training run. `MODULAR_RULES.md` §9: twenty rulesets at a 3 h
warm-start each is 60 hours, and screening is what keeps that number down to the ones that
are actually interesting.
