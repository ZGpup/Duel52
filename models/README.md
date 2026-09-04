# Shipped checkpoints

A `.d52nn` checkpoint is the whole agent: weights, the architecture that reads them, and the
two layout hashes that pin it to an encoder. `Weights::load` refuses a checkpoint whose
hashes do not match the build loading it, so a stale model fails with a one-line error rather
than by quietly playing badly.

Checkpoints here are tracked in git as ordinary blobs — no LFS, nothing to install. They are
~3.6 MB each, which is what a 949k-parameter fp32 net costs.

| File | Variant | Slots | Params | Provenance |
| --- | --- | --- | --- | --- |
| [duel52-split-gen016.d52nn](duel52-split-gen016.d52nn) | `split` | 21 | 949,267 | Phase 3 `train-fast`, generation 16 |

---

## duel52-split-gen016.d52nn

The first trained Duel 52 agent, and as far as I know the first one that has ever existed.
Produced by the Phase 3 AlphaZero loop on `configs/train-fast.toml` — the two-hour laptop
config, a shakedown run rather than a compute-serious one.

It beats the top of the hand-written ladder by a wide margin, while searching twelve times
less. That says it is far ahead of anything else available to play against; it says nothing
about how close to optimal it is, because there is no stronger reference to measure it
against — and the one external check there is says it is not close: **the project owner beats
it.** Closing that gap is Phase 4's problem (a scaled-up run), and measuring how far it really
is from optimal is Phase 6's (local best-response, and the CFR cross-check on a scaled down
variant).

### How it was made

```bash
.venv/bin/python -m duel52.train run --config configs/train-fast.toml --run-dir runs/third
```

| | |
| --- | --- |
| Config | `configs/train-fast.toml` (recorded verbatim as `train.toml.used` in the run dir) |
| Seed | `1000000` |
| Variant | `split`, `two_power = bottom`, `encoding_slots = 21` |
| Generations | 19 trained on top of a random init; **generation 16 is this file** |
| Self-play | 57,000 games, 7.49M positions, 64 PUCT simulations per decision |
| Wall clock | 1.94 h on an M-series Mac, `device = auto` |
| Network | residual MLP, width 128, 3 blocks, value head 128 → 949,267 parameters |
| `obs_dim` / `action_dim` | 4290 / 2194 |
| `obs_layout_hash` | `b1355a841a1fdc4a` |
| `action_layout_hash` | `5169f9461d627b39` |
| SHA-256 | `03de8583a4a3b204e96972ea90de7cde9d84745d98ef83f467ba0d3fe10040f1` |

**`--encoding-slots 21` is mandatory.** `encoding_slots` is what fixes `obs_dim`, so a
command that leaves it at the default 16 will be refused by the hash check. 21 rather than
16 because of `FINDINGS.md` F3.1.

### Why generation 16

Each generation is gated: the candidate must score ≥ 0.55 against the incumbent over 200
games before it replaces it. Thirteen of nineteen candidates passed. Generations 17, 18 and
19 all failed (0.503, 0.543, 0.528 — every one inside its own ±0.069 confidence interval of
even), which is three consecutive refusals and ends the run by design.

So gen016 is not "where we stopped", it is where measurable improvement stopped under this
config. Policy loss kept falling afterwards (2.02 → 1.94) without the resulting player being
any better, which is the shape of a run that has run out of the thing it was learning from —
64 simulations of search do not produce targets a 949k-parameter net cannot already fit.

### How strong it is

On the ladder, 200 games per pairing, seeds from 1:

| agent | Elo | ± | vs. anchor |
|---|---:|---:|---:|
| **`netmcts@64`** | **+1476** | 42 | 1.000 |
| `ismcts:800` | +981 | 19 | 0.996 |
| `flatmc:600` | +835 | 17 | 0.992 |
| `greedy` | +581 | 17 | 0.966 |
| `pimc:8x1` | +547 | 17 | 0.959 |
| `random` | +0 | 0 | 0.500 |

```bash
./target/release/duel52 ladder --games 200 --markdown --variant split --encoding-slots 21 \
    --agents random,greedy,flatmc:600,pimc:8x1,ismcts:800,netmcts:models/duel52-split-gen016.d52nn@64
```

**+495 Elo clear of the previous best**, on one twelfth its simulation budget. Two cautions
about reading that number:

- **This does not compare to the frozen Phase 2 ladder** (`FINDINGS.md` F2.1), for two
  reasons and the second is the serious one. Elo is roster-relative and anchored at
  `random = 0`, and that roster had no net and used `pimc:32x1`. But F2.1 also **predates the
  §4 mandatory-action ruling**, so it was fitted on a game where players could pass. This is
  the first ladder run on the rules as they now stand. `ismcts:800` reading +1186 there and
  +981 here is two different games, not a rung that got worse.
- **±42 is the widest interval in the table**, because Elo is steep at these margins: an
  agent that scores 0.93 sits where a small change in score is a large change in rating. The
  gap is better read as "several hundred Elo" than as 495 ± 46.

That said, the fit and the direct measurement agree. A +495 gap predicts a 0.945 score
against `ismcts:800`; the head-to-head below measured 0.930 ± 0.035. Two independent routes
to the same answer.

### Head to head

Scores are for the checkpoint's side; 0.5 is even. 200 games each, `--seed 1`, variant
`split`, `two_power = bottom`.

| Agent | Opponent | Score (95% CI) | W–L–D |
| --- | --- | --- | --- |
| `netpolicy` (no search) | `random` | **1.0000** ± 0.0000 | 200–0–0 |
| `netpolicy` (no search) | `greedy` | **0.9400** ± 0.0329 | 188–12–0 |
| `netmcts@64` | `greedy` | **0.9675** ± 0.0241 | 193–6–1 |
| `netmcts@64` | `ismcts:800` | **0.9300** ± 0.0354 | 186–14–0 |

```bash
./target/release/duel52 match --a netmcts:models/duel52-split-gen016.d52nn@64 \
    --b ismcts:800 --games 200 --seed 1 --encoding-slots 21
```

Two things in that table are worth more than the headline. The top rung falls to *one
twelfth* its simulation budget — 64 against 800 — so what the net contributes is not search
volume. And the policy head alone, argmax with no search whatsoever, already beats `greedy`
0.940; search on top of it is worth only another 0.027. Most of the strength is in the prior.

### What it does differently

`match` reports behaviour as well as score. Against `ismcts:800`:

| | `netmcts@64` | `ismcts:800` |
| --- | --- | --- |
| Cards in hand when lanes unlock | **7.98** | 1.00 |
| …in games it won vs lost | 8.09 vs 6.57, gap **+1.51 ± 0.73** | 1.14 vs 0.99, gap +0.15 ± 0.51 |
| Plays per game | 15.5 | 18.0 |
| Flip rate | 0.82 | 0.65 |
| Turns with nothing useful to do | 0.06 | 2.65 |

**The trained agent hoards, and the hoard predicts the win.** It arrives at the endgame
holding eight cards where the top hand-written rung arrives holding one, and within its own
games the winning ones are the ones where it held more — a gap of +1.51 cards that clears its
own confidence interval. Phase 2 looked for exactly this and found nothing (`FINDINGS.md`
F2.5), with the caveat that no agent on that ladder could hoard on purpose. This one can.

That is the first support hypothesis H2 has, and it is **not yet a finding** — it is one
pairing at one seed, and a between-agent difference cannot be told apart from a within-agent
one here. `PLAN.md`'s Phase 5 list already names the experiment that would settle it.

### Known limits

- **Trained only on `split`.** The observation layout is per-variant, so this checkpoint
  cannot be loaded against `base` or `mirrored` at all — the hash check will refuse it.
- **`stalemate_value = 0.0`** during training. It never mattered: since the §4
  mandatory-action ruling the engine cannot reach an engine-declared stalemate, so the term
  weighted nothing. See `FINDINGS.md` F3.6 for what it was guarding against.
- **Nothing here is a Phase 5 finding.** This checkpoint is an instrument that works, not a
  measurement of how Duel 52 should be played.
