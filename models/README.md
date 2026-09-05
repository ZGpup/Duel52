# Shipped checkpoints

A `.d52nn` checkpoint is the whole agent: weights, the architecture that reads them, and the
two layout hashes that pin it to an encoder. `Weights::load` refuses a checkpoint whose
hashes do not match the build loading it, so a stale model fails with a one-line error rather
than by quietly playing badly.

Checkpoints here are tracked in git as ordinary blobs — no LFS, nothing to install. They are
~3.6 MB each, which is what a 949k-parameter fp32 net costs.

| File | Variant | Slots | Params | Provenance |
| --- | --- | --- | --- | --- |
| **[duel52-split-gen022.d52nn](duel52-split-gen022.d52nn)** — the default | `split` | 21 | 949,267 | Phase 4 `train-2h`, warm-started from gen016 |
| [duel52-split-gen016.d52nn](duel52-split-gen016.d52nn) — superseded | `split` | 21 | 949,267 | Phase 3 `train-fast`, generation 16 |

**Play `gen022`.** `gen016` is kept because every Phase 3 finding is measured on it and
because it is the fixed opponent Phase 4 is scored against — it is a reference point, not a
second option.

---

## duel52-split-gen022.d52nn

**The strongest Duel 52 agent that exists.** Produced by `configs/train-2h.toml` on the same
laptop as its predecessor, and the whole of what it changes is the *teacher*: self-play
generates its policy targets at **256 simulations rather than 64**. `FINDINGS.md` F3.8 is why
— the policy target is the visit distribution, so training at 64 teaches the network to
imitate a search hundreds of Elo weaker than the same weights produce at 4096. The teacher
was capped, and this is the cap coming off.

### How it was made

```bash
.venv/bin/python -m duel52.train run --config configs/train-2h.toml \
    --run-dir runs/fourth --init-from models/duel52-split-gen016.d52nn
```

| | |
| --- | --- |
| Config | `configs/train-2h.toml` (recorded in the run dir; generations 2–8 under `train.toml.used.from-gen002`) |
| Started from | `duel52-split-gen016.d52nn`, **not** a random init |
| Seed | `3000000`, spanning seeds 3,000,000–3,009,600 |
| Variant | `split`, `two_power = bottom`, `encoding_slots = 21` |
| Generations | 8 played, 3 promoted; **generation 6 is this file** |
| Self-play | 9,600 games, 256 PUCT simulations per decision |
| Wall clock | ~1.9 h on an M-series Mac, 8 cores |
| Network | residual MLP, width 128, 3 blocks, value head 128 → 949,267 parameters |
| `obs_dim` / `action_dim` | 4290 / 2194 |
| `obs_layout_hash` | `b1355a841a1fdc4a` |
| `action_layout_hash` | `5169f9461d627b39` |
| SHA-256 | `3d499accba3b3af8c86ac2b09c0ac9f90abc2bcfb059f46f8a0b8567ef184bce` |

**Why "gen022" when the run calls it generation 6.** It is `runs/fourth`'s sixth generation,
and the 22nd generation of training counting the 16 it inherited. The run number would sort
below its own predecessor on the shelf, which is the one thing a filename here has to get
right. The file is byte-identical to `runs/fourth/checkpoints/gen006.d52nn`.

**The trunk is 128 × 3, the same as gen016, and that is a consequence rather than a choice.**
A warm start cannot change the shape of the network it inherits. `PLAN.md` §4.2 change 4 wants
six blocks; that needs a from-scratch run, which is what `configs/train-big.toml` is for.

### How strong it is

Against the checkpoint it was trained from, at **equal simulations**, 400 games, `--seed 1`:

| | score (95% CI) | W–L–D | Elo |
| --- | --- | --- | ---: |
| `netmcts:gen022@256` vs `netmcts:gen016@256` | **0.6150 ± 0.0474** | 244–152–4 | **+81** |

That is the honest headline, and it is worth more than the ladder row below because both
sides of it are real agents searching the same amount. On the frozen ladder, 400 games per
pairing, seeds from 1 (`FINDINGS.md` F4.2):

| agent | Elo | ± | vs. anchor |
|---|---:|---:|---:|
| **`netmcts:gen022@256`** | **+1788** | 58 | 1.000 |
| `ismcts:800` | +1052 | 15 | 0.998 |
| `flatmc:600` | +900 | 13 | 0.994 |
| `greedy` | +615 | 13 | 0.972 |
| `pimc:8x1` | +584 | 13 | 0.967 |
| `random` | +0 | 0 | 0.500 |

```bash
./target/release/duel52 ladder --games 400 --markdown --variant split --encoding-slots 21 \
    --agents random,greedy,flatmc:600,pimc:8x1,ismcts:800,netmcts:models/duel52-split-gen022.d52nn@256
```

Head to head, 200 games each, `--seed 1`, scores for this checkpoint's side, with gen016's
figures from its own section for comparison:

| Agent | Opponent | gen022 | gen016 |
| --- | --- | --- | --- |
| `netpolicy` (no search) | `random` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `netpolicy` (no search) | `greedy` | **0.9800** ± 0.0194 | 0.9400 ± 0.0329 |
| `netmcts` | `greedy` | **1.0000** ± 0.0000 | 0.9675 ± 0.0241 |
| `netmcts` | `ismcts:800` | **0.9900** ± 0.0138 | 0.9300 ± 0.0354 |

⚠️ The two `netmcts` rows are **not** a like-for-like comparison: gen022 searches 256 and
gen016 searched 64. The `netpolicy` rows are, because neither searches at all — and there the
policy head alone went 0.940 → 0.980 against `greedy`, which is the network improving rather
than the budget. Against `ismcts:800` it arrives at the endgame holding **8.36** cards to
`ismcts`'s 0.85, and has a turn with nothing useful to do **0.00** times a game against 2.40.

⚠️ **This is +1788 against gen016's +1476 and the difference is not +312.** Bradley–Terry pins
`random` at 0 and fits the rest to the whole graph, so pulling the top agent away stretches
everything below it — every hand-written rung moved up between the two tables without a line
of code changing. Ratings compare *within* one fit, never across two. Measured over the rungs
common to both, the gap is **+241 to +247**, and roughly three fifths of that is the larger
search budget rather than better weights. F4.2 has the full reconciliation.

### What it does differently

400 games against gen016, both at 256 simulations:

| | gen022 | gen016 |
| --- | ---: | ---: |
| cards in hand when lanes unlock | **6.87** | 6.41 |
| …in games won vs lost | 7.25 vs 6.30, gap **+0.94 ± 0.23** | 7.07 vs 5.98, gap +1.09 ± 0.25 |
| turns with nothing useful to do | **0.43** | 0.70 |
| flip rate | 0.91 | 0.88 |

It hoards slightly more and wastes fewer turns. The H2 hoard-predicts-win gap clears its own
interval for both nets, which makes it two independent agents showing the same thing rather
than one.

### Known limits

- **It has never beaten a human.** The owner's record against gen016 was 0–5 and there is no
  recorded series against this one yet. `duel52 play --record` exists now (`PLAN.md` §4.0), so
  the next series will at least be written down.
- **The value head is still the weak half.** It scores 0.655 held-out where gen016 scored
  0.774 — a real improvement, and still only about a third of outcome variance explained on a
  ±1 target. `FINDINGS.md` F4.1.
- **Trained only on `split`**, like its predecessor: the observation layout is per-variant and
  the hash check refuses `base` or `mirrored` outright.
- **The run ended on a learning-rate schedule, not on a plateau.** Generations 7 and 8 scored
  0.493 and 0.488 at a rate 16× below where they started. It stopped improving because the
  schedule stopped it, which means the config has more in it, not that the method does.

---

## duel52-split-gen016.d52nn

**Superseded by `gen022`, and kept deliberately.** Every Phase 3 finding is measured on this
checkpoint, and it is the frozen opponent the Phase 4 run is scored against — so it is a
fixed reference point rather than a second thing to play.

The first trained Duel 52 agent, and as far as I know the first one that has ever existed.
Produced by the Phase 3 AlphaZero loop on `configs/train-fast.toml` — the two-hour laptop
config, a shakedown run rather than a compute-serious one.

It beats the top of the hand-written ladder by a wide margin, while searching twelve times
less. That says it is far ahead of anything else available to play against; it says nothing
about how close to optimal it is, because there is no stronger reference to measure it
against — and the one external check there is says it is not close: **the project owner beats
it 5–0.** Closing that gap is Phase 4's problem (a scaled-up run), and measuring how far it really
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
