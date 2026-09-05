# Archived working documents

These are the **superseded** working documents, kept for provenance and frozen as of
**2026-09-05**. They are not maintained. Nothing should link to them as current, and no
decision should be taken from them without checking the live document first.

| Archived file | What replaced it |
|---|---|
| `PLAN.md` | `/PLAN.md`, rewritten around what is left to do rather than what has been done |
| `FINDINGS.md` | `/FINDINGS.md`, rewritten around the trained agents only |
| `DESIGN.md` | `/CLAUDE.md`, which carries the architecture facts that are still load-bearing |
| `OPEN_QUESTIONS.md` | Nothing. It closed: every rules question was answered and ported into `/game_rules.md` before it was archived |
| `REPLAY.md` | `/README.md`'s "Reading a recorded game" section, plus `duel52 replay`'s own output |

## Why they were archived rather than edited

They grew as working notes and recorded everything that had ever been tried, in the order it
was tried. That made them a good audit trail and a poor briefing: the current state of the
project was spread across a thousand lines of superseded intermediate results, and the phases
already finished took more space than the phases still open.

The rewrite kept the measurements and dropped the narrative. Where a number in the live
documents cites a finding by its old identifier (F2.5, F3.8, F4.1 and so on), the full
original entry is in this folder's `FINDINGS.md`.

## What is still only in here

Worth knowing before deleting the folder:

- **The Phase 1 random-play baseline and the Phase 2 hand-written ladder in full.** The live
  `FINDINGS.md` keeps only the conclusion that they exist and what they were for. All of the
  per-agent detail, and the reasoning that produced the five rungs, is here.
- **The build history of the training loop** (F3.1 through F3.6, F3.11, F3.12) — encoder
  density, throughput measurements, the stalemate-equilibrium episode, the device benchmark.
  The live documents carry the conclusions that still bind; the workings are here.
- **`DESIGN.md`'s long-form architecture rationale.** The parts still load-bearing were moved
  into `CLAUDE.md`; the alternatives considered and rejected were not.
- **Every superseded plan for Phase 4**, including the sizing arithmetic for hardware that was
  never rented.
