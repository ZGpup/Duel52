# Run plans

One dated plan per batch of jobs sent to someone else's compute: what runs, why, how to
launch it, how to resubmit it, and what to copy back. Name each `YYYY-MM-DD.md` for the day
the jobs are queued. Put any scripts it needs in a same-named directory beside it.

A plan is written **before** the run and is not rewritten afterwards. What the run actually
did goes into `FINDINGS.md`, `models/README.md` and the analysis documents. Add a short
**Outcome** section at the bottom of the plan, pointing to those, once the results are home.

| Plan | What |
|---|---|
| [2026-09-29](2026-09-29.md) | Five 4096-sim analysis corpora, a base-rules AlphaZero run, two 24 h R-NaD runs. Slurm |
