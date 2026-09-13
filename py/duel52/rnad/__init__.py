"""R-NaD — a second learner beside the AlphaZero loop. ``PLAN.md`` item 8.

    python -m duel52.rnad check --config configs/rnad-3h.toml
    python -m duel52.rnad bench --config configs/rnad-3h.toml
    python -m duel52.rnad run   --config configs/rnad-3h.toml --run-dir runs/rnad-3h

Regularised Nash Dynamics (Perolat et al., *Science* 2022), ported from OpenSpiel's
``rnad.py`` at ``d1dcdf5d``. Search-free: the actor samples moves from the network's policy
for a batch of games, and the learner fits the policy with NeuRD and the value with V-trace,
on rewards transformed against a regularisation policy that moves on a fixed schedule.

Beside, not instead. It shares the engine, the encoder, the network classes and the
``.d52nn`` format with ``duel52.train`` and writes checkpoints every measuring command reads
(``duel52 match --a netsample:<checkpoint>``). ``duel52.train`` imports nothing from here.

* ``core``    — the arithmetic, each function checked against the reference.
* ``actor``   — plays a batch of games through the engine's ``GameBatch``, sampling on device.
* ``learner`` — the four networks, one R-NaD step.
* ``loop``    — the run: logging, evaluation matches, checkpoints, resume.
* ``kuhn``    — Kuhn poker, a test fixture with a known equilibrium.
"""
