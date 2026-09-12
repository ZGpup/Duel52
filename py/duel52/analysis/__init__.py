"""Self-play analysis: read the corpora `duel52 analyze` writes, and render one document.

    .venv/bin/python -m duel52.analysis --help

The engine plays the games and writes down what happened, one row per player-game and one
row per card. Everything here is a fold over those two tables, so a new question is a
function in `metrics.py` and a re-render — never another run of the games.
"""

from .corpus import Corpus, load_dataset, datasets  # noqa: F401
from .metrics import METRICS, Context, Section, metric  # noqa: F401

__all__ = [
    "Corpus",
    "load_dataset",
    "datasets",
    "METRICS",
    "Context",
    "Section",
    "metric",
]
