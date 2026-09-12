"""Inline SVG charts, with no dependencies.

The report is a file on disk that has to render in a browser with nothing installed, so the
figures are SVG written out by hand rather than a plotting library. Three forms cover
everything here, chosen by the job the data is doing:

* :func:`grouped_bars` — magnitude compared across models, one group per rank.
* :func:`lines` — a trend read along an ordered axis (rank order, or hand-size margin).
* :func:`stacked_bars` — parts of a whole that sum to 1, one bar per model.

The palette is the validated categorical default, slots 1–3, which clear every colour-vision
and contrast gate on the all-pairs list; a fourth model takes slot 4 and the pairing is
checked against the adjacent list instead. Colour is never the only encoding — every series
is in the legend and every mark carries a `<title>`, which is the browser's own tooltip and
costs no JavaScript.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

#: Categorical slots, in fixed order. Never cycled: a fifth series folds into a table.
SERIES_SLOTS = 4


@dataclass
class Series:
    name: str
    values: Sequence[Optional[float]]
    errors: Optional[Sequence[Optional[float]]] = None


def _fmt(value: float, places: int) -> str:
    return f"{value:.{places}f}"


def _nice_bounds(low: float, high: float) -> tuple:
    """A rounded axis range that contains the data, with a little air."""
    if not math.isfinite(low) or not math.isfinite(high):
        return 0.0, 1.0
    if high - low < 1e-9:
        low, high = low - 0.5, high + 0.5
    span = high - low
    step = 10 ** math.floor(math.log10(span / 4.0)) if span > 0 else 1.0
    for multiple in (1, 2, 2.5, 5, 10):
        if span / (step * multiple) <= 5:
            step *= multiple
            break
    return math.floor(low / step) * step, math.ceil(high / step) * step


def _ticks(low: float, high: float, count: int = 5) -> List[float]:
    if high <= low:
        return [low]
    step = (high - low) / count
    return [low + step * i for i in range(count + 1)]


class _Canvas:
    """The bits every chart shares: a frame, a y axis, and category labels."""

    def __init__(
        self,
        width: int,
        height: int,
        left: int = 54,
        bottom: int = 34,
        top: int = 14,
        right_pad: int = 12,
    ):
        self.width = width
        self.height = height
        self.left = left
        self.right = width - right_pad
        self.top = top
        self.bottom = height - bottom
        self.parts: List[str] = []

    @property
    def plot_width(self) -> float:
        return self.right - self.left

    @property
    def plot_height(self) -> float:
        return self.bottom - self.top

    def y_of(self, value: float, low: float, high: float) -> float:
        if high <= low:
            return self.bottom
        return self.bottom - (value - low) / (high - low) * self.plot_height

    def axis(self, low: float, high: float, places: int, ticks: int = 5) -> None:
        for value in _ticks(low, high, ticks):
            y = self.y_of(value, low, high)
            self.parts.append(
                f'<line x1="{self.left}" y1="{y:.1f}" x2="{self.right}" y2="{y:.1f}" '
                f'class="grid"/>'
            )
            self.parts.append(
                f'<text x="{self.left - 8}" y="{y + 3.5:.1f}" class="tick" '
                f'text-anchor="end">{_fmt(value, places)}</text>'
            )

    def rule(self, value: float, low: float, high: float, label: str) -> None:
        y = self.y_of(value, low, high)
        self.parts.append(
            f'<line x1="{self.left}" y1="{y:.1f}" x2="{self.right}" y2="{y:.1f}" '
            f'class="rule"/>'
        )
        # Above the plot rather than on the line: a bar that reaches the right-hand end of
        # the axis sits exactly where the label used to be, and the two overlapped.
        self.parts.append(
            f'<text x="{self.right}" y="{self.top - 5:.1f}" class="rule-label" '
            f'text-anchor="end">{html.escape(label)}</text>'
        )

    def categories(self, labels: Sequence[str], band: float) -> None:
        for i, label in enumerate(labels):
            x = self.left + band * (i + 0.5)
            self.parts.append(
                f'<text x="{x:.1f}" y="{self.bottom + 16:.1f}" class="tick" '
                f'text-anchor="middle">{html.escape(str(label))}</text>'
            )

    def render(self, legend: Sequence[tuple], caption: str = "") -> str:
        """``legend`` is ``(name, css_colour)`` pairs. A single series needs no legend box —
        the section title names it."""
        body = "".join(self.parts)
        chips = ""
        if len(legend) > 1:
            chips = '<div class="legend">' + "".join(
                f'<span class="chip"><i style="background:{colour}"></i>'
                f"{html.escape(name)}</span>"
                for name, colour in legend
            ) + "</div>"
        cap = f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ""
        return (
            f'<figure class="chart">{chips}'
            f'<svg viewBox="0 0 {self.width} {self.height}" role="img" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg>{cap}</figure>'
        )


def grouped_bars(
    categories: Sequence[str],
    series: Sequence[Series],
    *,
    places: int = 3,
    baseline: Optional[float] = None,
    baseline_label: str = "",
    unit: str = "",
    caption: str = "",
    height: int = 260,
    width: int = 820,
) -> str:
    """One group of bars per category, one bar per series.

    Bars are anchored to the axis floor, so with a ``baseline`` of 0.5 the eye reads distance
    from the line rather than bar length — which is the honest way to show a win rate.
    """
    series = list(series)[:SERIES_SLOTS]
    values = [v for s in series for v in s.values if v is not None and math.isfinite(v)]
    if not values:
        return ""
    spread = []
    for s in series:
        for i, v in enumerate(s.values):
            if v is None or not math.isfinite(v):
                continue
            e = (s.errors[i] if s.errors and s.errors[i] is not None else 0.0) or 0.0
            spread.extend([v - e, v + e])
    low, high = _nice_bounds(min(spread), max(spread))
    if baseline is not None:
        low, high = min(low, baseline), max(high, baseline)
    floor = baseline if baseline is not None else low

    # Room above the plot for the reference label, which lives there rather than on the line.
    canvas = _Canvas(width, height, top=22 if baseline_label else 14)
    canvas.axis(low, high, places)
    band = canvas.plot_width / max(len(categories), 1)
    # 2px of surface between adjacent bars, per the mark spec.
    slot_width = max((band - 10) / max(len(series), 1) - 2, 2)
    base_y = canvas.y_of(floor, low, high)
    for si, s in enumerate(series):
        colour = f"var(--series-{si + 1})"
        for i, value in enumerate(s.values):
            if value is None or not math.isfinite(value):
                continue
            x = canvas.left + band * i + 5 + si * (slot_width + 2)
            y = canvas.y_of(value, low, high)
            top, bottom = min(y, base_y), max(y, base_y)
            title = (
                f"{s.name} · {categories[i]}: {_fmt(value, places)}{unit}"
                if i < len(categories)
                else s.name
            )
            error = s.errors[i] if s.errors and s.errors[i] is not None else None
            if error is not None and math.isfinite(error):
                title += f" ± {_fmt(error, places)}"
            canvas.parts.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{slot_width:.1f}" '
                f'height="{max(bottom - top, 1.0):.1f}" rx="3" fill="{colour}">'
                f"<title>{html.escape(title)}</title></rect>"
            )
            if error is not None and math.isfinite(error) and error > 0:
                cx = x + slot_width / 2
                y0 = canvas.y_of(value - error, low, high)
                y1 = canvas.y_of(value + error, low, high)
                cap = min(slot_width / 2, 4)
                canvas.parts.append(
                    f'<path d="M{cx - cap:.1f} {y1:.1f}H{cx + cap:.1f}M{cx:.1f} {y1:.1f}'
                    f'V{y0:.1f}M{cx - cap:.1f} {y0:.1f}H{cx + cap:.1f}" class="whisker"/>'
                )
    if baseline is not None:
        canvas.rule(baseline, low, high, baseline_label)
    canvas.categories(categories, band)
    return canvas.render(
        [(s.name, f"var(--series-{i + 1})") for i, s in enumerate(series)], caption
    )


def lines(
    categories: Sequence[str],
    series: Sequence[Series],
    *,
    places: int = 1,
    unit: str = "",
    caption: str = "",
    height: int = 260,
    width: int = 820,
) -> str:
    """A trend along an ordered axis. Markers are 8px so a single point is still findable."""
    series = list(series)[:SERIES_SLOTS]
    spread = [v for s in series for v in s.values if v is not None and math.isfinite(v)]
    if not spread:
        return ""
    low, high = _nice_bounds(min(spread), max(spread))
    canvas = _Canvas(width, height)
    canvas.axis(low, high, places)
    band = canvas.plot_width / max(len(categories), 1)
    for si, s in enumerate(series):
        colour = f"var(--series-{si + 1})"
        points = []
        for i, value in enumerate(s.values):
            if value is None or not math.isfinite(value):
                continue
            points.append((canvas.left + band * (i + 0.5), canvas.y_of(value, low, high), i, value))
        if len(points) > 1:
            path = "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y, _, _ in points)
            canvas.parts.append(f'<path d="{path}" fill="none" stroke="{colour}" '
                                f'stroke-width="2" stroke-linejoin="round"/>')
        for x, y, i, value in points:
            label = categories[i] if i < len(categories) else ""
            canvas.parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}" '
                f'stroke="var(--surface-1)" stroke-width="2">'
                f"<title>{html.escape(f'{s.name} · {label}: {_fmt(value, places)}{unit}')}"
                f"</title></circle>"
            )
    canvas.categories(categories, band)
    return canvas.render(
        [(s.name, f"var(--series-{i + 1})") for i, s in enumerate(series)], caption
    )


def intervals(
    categories: Sequence[str],
    series: Sequence[Series],
    *,
    places: int = 3,
    reference: Optional[float] = None,
    reference_label: str = "",
    unit: str = "",
    caption: str = "",
    width: int = 820,
    row_height: int = 22,
) -> str:
    """A dot with a 95% whisker per estimate, laid out horizontally against a reference line.

    The right form when the interesting question is *how far from the reference, relative to
    the uncertainty* — a bar anchored at 0.500 for a score of 0.505 is a one-pixel sliver
    that says nothing, while a dot two whisker-lengths off the line says everything.
    """
    series = list(series)[:SERIES_SLOTS]
    spread = []
    for s in series:
        for i, v in enumerate(s.values):
            if v is None or not math.isfinite(v):
                continue
            e = (s.errors[i] if s.errors and s.errors[i] is not None else 0.0) or 0.0
            if not math.isfinite(e):
                e = 0.0
            spread.extend([v - e, v + e])
    if not spread:
        return ""
    low, high = _nice_bounds(min(spread), max(spread))
    if reference is not None:
        low, high = min(low, reference), max(high, reference)

    rows = len(categories)
    lanes = max(len(series), 1)
    height = 30 + rows * max(row_height * lanes, row_height) + 26
    # Ticks are centred under their gridline, so the axis needs room for half the last
    # label — without it "1.000" is sliced down the middle by the viewBox.
    canvas = _Canvas(width, height, left=88, bottom=30, top=22, right_pad=34)

    def x_of(value: float) -> float:
        if high <= low:
            return canvas.left
        return canvas.left + (value - low) / (high - low) * canvas.plot_width

    for value in _ticks(low, high, 5):
        x = x_of(value)
        canvas.parts.append(
            f'<line x1="{x:.1f}" y1="{canvas.top}" x2="{x:.1f}" y2="{canvas.bottom:.1f}" '
            f'class="grid"/>'
        )
        canvas.parts.append(
            f'<text x="{x:.1f}" y="{canvas.bottom + 16:.1f}" class="tick" '
            f'text-anchor="middle">{_fmt(value, places)}</text>'
        )
    if reference is not None:
        x = x_of(reference)
        canvas.parts.append(
            f'<line x1="{x:.1f}" y1="{canvas.top}" x2="{x:.1f}" y2="{canvas.bottom:.1f}" '
            f'class="rule"/>'
        )
        if reference_label:
            canvas.parts.append(
                f'<text x="{x:.1f}" y="{canvas.top - 8:.1f}" class="rule-label" '
                f'text-anchor="middle">{html.escape(reference_label)}</text>'
            )

    band = canvas.plot_height / max(rows, 1)
    for i, label in enumerate(categories):
        centre = canvas.top + band * (i + 0.5)
        canvas.parts.append(
            f'<text x="{canvas.left - 10}" y="{centre + 3.5:.1f}" class="tick" '
            f'text-anchor="end">{html.escape(str(label))}</text>'
        )
        for si, s in enumerate(series):
            value = s.values[i] if i < len(s.values) else None
            if value is None or not math.isfinite(value):
                continue
            offset = (si - (lanes - 1) / 2) * min(band / lanes, 9)
            y = centre + offset
            colour = f"var(--series-{si + 1})"
            error = s.errors[i] if s.errors and s.errors[i] is not None else None
            title = f"{s.name} · {label}: {_fmt(value, places)}{unit}"
            if error is not None and math.isfinite(error):
                title += f" ± {_fmt(error, places)}"
                x0, x1 = x_of(value - error), x_of(value + error)
                canvas.parts.append(
                    f'<path d="M{x0:.1f} {y - 4:.1f}V{y + 4:.1f}M{x0:.1f} {y:.1f}H{x1:.1f}'
                    f'M{x1:.1f} {y - 4:.1f}V{y + 4:.1f}" stroke="{colour}" '
                    f'stroke-width="1.5" fill="none" opacity="0.75"/>'
                )
            canvas.parts.append(
                f'<circle cx="{x_of(value):.1f}" cy="{y:.1f}" r="4.5" fill="{colour}" '
                f'stroke="var(--surface-1)" stroke-width="1.5">'
                f"<title>{html.escape(title)}</title></circle>"
            )
    return canvas.render(
        [(s.name, f"var(--series-{i + 1})") for i, s in enumerate(series)], caption
    )


def stacked_bars(
    categories: Sequence[str],
    layers: Sequence[Series],
    *,
    caption: str = "",
    height: int = 240,
    width: int = 820,
    palette: Optional[Sequence[str]] = None,
) -> str:
    """Parts of a whole, one bar per category. Values are shares and must sum to 1."""
    layers = list(layers)
    if not layers:
        return ""
    canvas = _Canvas(width, height)
    canvas.axis(0.0, 1.0, 1)
    band = canvas.plot_width / max(len(categories), 1)
    bar = min(band - 8, 46)
    colours = list(palette) if palette else [f"var(--series-{i + 1})" for i in range(len(layers))]
    for i, label in enumerate(categories):
        x = canvas.left + band * i + (band - bar) / 2
        cursor = 0.0
        for li, layer in enumerate(layers):
            value = layer.values[i] if i < len(layer.values) else None
            if value is None or not math.isfinite(value) or value <= 0:
                continue
            y1 = canvas.y_of(cursor, 0.0, 1.0)
            y0 = canvas.y_of(cursor + value, 0.0, 1.0)
            cursor += value
            # 2px of surface between segments, so a thin slice still reads as its own.
            canvas.parts.append(
                f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bar:.1f}" '
                f'height="{max(y1 - y0 - 2, 1.0):.1f}" rx="2" '
                f'fill="{colours[li % len(colours)]}">'
                f"<title>{html.escape(f'{label} · {layer.name}: {value * 100:.1f}%')}"
                f"</title></rect>"
            )
    canvas.categories(categories, band)
    return canvas.render(
        [(layer.name, colours[i % len(colours)]) for i, layer in enumerate(layers)], caption
    )
