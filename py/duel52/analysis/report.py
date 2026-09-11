"""Rendering the sections to Markdown and to HTML.

Two serialisers over one list of :class:`~duel52.analysis.metrics.Section`, because the two
are read in different places and neither is a substitute for the other:

* **Markdown** is the canonical form — greppable, diffable, and a row of it can be pasted
  into `FINDINGS.md`. It carries no figures, because a Markdown file cannot inline an SVG in
  a way that survives being viewed anywhere.
* **HTML** is the same tables plus every figure inline, in one self-contained file that opens
  in a browser with nothing installed. Its tables also sort on any column, which is the one
  thing it can offer that a static form cannot — a per-rank table is thirteen rows whose
  interesting question is usually *which card is at the top of this column*.

Every figure sits beside the table it draws, which is also what discharges the contrast
relief rule: no value is ever available only as a colour.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

from .corpus import Corpus
from .metrics import METRICS, Context, Figure, Note, Section, Table


def build(corpora: List[Corpus], ctx: Context) -> List[Section]:
    sections = []
    for fn in METRICS:
        section = fn(corpora, ctx)
        if section is not None:
            sections.append(section)
    return sections


def _header_facts(corpora: Sequence[Corpus]) -> List[tuple]:
    meta = corpora[0].meta
    seeds = sorted(
        (chunk.meta["first_seed"], chunk.meta["first_seed"] + chunk.meta["deals"] - 1)
        for corpus in corpora
        for chunk in corpus.chunks
    )
    span = f"{seeds[0][0]}–{seeds[-1][1]}" if seeds else "—"
    return [
        ("variant", meta["variant"]),
        ("ruleset", meta["rules_label"]),
        ("the 2's power", meta["two_power"]),
        ("lanes", f"{meta['lanes']}, {meta['lanes_to_win']} to win"),
        ("hand size", str(meta["hand_size"])),
        ("stalemate", f"{meta['stalemate_quiet_plies']} quiet turns"),
        ("deal seeds", span),
        ("models", ", ".join(c.label for c in corpora)),
        ("games per model", ", ".join(f"{c.n_games:,}" for c in corpora)),
    ]


# ==================================================================== markdown ==


def _md_table(table: Table) -> str:
    out = ["| " + " | ".join(table.columns) + " |"]
    out.append("|" + "|".join("---:" if a == "r" else "---" for a in table.align) + "|")
    for row in table.rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")

    text = "\n".join(out)
    if table.caption:
        text += f"\n\n*{table.caption}*"
    return text


def markdown(corpora: List[Corpus], sections: List[Section], ctx: Context) -> str:
    meta = corpora[0].meta
    lines = [
        f"# Duel 52 — self-play analysis: `{meta['variant']}`",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · "
        f"corpus schema {meta['schema']} · figures in "
        f"[`{ctx.dataset}.html`]({ctx.dataset}.html)",
        "",
    ]
    for key, value in _header_facts(corpora):
        lines.append(f"- **{key}** — {value}")
    lines.append("")
    lines.append(
        "Every agent plays **itself**. Intervals are 95% and clustered on the deal, since "
        "both games of a colour-paired deal hold the same cards. Turn numbers are the "
        "player's own turns, 1-based."
    )
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    for section in sections:
        lines.append(f"- [{section.title}](#{_slug(section.title)})")
    lines.append("")

    for section in sections:
        lines.append(f"## {section.title}")
        lines.append("")
        if section.note:
            lines.append(section.note)
            lines.append("")
        for block in section.blocks:
            if isinstance(block, Table):
                lines.append(_md_table(block))
                lines.append("")
            elif isinstance(block, Note):
                lines.append(f"> {block.text}")
                lines.append("")
        figures = sum(1 for b in section.blocks if isinstance(b, Figure) and b.svg)
        if figures:
            what = "figure is" if figures == 1 else f"{figures} figures are"
            lines.append(
                f"*The {what} in "
                f"[`{ctx.dataset}.html`]({ctx.dataset}.html#{_slug(section.title)}).*"
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


# ======================================================================== html ==

_STYLE = """
:root {
  color-scheme: light;
  --surface-0: #ffffff;
  --surface-1: #fcfcfb;
  --surface-2: #f4f3f0;
  --border:    #e3e1dc;
  --grid:      #ebe9e4;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #78766f;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --fate-1: #2a78d6;
  --fate-2: #eb6834;
  --fate-3: #1baf7a;
  --fate-4: #4a3aa7;
  --fate-5: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #131312;
    --surface-1: #1a1a19;
    --surface-2: #232320;
    --border:    #33322e;
    --grid:      #2b2a27;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #96948a;
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --fate-1: #3987e5;
    --fate-2: #d95926;
    --fate-3: #199e70;
    --fate-4: #9085e9;
    --fate-5: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #131312;
  --surface-1: #1a1a19;
  --surface-2: #232320;
  --border:    #33322e;
  --grid:      #2b2a27;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #96948a;
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --series-4: #c98500;
  --fate-1: #3987e5;
  --fate-2: #d95926;
  --fate-3: #199e70;
  --fate-4: #9085e9;
  --fate-5: #e66767;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--surface-0);
  color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 1040px; margin: 0 auto; padding: 40px 24px 96px; }
h1 { font-size: 26px; letter-spacing: -0.01em; margin: 0 0 4px; }
h2 {
  font-size: 19px; letter-spacing: -0.01em; margin: 56px 0 10px;
  padding-top: 20px; border-top: 1px solid var(--border);
}
h1 + p { color: var(--text-secondary); margin-top: 0; }
p { max-width: 74ch; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1px;
         background: var(--border); border: 1px solid var(--border); border-radius: 8px;
         overflow: hidden; margin: 20px 0 28px; }
.facts div { background: var(--surface-1); padding: 10px 14px; }
.facts dt { color: var(--text-muted); font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.06em; }
.facts dd { margin: 2px 0 0; font-size: 14px; }
nav ol { columns: 2; gap: 32px; padding-left: 20px; color: var(--text-secondary); }
nav a { color: inherit; }
.note { color: var(--text-secondary); max-width: 74ch; }
.note strong { color: var(--text-primary); }
blockquote { border-left: 3px solid var(--border); margin: 16px 0; padding: 2px 0 2px 14px;
             color: var(--text-secondary); max-width: 72ch; }
.scroll { overflow-x: auto; margin: 18px 0; }
table { border-collapse: collapse; font-size: 13.5px; min-width: 100%; }
th, td { padding: 6px 12px; white-space: nowrap; border-bottom: 1px solid var(--border); }
th.l, td.l { text-align: left; }
th.r, td.r { text-align: right; }
thead th { color: var(--text-muted); font-weight: 600; font-size: 11.5px;
           text-transform: uppercase; letter-spacing: 0.05em; }
tbody tr:hover td { background: var(--surface-2); }
/* Sortable headers. The arrow hangs off the side the column is NOT aligned to, so a right-
   aligned header's text still ends flush with the numbers under it. It is always the same
   box whatever it holds, so nothing reflows when the state changes. */
th[aria-sort] { cursor: pointer; user-select: none; }
th[aria-sort]:hover, th[aria-sort]:not([aria-sort="none"]) { color: var(--text-primary); }
th[aria-sort]:focus-visible { outline: 2px solid var(--series-1); outline-offset: -2px; }
th[aria-sort].l::after, th[aria-sort].r::before {
  content: "⇅"; display: inline-block; width: 1em; font-size: 10px;
  text-align: center; opacity: 0; font-weight: 400; letter-spacing: 0;
}
th[aria-sort].l::after  { margin-left: 3px; }
th[aria-sort].r::before { margin-right: 3px; }
th[aria-sort]:hover.l::after, th[aria-sort]:focus-visible.l::after,
th[aria-sort]:hover.r::before, th[aria-sort]:focus-visible.r::before { opacity: 0.45; }
th[aria-sort="ascending"].l::after, th[aria-sort="ascending"].r::before {
  content: "▲"; opacity: 1;
}
th[aria-sort="descending"].l::after, th[aria-sort="descending"].r::before {
  content: "▼"; opacity: 1;
}
caption, figcaption { caption-side: bottom; color: var(--text-muted); font-size: 12.5px;
                      text-align: left; padding-top: 8px; max-width: 74ch; }
.chart { margin: 20px 0 26px; padding: 14px 12px 8px; background: var(--surface-1);
         border: 1px solid var(--border); border-radius: 10px; }
.chart svg { width: 100%; height: auto; display: block; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; padding: 0 4px 10px; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px;
        color: var(--text-secondary); }
.chip i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.grid { stroke: var(--grid); stroke-width: 1; }
.rule { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 4 3; }
.rule-label, .tick { fill: var(--text-muted); font-size: 10.5px;
                     font-family: ui-sans-serif, system-ui, sans-serif; }
.whisker { stroke: var(--text-primary); stroke-width: 1.2; fill: none; opacity: 0.55; }
"""

# Sorting, in the one file and with nothing installed — the same promise the figures make.
#
# Three things it is careful about, each of which was a wrong answer first:
#
# * **A third click restores the document order.** Every table here is written in an order
#   that carries information — the deck for a rank table, the lineage for a model one — so
#   a sort that cannot be undone destroys something the reader came for.
# * **A cell is not a number just because it starts with one.** `gen031@1000` would sort as
#   31. So only a right-aligned column is read numerically, which is exactly the distinction
#   `Table.align` already draws, and a column that means something else entirely (`rank`)
#   says so with `data-s`.
# * **A blank sorts last in both directions.** `—` is "no observations", not "zero", and
#   reversing it to the top of a descending column would read as a result.
_SCRIPT = r"""
(function () {
  var BLANK = /^(?:|—|–|-|n\/a)$/;
  var NUMBER = /[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?/;

  function key(cell) {
    if (!cell) return null;
    if (cell.dataset.s !== undefined) return Number(cell.dataset.s);
    var text = cell.textContent.trim();
    if (BLANK.test(text)) return null;
    if (cell.classList.contains('r')) {
      // The first number, so "0.5170 ± 0.0250" sorts on the estimate rather than the
      // interval, "1.41 h" on the hours, and "43 – 51" on the bottom of the range.
      var found = text.replace(/,/g, '').match(NUMBER);
      if (found) return Number(found[0]);
    }
    return text.toLowerCase();
  }

  function compare(a, b) {
    var an = typeof a === 'number', bn = typeof b === 'number';
    if (an && bn) return a - b;
    if (an !== bn) return an ? -1 : 1;
    return a.localeCompare(b, undefined, { numeric: true });
  }

  function sort(table, column) {
    var head = table.tHead.rows[0];
    var was = head.cells[column].getAttribute('aria-sort');
    var now = was === 'none' ? 'ascending' : was === 'ascending' ? 'descending' : 'none';
    for (var i = 0; i < head.cells.length; i++) {
      head.cells[i].setAttribute('aria-sort', 'none');
    }
    head.cells[column].setAttribute('aria-sort', now);

    var body = table.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    rows.forEach(function (row, i) {
      if (row.dataset.i === undefined) row.dataset.i = i;
    });
    if (now === 'none') {
      rows.sort(function (a, b) { return a.dataset.i - b.dataset.i; });
    } else {
      var dir = now === 'ascending' ? 1 : -1;
      var keys = new Map();
      rows.forEach(function (row) { keys.set(row, key(row.cells[column])); });
      var present = rows.filter(function (row) { return keys.get(row) !== null; });
      var blank = rows.filter(function (row) { return keys.get(row) === null; });
      // The document-order tiebreak is applied after the direction, not reversed with it,
      // so descending is a mirror of ascending and equal rows never shuffle.
      present.sort(function (a, b) {
        return dir * compare(keys.get(a), keys.get(b)) || a.dataset.i - b.dataset.i;
      });
      rows = present.concat(blank);
    }
    rows.forEach(function (row) { body.appendChild(row); });
  }

  function header(node) {
    return node && node.closest ? node.closest('th[aria-sort]') : null;
  }

  document.addEventListener('click', function (event) {
    var th = header(event.target);
    if (th) sort(th.closest('table'), th.cellIndex);
  });

  document.addEventListener('keydown', function (event) {
    if (event.key !== 'Enter' && event.key !== ' ' && event.key !== 'Spacebar') return;
    var th = header(event.target);
    if (!th) return;
    event.preventDefault();
    sort(th.closest('table'), th.cellIndex);
  });
})();
"""


def _html_table(table: Table) -> str:
    """One table, with every column sortable.

    `aria-sort` is both the accessibility contract and the only place the sort state lives —
    the CSS draws the arrow from it and the script cycles it none → ascending → descending →
    none. The third click matters here: these tables are written in an order that means
    something (the deck, the lineage), so sorting has to be undoable.

    A cell carries an explicit `data-s` only where `Table.sort_keys` supplies one. Everything
    else the script reads off the alignment the table already declares.
    """

    def cell(tag: str, value, index: int, extra: str = "") -> str:
        side = table.align[index] if index < len(table.align) else "r"
        return f'<{tag} class="{side}"{extra}>{html.escape(str(value))}</{tag}>'

    head = "".join(
        cell("th", c, i, ' tabindex="0" aria-sort="none"')
        for i, c in enumerate(table.columns)
    )
    body = []
    for r, row in enumerate(table.rows):
        cells = []
        for i, value in enumerate(row):
            keys = table.sort_keys.get(i)
            key = f' data-s="{keys[r]}"' if keys is not None and r < len(keys) else ""
            cells.append(cell("td", value, i, key))
        body.append("<tr>" + "".join(cells) + "</tr>")
    caption = f"<caption>{_inline(table.caption)}</caption>" if table.caption else ""
    return f'<div class="scroll"><table>{caption}<thead><tr>{head}</tr></thead>'\
           f'<tbody>{"".join(body)}</tbody></table></div>'


def _inline(text: str) -> str:
    """The little Markdown the notes actually use: `code`, **bold**, and paragraphs."""
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return out


def _note_html(text: str) -> str:
    parts = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("* "):
            items = "".join(
                f"<li>{_inline(line[2:].strip())}</li>"
                for line in block.splitlines()
                if line.strip().startswith("* ")
            )
            parts.append(f'<ul class="note">{items}</ul>')
        else:
            parts.append(f'<p class="note">{_inline(block)}</p>')
    return "".join(parts)


def html_report(corpora: List[Corpus], sections: List[Section], ctx: Context) -> str:
    meta = corpora[0].meta
    facts = "".join(
        f"<div><dt>{html.escape(k)}</dt><dd>{_inline(str(v))}</dd></div>"
        for k, v in _header_facts(corpora)
    )
    contents = "".join(
        f'<li><a href="#{_slug(s.title)}">{html.escape(s.title)}</a></li>' for s in sections
    )
    body = []
    for section in sections:
        body.append(f'<h2 id="{_slug(section.title)}">{html.escape(section.title)}</h2>')
        if section.note:
            body.append(_note_html(section.note))
        for block in section.blocks:
            if isinstance(block, Table):
                body.append(_html_table(block))
            elif isinstance(block, Figure):
                if block.svg:
                    body.append(block.svg)
            elif isinstance(block, Note):
                body.append(f"<blockquote>{_inline(block.text)}</blockquote>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duel 52 analysis — {html.escape(meta['variant'])}</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<h1>Duel 52 — self-play analysis: {html.escape(meta['variant'])}</h1>
<p>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · corpus schema
{meta['schema']} · tables also in <code>{html.escape(ctx.dataset)}.md</code></p>
<dl class="facts">{facts}</dl>
<p class="note">Every agent plays <strong>itself</strong>. Intervals are 95% and clustered on
the deal, since both games of a colour-paired deal hold the same cards. Turn numbers are the
player's own turns, 1-based. <strong>Click a column heading</strong> to sort the table by it;
a second click reverses it and a third restores the order it was written in.</p>
<nav><ol>{contents}</ol></nav>
{"".join(body)}
</main>
<script>{_SCRIPT}</script>
</body>
</html>
"""


def write(corpora: List[Corpus], ctx: Context, out_dir: Path) -> List[Path]:
    sections = build(corpora, ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{ctx.dataset}.md"
    html_path = out_dir / f"{ctx.dataset}.html"
    md_path.write_text(markdown(corpora, sections, ctx))
    html_path.write_text(html_report(corpora, sections, ctx))
    return [md_path, html_path]
