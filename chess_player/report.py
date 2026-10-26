"""Render a GameAnalysis as Markdown, JSON or a standalone HTML page."""

from __future__ import annotations

import html
import json

from .analysis import LABELS_FA, GameAnalysis

_CLASS_ORDER = ["best", "excellent", "good", "inaccuracy", "mistake", "blunder"]


def fmt_cp(cp: int) -> str:
    """Centipawns -> pawns, e.g. 137 -> '+1.37'. Mate-height values clamp."""
    if abs(cp) >= 9000:
        return "#M" if cp > 0 else "-M"
    return f"{cp / 100:+.2f}"


def to_json(analysis: GameAnalysis) -> str:
    return json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2)


def to_markdown(analysis: GameAnalysis) -> str:
    h = analysis.headers
    title = f"{h.get('White', '?')} vs {h.get('Black', '?')}"
    lines = [
        f"# تحلیل بازی: {title}",
        "",
        f"- **نتیجه:** {analysis.result}",
        f"- **موتور:** {analysis.engine_name} (عمق {analysis.depth})",
        f"- **دقت سفید:** {analysis.accuracy('white')}٪  |  ACPL: {analysis.acpl('white')}",
        f"- **دقت سیاه:** {analysis.accuracy('black')}٪  |  ACPL: {analysis.acpl('black')}",
        "",
        "| # | حرکت | ارزیابی بعد (سفید) | طبقه‌بندی | افت Win% | بهترین حرکت موتور |",
        "|---|------|--------------------|-----------|----------|-------------------|",
    ]
    for m in analysis.moves:
        num = f"{m.move_number}." if m.color == "white" else f"{m.move_number}..."
        best = m.best_move_san or "-"
        if m.classification in ("best",):
            best = "—"
        lines.append(
            f"| {num} | {m.san} | {fmt_cp(m.eval_after_white_cp)} | "
            f"{LABELS_FA[m.classification]} | {m.win_loss:.1f} | {best} |"
        )

    lines += ["", "## لحظه‌های حساس (اشتباه و بدتر)", ""]
    critical = [m for m in analysis.moves if m.classification in ("mistake", "blunder")]
    if not critical:
        lines.append("اشتباه یا اشتباه فاحشی پیدا نشد. 🎉")
    for m in critical:
        num = f"{m.move_number}." if m.color == "white" else f"{m.move_number}..."
        line = " ".join(m.best_line_san)
        lines.append(
            f"- **{num} {m.san}** ({m.label_fa}) — موتور می‌خواست: "
            f"`{m.best_move_san}` با ادامهٔ `{line}` | FEN قبل از حرکت: `{m.fen_before}`"
        )
    lines.append("")
    return "\n".join(lines)


def _eval_svg(analysis: GameAnalysis, width: int = 720, height: int = 180) -> str:
    """Eval graph: clipped to +-500 cp, White is up."""
    if not analysis.moves:
        return ""
    mid = height / 2
    points_white, points_black = [], []
    for i, m in enumerate(analysis.moves):
        x = (i / max(1, len(analysis.moves) - 1)) * width if len(analysis.moves) > 1 else width / 2
        y = mid - max(-500, min(500, m.eval_after_white_cp)) / 500 * (mid - 4)
        points_white.append((x, y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points_white)
    area = f"0,{mid} " + poly + f" {width},{mid}"
    return f"""<svg viewBox="0 0 {width} {height}" style="width:100%;background:#222;border-radius:8px">
  <polygon points="{area}" fill="#e8e8e8" opacity="0.85"/>
  <line x1="0" y1="{mid}" x2="{width}" y2="{mid}" stroke="#888" stroke-width="1"/>
  <polyline points="{poly}" fill="none" stroke="#4caf50" stroke-width="2"/>
</svg>"""


_BADGE_COLORS = {
    "best": "#2e7d32", "excellent": "#43a047", "good": "#7cb342",
    "inaccuracy": "#f9a825", "mistake": "#ef6c00", "blunder": "#c62828",
}


def to_html(analysis: GameAnalysis) -> str:
    h = {k: html.escape(v) for k, v in analysis.headers.items()}
    rows = []
    for m in analysis.moves:
        num = f"{m.move_number}." if m.color == "white" else f"{m.move_number}…"
        color = _BADGE_COLORS[m.classification]
        rows.append(
            f"<tr><td>{num}</td><td><b>{html.escape(m.san)}</b></td>"
            f"<td>{fmt_cp(m.eval_after_white_cp)}</td>"
            f'<td><span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:10px">{LABELS_FA[m.classification]}</span></td>'
            f"<td>{m.win_loss:.1f}</td><td>{m.accuracy:.0f}٪</td>"
            f"<td>{html.escape(m.best_move_san or '—')}</td></tr>"
        )
    counts_rows = []
    for color_name, fa in (("white", "سفید"), ("black", "سیاه")):
        c = analysis.counts(color_name)
        cells = "".join(f"<td>{c.get(k, 0)}</td>" for k in _CLASS_ORDER)
        counts_rows.append(f"<tr><td>{fa}</td>{cells}</tr>")
    header_cells = "".join(f"<th>{LABELS_FA[k]}</th>" for k in _CLASS_ORDER)

    return f"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<title>تحلیل بازی شطرنج</title>
<style>body{{font-family:Tahoma,sans-serif;background:#181818;color:#eee;max-width:860px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #333;padding:5px;text-align:center}}
.card{{background:#242424;border-radius:10px;padding:14px;margin:12px 0}}</style></head><body>
<h1>♞ {h.get('White','سفید')} – {h.get('Black','سیاه')} ({html.escape(analysis.result)})</h1>
<div class="card">موتور: {html.escape(analysis.engine_name)} — عمق {analysis.depth}<br>
دقت سفید: <b>{analysis.accuracy('white')}٪</b> (ACPL {analysis.acpl('white')}) ·
دقت سیاه: <b>{analysis.accuracy('black')}٪</b> (ACPL {analysis.acpl('black')})</div>
<div class="card"><h3>نمودار ارزیابی (سفید بالا)</h3>{_eval_svg(analysis)}</div>
<div class="card"><table><tr><th></th>{header_cells}</tr>{''.join(counts_rows)}</table></div>
<div class="card"><table>
<tr><th>#</th><th>حرکت</th><th>ارزیابی</th><th>کیفیت</th><th>افت Win%</th><th>دقت</th><th>پیشنهاد موتور</th></tr>
{''.join(rows)}</table></div>
<p style="color:#999">تحلیل پس از پایان بازی — برای تمرین و یادگیری. استفاده از موتور حین بازی آنلاین درجه‌دار تقلب است.</p>
</body></html>"""
