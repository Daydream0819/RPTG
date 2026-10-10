import argparse
import csv
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "code" / "results" / "merged" / "merged_w_sweep_with_adg_satdrift.csv"
OUT_DIR = ROOT / "RPTG_AAAI2027_Submission" / "Figures" / "GuidanceScaleSweep"
PDF_PATH = OUT_DIR / "fig_guidance_scale_sweep_1x4.pdf"
SVG_PATH = OUT_DIR / "fig_guidance_scale_sweep_1x4.svg"
TEX_PATH = OUT_DIR / "fig_guidance_scale_sweep_1x4.tex"
TITLE_FONT = "TimesNewRoman"

try:
    pdfmetrics.registerFont(
        TTFont(TITLE_FONT, "/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    )
except Exception:
    TITLE_FONT = "Times-Roman"


METHOD_LABELS = {
    "cfg": "CFG",
    "adg": "ADG",
    "apg": "APG",
    "fdg": "FDG",
    "rptg": "RPTG",
    "rptg_hr": "RPTG-HR",
    "rptg_apg": "RPTG",
    "rptg_hr_apg": "RPTG-HR",
}

METHOD_ORDER = ["cfg", "adg", "apg", "fdg", "rptg", "rptg_hr"]

COLORS = {
    "cfg": colors.HexColor("#000000"),
    "adg": colors.HexColor("#009E73"),
    "apg": colors.HexColor("#0072B2"),
    "fdg": colors.HexColor("#7B3294"),
    "rptg": colors.HexColor("#D62728"),
    "rptg_hr": colors.HexColor("#D62728"),
}

LINE_DASH = {
    "cfg": None,
    "adg": [4.2, 1.8],
    "apg": [2.4, 1.7],
    "fdg": [1.0, 1.4],
    "rptg": None,
    "rptg_hr": [4.6, 1.8],
}

METRICS = [
    ("fid", "FID", "lower is better", False),
    ("clip_mean", "CLIP", "higher is better", True),
    ("saturation_mean", "Saturation", "diagnostic", False),
    ("contrast_mean", "Contrast", "lower is better", False),
]

Y_AXIS = {
    "fid": (70.0, 76.0, [70.0, 72.0, 74.0, 76.0]),
    "clip_mean": (0.340, 0.358, [0.340, 0.345, 0.350, 0.355]),
    "saturation_mean": (0.18, 0.60, [0.20, 0.30, 0.40, 0.50, 0.60]),
    "contrast_mean": (0.16, 0.32, [0.17, 0.21, 0.25, 0.29, 0.32]),
}

CLIPPED_AXIS = {
    "fid": True,
    "clip_mean": True,
    "saturation_mean": False,
    "contrast_mean": True,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot guidance-scale robustness from a merged sweep CSV."
    )
    parser.add_argument("--data_path", type=Path, default=DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def configure_paths(data_path, output_dir):
    global DATA_PATH, OUT_DIR, PDF_PATH, SVG_PATH, TEX_PATH
    DATA_PATH = data_path
    OUT_DIR = output_dir
    PDF_PATH = OUT_DIR / "fig_guidance_scale_sweep_1x4.pdf"
    SVG_PATH = OUT_DIR / "fig_guidance_scale_sweep_1x4.svg"
    TEX_PATH = OUT_DIR / "fig_guidance_scale_sweep_1x4.tex"


def load_rows():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            method = row["display_method"] if row.get("display_method") else row["method"]
            if method in ("rptg_apg", "rptg"):
                method = "rptg"
            elif method in ("rptg_hr_apg", "rptg_hr"):
                method = "rptg_hr"
            row["method_norm"] = method
            row["w"] = float(row["w"])
            for key, *_ in METRICS:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def nice_range(values, pad_fraction=0.07):
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        delta = abs(lo) * 0.1 if lo else 1.0
        return lo - delta, hi + delta
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def draw_text_centered(c, text, x, y, size=7, font="Helvetica"):
    c.setFont(font, size)
    c.drawString(x - stringWidth(text, font, size) / 2, y, text)


def format_tick(value, metric):
    if metric == "clip_mean":
        return f"{value:.3f}"
    if metric in ("saturation_drift_mean", "saturation_mean"):
        return f"{value:.2f}"
    if metric == "contrast_mean":
        return f"{value:.2f}"
    return f"{value:.0f}"


def draw_marker(c, method, x, y, radius=1.45):
    c.setFillColor(COLORS[method])
    c.setStrokeColor(COLORS[method])
    if method == "cfg":
        c.circle(x, y, radius, fill=1, stroke=0)
    elif method == "adg":
        c.setLineWidth(0.6)
        c.line(x - radius * 1.2, y - radius * 1.2, x + radius * 1.2, y + radius * 1.2)
        c.line(x - radius * 1.2, y + radius * 1.2, x + radius * 1.2, y - radius * 1.2)
    elif method == "apg":
        c.rect(x - radius, y - radius, 2 * radius, 2 * radius, fill=1, stroke=0)
    elif method == "fdg":
        path = c.beginPath()
        path.moveTo(x, y + radius * 1.15)
        path.lineTo(x + radius * 1.15, y)
        path.lineTo(x, y - radius * 1.15)
        path.lineTo(x - radius * 1.15, y)
        path.close()
        c.drawPath(path, fill=1, stroke=0)
    elif method == "rptg":
        path = c.beginPath()
        path.moveTo(x, y + radius * 1.25)
        path.lineTo(x + radius * 1.1, y - radius * 0.75)
        path.lineTo(x - radius * 1.1, y - radius * 0.75)
        path.close()
        c.drawPath(path, fill=1, stroke=0)
    else:
        path = c.beginPath()
        path.moveTo(x, y - radius * 1.25)
        path.lineTo(x + radius * 1.1, y + radius * 0.75)
        path.lineTo(x - radius * 1.1, y + radius * 0.75)
        path.close()
        c.drawPath(path, fill=1, stroke=0)


def draw_metric_title(c, title, higher_better, x, y):
    font = TITLE_FONT
    size = 9.0
    if higher_better is None:
        text = title
    else:
        arrow = "↑" if higher_better else "↓"
        text = f"{title} {arrow}"
    c.setFillColor(colors.black)
    c.setFont(font, size)
    c.drawString(x - stringWidth(text, font, size) / 2, y, text)


def draw_line_chart(c, rows, x0, y0, width, height, metric, title, subtitle, higher_better):
    plot_left = x0 + 24
    plot_right = x0 + width - 5
    plot_bottom = y0 + 18
    plot_top = y0 + height - 18
    ws = sorted({row["w"] for row in rows})
    if metric in Y_AXIS:
        y_min, y_max, y_ticks = Y_AXIS[metric]
    else:
        values = [row[metric] for row in rows]
        y_min, y_max = nice_range(values)
        y_ticks = [
            y_min,
            (y_min + y_max) / 4,
            (y_min + y_max) / 2,
            y_min + 3 * (y_max - y_min) / 4,
            y_max,
        ]
    x_min, x_max = min(ws), max(ws)

    def sx(w):
        return plot_left + (w - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(v):
        v = min(max(v, y_min), y_max)
        return plot_bottom + (v - y_min) / (y_max - y_min) * (plot_top - plot_bottom)

    c.setStrokeColor(colors.HexColor("#B8B8B8"))
    c.setDash(1.0, 2.0)
    c.setLineWidth(0.28)
    for tick in y_ticks:
        y = sy(tick)
        c.line(plot_left, y, plot_right, y)
    for w in ws:
        x = sx(w)
        c.line(x, plot_bottom, x, plot_top)
    c.setDash()

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.line(plot_left, plot_bottom, plot_right, plot_bottom)
    c.line(plot_left, plot_bottom, plot_left, plot_top)
    c.line(plot_right, plot_bottom, plot_right, plot_top)
    c.line(plot_left, plot_top, plot_right, plot_top)

    c.setFillColor(colors.black)
    draw_metric_title(c, title, higher_better, (plot_left + plot_right) / 2, y0 + height - 10)
    if CLIPPED_AXIS.get(metric):
        c.setFont("Times-Roman", 5.8)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawRightString(plot_right, plot_top + 2.2, "zoomed")

    c.setFont("Times-Roman", 6.7)
    c.setFillColor(colors.HexColor("#555555"))
    for w in ws:
        label = f"{w:g}"
        draw_text_centered(c, label, sx(w), plot_bottom - 8, 6.2, "Times-Roman")
    for tick in y_ticks:
        c.drawRightString(plot_left - 3, sy(tick) - 2, format_tick(tick, metric))

    c.setFont("Times-Italic", 7.4)
    c.setFillColor(colors.HexColor("#444444"))
    draw_text_centered(c, "w", (plot_left + plot_right) / 2, y0 + 3, 7.2, "Times-Italic")

    for method in METHOD_ORDER:
        method_rows = sorted(
            [row for row in rows if row["method_norm"] == method],
            key=lambda item: item["w"],
        )
        if not method_rows:
            continue
        c.setStrokeColor(COLORS[method])
        c.setFillColor(COLORS[method])
        c.setLineWidth(1.15 if method.startswith("rptg") else 0.95)
        dash = LINE_DASH[method]
        if dash:
            c.setDash(*dash)
        else:
            c.setDash()
        points = [
            (
                sx(row["w"]),
                sy(row[metric]),
                row[metric] > y_max,
                row[metric] < y_min,
            )
            for row in method_rows
        ]
        for (x1, y1, _, _), (x2, y2, _, _) in zip(points, points[1:]):
            c.line(x1, y1, x2, y2)
        c.setDash()
        for x, y, clipped_high, clipped_low in points:
            if clipped_high or clipped_low:
                c.setFillColor(COLORS[method])
                c.setStrokeColor(COLORS[method])
                path = c.beginPath()
                if clipped_high:
                    path.moveTo(x, plot_top + 3.4)
                    path.lineTo(x - 1.7, plot_top - 0.4)
                    path.lineTo(x + 1.7, plot_top - 0.4)
                else:
                    path.moveTo(x, plot_bottom - 3.4)
                    path.lineTo(x - 1.7, plot_bottom + 0.4)
                    path.lineTo(x + 1.7, plot_bottom + 0.4)
                path.close()
                c.drawPath(path, fill=1, stroke=0)
            else:
                draw_marker(c, method, x, y, 1.72 if method.startswith("rptg") else 1.36)


def draw_inset_legend(c, x, y):
    box_w = 45
    box_h = 42
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#B0B0B0"))
    c.setLineWidth(0.35)
    c.rect(x - 2, y - 3, box_w, box_h, fill=1, stroke=1)
    c.setFont("Times-Roman", 5.3)
    for idx, method in enumerate(METHOD_ORDER):
        yy = y + box_h - 7 - idx * 6.4
        c.setStrokeColor(COLORS[method])
        c.setFillColor(COLORS[method])
        c.setLineWidth(0.95 if method.startswith("rptg") else 0.75)
        dash = LINE_DASH[method]
        if dash:
            c.setDash(*dash)
        else:
            c.setDash()
        c.line(x + 2, yy, x + 11, yy)
        c.setDash()
        draw_marker(c, method, x + 6.5, yy, 1.38 if method.startswith("rptg") else 1.08)
        c.setFillColor(colors.black)
        c.drawString(x + 14, yy - 1.7, METHOD_LABELS[method])


def draw_bottom_legend(c, x0, y, width):
    item_w = 70
    total_w = item_w * len(METHOD_ORDER)
    start_x = x0 + (width - total_w) / 2
    c.setFont("Times-Roman", 7.0)
    for idx, method in enumerate(METHOD_ORDER):
        x = start_x + idx * item_w
        c.setStrokeColor(COLORS[method])
        c.setFillColor(COLORS[method])
        c.setLineWidth(1.15 if method.startswith("rptg") else 0.92)
        dash = LINE_DASH[method]
        if dash:
            c.setDash(*dash)
        else:
            c.setDash()
        c.line(x, y, x + 14, y)
        c.setDash()
        draw_marker(c, method, x + 7, y, 1.42 if method.startswith("rptg") else 1.18)
        c.setFillColor(colors.black)
        c.drawString(x + 18, y - 2.1, METHOD_LABELS[method])


def draw_summary_table(c, rows, x0, y0, width, height):
    c.setFillColor(colors.black)
    draw_text_centered(c, "w = 15 Summary", x0 + width / 2, y0 + height - 10, 8.3, "Times-Bold")

    rows15 = {row["method_norm"]: row for row in rows if math.isclose(row["w"], 15.0)}
    table_x = x0 + 6
    table_y = y0 + 20
    col_widths = [35, 28, 28, 25]
    row_h = 12
    headers = ["Method", "FID↓", "CLIP↑", "Sat.D↓"]

    best_fid = min(rows15[m]["fid"] for m in METHOD_ORDER)
    best_clip = max(rows15[m]["clip_mean"] for m in METHOD_ORDER)
    best_sat = min(rows15[m]["saturation_drift_mean"] for m in METHOD_ORDER)

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.55)
    c.setFillColor(colors.white)
    c.rect(table_x, table_y + 5 * row_h, sum(col_widths), row_h, fill=1, stroke=0)

    x = table_x
    c.setFont("Times-Bold", 6.3)
    c.setFillColor(colors.HexColor("#222222"))
    for header, cw in zip(headers, col_widths):
        draw_text_centered(c, header, x + cw / 2, table_y + 5 * row_h + 4, 6.3, "Times-Bold")
        x += cw

    for ridx, method in enumerate(METHOD_ORDER):
        y = table_y + (4 - ridx) * row_h
        row = rows15[method]
        values = [
            METHOD_LABELS[method],
            f"{row['fid']:.2f}",
            f"{row['clip_mean']:.4f}",
            f"{row['saturation_drift_mean']:.3f}",
        ]
        x = table_x
        for cidx, (value, cw) in enumerate(zip(values, col_widths)):
            bold = (
                cidx == 1 and math.isclose(row["fid"], best_fid)
                or cidx == 2 and math.isclose(row["clip_mean"], best_clip)
                or cidx == 3 and math.isclose(row["saturation_drift_mean"], best_sat)
            )
            font = "Times-Bold" if bold else "Times-Roman"
            c.setFont(font, 6.1)
            c.setFillColor(COLORS[method] if cidx == 0 else colors.HexColor("#222222"))
            draw_text_centered(c, value, x + cw / 2, y + 4, 6.1, font)
            x += cw

    total_w = sum(col_widths)
    total_h = 6 * row_h
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.55)
    for i in (0, 5, 6):
        y = table_y + i * row_h
        c.line(table_x, y, table_x + total_w, y)
    c.setStrokeColor(colors.HexColor("#C0C0C0"))
    c.setLineWidth(0.3)
    for i in range(1, 5):
        y = table_y + i * row_h
        c.line(table_x, y, table_x + total_w, y)


def make_pdf(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width = 600
    height = 130
    margin = 5
    gap = 6
    panel_w = (width - 2 * margin - 3 * gap) / 4
    panel_h = 116
    panel_y = 13

    c = canvas.Canvas(str(PDF_PATH), pagesize=(width, height))
    c.setTitle("Guidance-scale robustness")
    for idx, (metric, title, subtitle, higher) in enumerate(METRICS):
        x0 = margin + idx * (panel_w + gap)
        draw_line_chart(c, rows, x0, panel_y, panel_w, panel_h, metric, title, subtitle, higher)
    draw_bottom_legend(c, margin, 7, width - 2 * margin)
    c.showPage()
    c.save()


def make_svg_placeholder():
    # Compact vector wrapper for quick browser preview. The PDF is the canonical paper figure.
    SVG_PATH.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="1044" height="310" viewBox="0 0 1044 310">
<rect width="1044" height="310" fill="white"/>
<text x="522" y="150" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="22">
Guidance-scale sweep figure generated as PDF.
</text>
<text x="522" y="180" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="16" fill="#555">
Use fig_guidance_scale_sweep_1x4.pdf in LaTeX.
</text>
</svg>
""",
        encoding="utf-8",
    )


def make_tex():
    TEX_PATH.write_text(
        r"""\begin{figure*}[t]
  \centering
  \makebox[\textwidth][c]{\includegraphics[width=\textwidth]{Figures/GuidanceScaleSweep/fig_guidance_scale_sweep_1x4.pdf}}
  \caption{Guidance-scale robustness on SD3 using 1,000 MS-COCO validation prompts. We compare CFG, ADG, APG, FDG, RPTG, and RPTG-HR across guidance scales \(w \in \{3,5,7.5,10,12.5,15\}\) using FID, CLIP Score, Saturation, and Contrast. Zoomed axes are used to make differences among artifact-control methods visible; boundary triangles indicate values outside the displayed range. RPTG and RPTG-HR maintain low artifacts under strong guidance while preserving text-image alignment.}
  \label{fig:guidance_scale_sweep}
\end{figure*}
""",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    configure_paths(args.data_path, args.output_dir)
    rows = load_rows()
    make_pdf(rows)
    make_svg_placeholder()
    make_tex()
    print(PDF_PATH)
    print(TEX_PATH)


if __name__ == "__main__":
    main()
