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
DEFAULT_DATA = ROOT / "code" / "results" / "merged" / "merged_w_sweep_with_adg_satdrift.csv"
DEFAULT_OUT = ROOT / "code" / "results" / "figures" / "guidance_scale_sweep_with_adg_clean"

FONT = "Times-Roman"
FONT_BOLD = "Times-Bold"
try:
    pdfmetrics.registerFont(
        TTFont("TimesNewRoman", "/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("TimesNewRomanBold", "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf")
    )
    FONT = "TimesNewRoman"
    FONT_BOLD = "TimesNewRomanBold"
except Exception:
    pass


METHODS = ["cfg", "adg", "apg", "fdg", "rptg", "rptg_hr"]
LABELS = {
    "cfg": "CFG",
    "adg": "ADG",
    "apg": "APG",
    "fdg": "FDG",
    "rptg": "RPTG",
    "rptg_hr": "RPTG-HR",
}
ALIASES = {
    "rptg_apg": "rptg",
    "rptg_hr_apg": "rptg_hr",
}

# Color-blind friendly, with restrained baselines and warm emphasis for ours.
PALETTE = {
    "cfg": colors.HexColor("#7A7A7A"),
    "adg": colors.HexColor("#8E63A9"),
    "apg": colors.HexColor("#D18F00"),
    "fdg": colors.HexColor("#2C8F7B"),
    "rptg": colors.HexColor("#C83E36"),
    "rptg_hr": colors.HexColor("#8F1D2C"),
}
LINE_STYLE = {
    "cfg": [3.0, 2.0],
    "adg": [1.2, 1.8],
    "apg": [5.0, 2.0],
    "fdg": [6.0, 2.2, 1.2, 2.2],
    "rptg": None,
    "rptg_hr": None,
}

PANEL_KEYS = ["fid", "clip_mean", "saturation_mean", "contrast_mean"]
PANELS_2X2 = [
    ("fid", "FID", (47.0, 94.0), [50, 60, 70, 80, 90], False),
    ("clip_mean", "CLIP Score", (0.322, 0.359), [0.325, 0.335, 0.345, 0.355], False),
    ("saturation_mean", "Saturation", (0.17, 0.61), [0.20, 0.30, 0.40, 0.50, 0.60], False),
    ("contrast_mean", "Contrast", (0.145, 0.325), [0.15, 0.20, 0.25, 0.30], False),
]
PANELS_1X4 = [
    ("fid", "FID", (69.0, 79.0), [69, 72, 75, 78], True),
    ("clip_mean", "CLIP Score", (0.344, 0.358), [0.345, 0.350, 0.355], True),
    ("saturation_mean", "Saturation", (0.18, 0.60), [0.20, 0.30, 0.40, 0.50, 0.60], False),
    ("contrast_mean", "Contrast", (0.145, 0.315), [0.15, 0.20, 0.25, 0.30], False),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--name", default="fig_guidance_scale_sweep_clean_2x2")
    parser.add_argument("--layout", choices=("2x2", "1x4"), default="2x2")
    return parser.parse_args()


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            method = row.get("display_method") or row["method"]
            method = ALIASES.get(method, method)
            if method not in METHODS:
                continue
            item = {"method": method, "w": float(row["w"])}
            for key in PANEL_KEYS:
                item[key] = float(row[key])
            rows.append(item)
    return rows


def text_width(text, font, size):
    return stringWidth(text, font, size)


def centered(c, text, x, y, size=7.0, font=FONT):
    c.setFont(font, size)
    c.drawString(x - text_width(text, font, size) / 2, y, text)


def tick_label(value, key):
    if key == "fid":
        return f"{value:.0f}"
    if key == "clip_mean":
        return f"{value:.3f}"
    return f"{value:.2f}"


def marker(c, method, x, y, r=2.0, fill=True):
    c.setStrokeColor(PALETTE[method])
    c.setFillColor(PALETTE[method] if fill else colors.white)
    c.setLineWidth(0.8)
    if method == "cfg":
        c.circle(x, y, r, fill=0, stroke=1)
    elif method == "adg":
        c.setLineWidth(1.0)
        c.line(x - r, y - r, x + r, y + r)
        c.line(x - r, y + r, x + r, y - r)
    elif method == "apg":
        c.rect(x - r, y - r, 2 * r, 2 * r, fill=1, stroke=0)
    elif method == "fdg":
        path = c.beginPath()
        path.moveTo(x, y + 1.25 * r)
        path.lineTo(x + 1.25 * r, y)
        path.lineTo(x, y - 1.25 * r)
        path.lineTo(x - 1.25 * r, y)
        path.close()
        c.drawPath(path, fill=1, stroke=0)
    elif method == "rptg":
        path = c.beginPath()
        path.moveTo(x, y + 1.35 * r)
        path.lineTo(x + 1.18 * r, y - 0.8 * r)
        path.lineTo(x - 1.18 * r, y - 0.8 * r)
        path.close()
        c.drawPath(path, fill=1, stroke=0)
    else:
        path = c.beginPath()
        path.moveTo(x, y - 1.35 * r)
        path.lineTo(x + 1.18 * r, y + 0.8 * r)
        path.lineTo(x - 1.18 * r, y + 0.8 * r)
        path.close()
        c.drawPath(path, fill=1, stroke=0)


def draw_legend(c, x0, y0, width):
    item_w = width / 6
    c.setFont(FONT, 7.8)
    for idx, method in enumerate(METHODS):
        x = x0 + idx * item_w + 6
        y = y0
        c.setStrokeColor(PALETTE[method])
        c.setLineWidth(1.45 if method.startswith("rptg") else 1.0)
        dash = LINE_STYLE[method]
        if dash:
            c.setDash(dash)
        else:
            c.setDash()
        c.line(x, y, x + 13, y)
        c.setDash()
        marker(c, method, x + 6.5, y, 1.65 if method.startswith("rptg") else 1.45)
        c.setFillColor(colors.HexColor("#222222"))
        c.drawString(x + 17, y - 2.2, LABELS[method])


def draw_clipped_marker(c, method, x, y, high=True):
    c.setFillColor(PALETTE[method])
    c.setStrokeColor(PALETTE[method])
    path = c.beginPath()
    if high:
        path.moveTo(x, y + 3.0)
        path.lineTo(x - 1.8, y - 0.4)
        path.lineTo(x + 1.8, y - 0.4)
    else:
        path.moveTo(x, y - 3.0)
        path.lineTo(x - 1.8, y + 0.4)
        path.lineTo(x + 1.8, y + 0.4)
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def draw_panel(c, rows, spec, x0, y0, width, height, compact=False):
    key, title, (y_min, y_max), y_ticks, clipped_axis = spec
    left = x0 + (24 if compact else 34)
    right = x0 + width - (5 if compact else 10)
    bottom = y0 + (19 if compact else 24)
    top = y0 + height - (18 if compact else 24)
    ws = sorted({row["w"] for row in rows})
    x_min, x_max = min(ws), max(ws)

    def sx(w):
        return left + (w - x_min) / (x_max - x_min) * (right - left)

    def sy(value):
        value = min(max(value, y_min), y_max)
        return bottom + (value - y_min) / (y_max - y_min) * (top - bottom)

    # High-guidance region, where robustness differences matter most.
    c.setFillColor(colors.HexColor("#F4F4F4"))
    c.rect(sx(10), bottom, right - sx(10), top - bottom, fill=1, stroke=0)

    c.setStrokeColor(colors.HexColor("#D7D7D7"))
    c.setLineWidth(0.35)
    c.setDash(1.0, 2.0)
    for tick in y_ticks:
        y = sy(tick)
        c.line(left, y, right, y)
    for w in ws:
        c.line(sx(w), bottom, sx(w), top)
    c.setDash()

    c.setStrokeColor(colors.HexColor("#202020"))
    c.setLineWidth(0.55)
    c.rect(left, bottom, right - left, top - bottom, fill=0, stroke=1)

    c.setFillColor(colors.black)
    centered(c, title, (left + right) / 2, y0 + height - (11 if compact else 13), 8.2 if compact else 9.1, FONT_BOLD)
    if clipped_axis:
        c.setFont(FONT, 5.8 if compact else 6.4)
        c.setFillColor(colors.HexColor("#6A6A6A"))
        c.drawRightString(right, y0 + height - (11 if compact else 13), "zoomed")

    c.setFont(FONT, 6.3 if compact else 7.0)
    c.setFillColor(colors.HexColor("#4F4F4F"))
    for tick in y_ticks:
        c.drawRightString(left - 4, sy(tick) - 2.1, tick_label(tick, key))
    for w in ws:
        centered(c, f"{w:g}", sx(w), bottom - (9 if compact else 11), 5.8 if compact else 6.9, FONT)
    centered(c, "w", (left + right) / 2, y0 + (3 if compact else 5), 6.7 if compact else 7.2, FONT)

    for method in METHODS:
        series = sorted([row for row in rows if row["method"] == method], key=lambda row: row["w"])
        if not series:
            continue
        points = [
            (sx(row["w"]), sy(row[key]), row[key] > y_max, row[key] < y_min)
            for row in series
        ]
        c.setStrokeColor(PALETTE[method])
        c.setLineWidth(1.65 if method.startswith("rptg") else 1.05)
        dash = LINE_STYLE[method]
        if dash:
            c.setDash(dash)
        else:
            c.setDash()
        for (x1, y1, _, _), (x2, y2, _, _) in zip(points, points[1:]):
            c.line(x1, y1, x2, y2)
        c.setDash()
        for x, y, clipped_high, clipped_low in points:
            if clipped_axis and clipped_high:
                draw_clipped_marker(c, method, x, top, high=True)
            elif clipped_axis and clipped_low:
                draw_clipped_marker(c, method, x, bottom, high=False)
            else:
                marker(c, method, x, y, 1.75 if method.startswith("rptg") else 1.45)


def make_pdf(rows, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    width = 520
    height = 330
    margin_x = 18
    top_legend_y = height - 20
    panel_gap_x = 18
    panel_gap_y = 18
    panel_w = (width - 2 * margin_x - panel_gap_x) / 2
    panel_h = 130
    bottom_y = 14
    top_y = bottom_y + panel_h + panel_gap_y

    c = canvas.Canvas(str(out_pdf), pagesize=(width, height))
    c.setTitle("Guidance-scale robustness with ADG")
    draw_legend(c, margin_x + 4, top_legend_y, width - 2 * margin_x - 8)
    positions = [
        (margin_x, top_y),
        (margin_x + panel_w + panel_gap_x, top_y),
        (margin_x, bottom_y),
        (margin_x + panel_w + panel_gap_x, bottom_y),
    ]
    for spec, (x0, y0) in zip(PANELS_2X2, positions):
        draw_panel(c, rows, spec, x0, y0, panel_w, panel_h)
    c.showPage()
    c.save()


def make_pdf_1x4(rows, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    width = 522
    height = 142
    margin_x = 8
    gap = 8
    panel_w = (width - 2 * margin_x - 3 * gap) / 4
    panel_h = 108
    panel_y = 8

    c = canvas.Canvas(str(out_pdf), pagesize=(width, height))
    c.setTitle("Guidance-scale robustness with ADG")
    draw_legend(c, margin_x + 2, height - 13, width - 2 * margin_x - 4)
    for idx, spec in enumerate(PANELS_1X4):
        x0 = margin_x + idx * (panel_w + gap)
        draw_panel(c, rows, spec, x0, panel_y, panel_w, panel_h, compact=True)
    c.showPage()
    c.save()


def main():
    args = parse_args()
    rows = load_rows(args.data_path)
    out_pdf = args.output_dir / f"{args.name}.pdf"
    if args.layout == "1x4":
        make_pdf_1x4(rows, out_pdf)
    else:
        make_pdf(rows, out_pdf)
    print(out_pdf)


if __name__ == "__main__":
    main()
