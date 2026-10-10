import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "code" / "results" / "directional_energy_summary.csv"
OUT_DIR = ROOT / "RPTG_AAAI2027_Submission" / "Figures" / "DirectionalEnergy"
RESULT_DIR = ROOT / "code" / "results" / "figures" / "directional_energy_reportlab"

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


def load_values():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in (
                "cfg_scale",
                "weighted_energy_mean",
                "positive_radial_coupling_mean",
            ):
                row[key] = float(row[key])
            rows.append(row)

    scales = sorted({row["cfg_scale"] for row in rows})

    def value(scale, component, column):
        for row in rows:
            if row["cfg_scale"] == scale and row["component"] == component:
                return row[column]
        raise KeyError((scale, component, column))

    values = {
        "all_radial": [],
        "low_radial": [],
        "high_radial": [],
        "all_tangential": [],
        "low_tangential": [],
        "high_tangential": [],
    }
    for scale in scales:
        low_pos = value(scale, "low_positive_radial", "weighted_energy_mean")
        low_neg = value(scale, "low_negative_radial", "weighted_energy_mean")
        low_tan = value(scale, "low_tangential", "weighted_energy_mean")
        high_pos = value(scale, "high_positive_radial", "weighted_energy_mean")
        high_neg = value(scale, "high_negative_radial", "weighted_energy_mean")
        high_tan = value(scale, "high_tangential", "weighted_energy_mean")
        low_coupling = value(scale, "low_positive_radial", "positive_radial_coupling_mean")
        high_coupling = value(scale, "high_positive_radial", "positive_radial_coupling_mean")

        low_radial = low_coupling + low_pos + low_neg
        high_radial = high_coupling + high_pos + high_neg
        values["low_radial"].append(low_radial)
        values["high_radial"].append(high_radial)
        values["low_tangential"].append(low_tan)
        values["high_tangential"].append(high_tan)
        values["all_radial"].append(low_radial + high_radial)
        values["all_tangential"].append(low_tan + high_tan)
    return scales, values


def text_width(text, font, size):
    return pdfmetrics.stringWidth(text, font, size)


def draw_centered(c, text, x, y, size, font=FONT):
    c.setFont(font, size)
    c.drawString(x - text_width(text, font, size) / 2, y, text)


def draw_marker(c, marker, x, y, radius, color):
    c.setFillColor(color)
    c.setStrokeColor(color)
    if marker == "o":
        c.circle(x, y, radius, fill=1, stroke=0)
    elif marker == "d":
        path = c.beginPath()
        path.moveTo(x, y + radius * 1.35)
        path.lineTo(x + radius * 1.35, y)
        path.lineTo(x, y - radius * 1.35)
        path.lineTo(x - radius * 1.35, y)
        path.close()
        c.drawPath(path, fill=1, stroke=0)


def draw_label_box(c, text, x, y, color):
    size = 8.8
    pad_x = 3.2
    pad_y = 1.8
    width = text_width(text, FONT_BOLD, size) + 2 * pad_x
    height = size + 2 * pad_y
    c.setFillColor(colors.Color(1, 1, 1, alpha=0.88))
    c.rect(x - pad_x, y - pad_y, width, height, fill=1, stroke=0)
    c.setFillColor(color)
    c.setFont(FONT_BOLD, size)
    c.drawString(x, y, text)


def make_pdf(out_path):
    scales, values = load_values()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 252, 205
    left, right = 38, 8
    bottom, top = 34, 11
    plot_left, plot_right = left, width - right
    plot_bottom, plot_top = bottom, height - top
    x_min, x_max = 2.9, 15.7
    y_min, y_max = 0.0, 2.5

    def sx(x):
        return plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(y):
        return plot_bottom + (y - y_min) / (y_max - y_min) * (plot_top - plot_bottom)

    c = canvas.Canvas(str(out_path), pagesize=(width, height))
    c.setTitle("Directional spectral drift")

    # Grid and axes.
    c.setStrokeColor(colors.HexColor("#E0E0E0"))
    c.setLineWidth(0.32)
    for y in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        c.line(plot_left, sy(y), plot_right, sy(y))
    for x in scales:
        c.line(sx(x), plot_bottom, sx(x), plot_top)

    c.setStrokeColor(colors.HexColor("#222222"))
    c.setLineWidth(0.7)
    c.line(plot_left, plot_bottom, plot_right, plot_bottom)
    c.line(plot_left, plot_bottom, plot_left, plot_top)

    c.setFillColor(colors.HexColor("#222222"))
    c.setFont(FONT, 7.0)
    for y in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        c.drawRightString(plot_left - 4, sy(y) - 2.2, f"{y:g}")
    for x in scales:
        draw_centered(c, f"{x:g}", sx(x), plot_bottom - 10, 7.0, FONT)

    draw_centered(c, "Guidance scale w", (plot_left + plot_right) / 2, 7.5, 8.7, FONT)
    c.saveState()
    c.translate(8.0, (plot_bottom + plot_top) / 2)
    c.rotate(90)
    draw_centered(c, "Mean spectral drift term", 0, 0, 8.7, FONT)
    c.restoreState()

    red = colors.HexColor("#D62728")
    orange = colors.HexColor("#E67E00")
    blue = colors.HexColor("#0072B2")
    series = [
        ("all_radial", "ΔX_r", red, [], "o", 1.8),
        ("low_radial", "ΔX_{r,L}", orange, [5.0, 2.4], "o", 1.75),
        ("high_radial", "ΔX_{r,H}", blue, [], "o", 1.65),
        ("all_tangential", "ΔX_τ", red, [], "d", 1.55),
        ("low_tangential", "ΔX_{τ,L}", orange, [5.0, 2.4], "d", 1.5),
        ("high_tangential", "ΔX_{τ,H}", blue, [], "d", 1.45),
    ]

    for key, _, color, dash, marker, line_width in series:
        points = [(sx(x), sy(y)) for x, y in zip(scales, values[key])]
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(line_width)
        if dash:
            c.setDash(dash)
        else:
            c.setDash()
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            c.line(x1, y1, x2, y2)
        c.setDash()
        for x, y in points:
            draw_marker(c, marker, x, y, 1.85 if key.endswith("radial") else 1.65, color)

    draw_label_box(c, "ΔX_r (full radial)", sx(9.7), sy(1.82), red)
    draw_label_box(c, "ΔX_{r,L} (low radial)", sx(13.25), sy(1.55), orange)

    # Legend as a compact two-column block.
    legend_x, legend_y = plot_left + 5, plot_top - 15
    legend_col_w, legend_row_h = 58, 12.6
    box_w, box_h = 125, 42
    c.setFillColor(colors.Color(1, 1, 1, alpha=0.93))
    c.setStrokeColor(colors.HexColor("#B8B8B8"))
    c.setLineWidth(0.45)
    c.rect(legend_x - 3, legend_y - box_h + 5, box_w, box_h, fill=1, stroke=1)
    for idx, (key, label, color, dash, marker, line_width) in enumerate(series):
        col = idx // 3
        row = idx % 3
        x = legend_x + col * legend_col_w
        y = legend_y - row * legend_row_h
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(line_width)
        if dash:
            c.setDash(dash)
        else:
            c.setDash()
        c.line(x, y, x + 14, y)
        c.setDash()
        draw_marker(c, marker, x + 7, y, 1.55, color)
        c.setFont(FONT, 8.0)
        c.setFillColor(colors.HexColor("#111111"))
        c.drawString(x + 17, y - 2.5, label)

    c.showPage()
    c.save()


def main():
    pdf_path = OUT_DIR / "fig_directional_energy_reportlab.pdf"
    result_pdf = RESULT_DIR / "fig_directional_energy_reportlab.pdf"
    make_pdf(pdf_path)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_pdf.write_bytes(pdf_path.read_bytes())
    print(pdf_path)
    print(result_pdf)


if __name__ == "__main__":
    main()
