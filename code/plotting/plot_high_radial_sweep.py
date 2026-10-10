import argparse
import csv
import html
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
except ModuleNotFoundError:
    plt = None
    FormatStrFormatter = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WS_SWEEP = ROOT / "code" / "resultes" / "w_sweep" / "merged_w_sweep_results.csv"
DEFAULT_HIGH_RADIAL = ROOT / "code" / "resultes" / "high_radial" / "benchmark_results.csv"
DEFAULT_OUT_DIR = ROOT / "RPTG_AAAI2027_Submission" / "Figures" / "HighRadialSweep"


METRICS = [
    ("clip_mean", "CLIP", True),
    ("fid", "FID", False),
    ("saturation_drift_mean", "Sat. Drift", False),
    ("contrast_mean", "Contrast", False),
]

TABLE_FIELDS = [
    ("gamma_h", r"$\gamma_H$"),
    ("clip_mean", "CLIP $\\uparrow$"),
    ("fid", "FID $\\downarrow$"),
    ("saturation_drift_mean", "Sat. Drift $\\downarrow$"),
    ("contrast_mean", "Contrast"),
    ("brightness_drift_mean", "Brightness Drift $\\downarrow$"),
    ("color_shift_mean", "Color Shift $\\downarrow$"),
]

HIGH_RADIAL_GAMMAS = {
    "rptg_hr_g0p1": 0.1,
    "rptg_hr_g0p5": 0.5,
    "rptg_hr_g0p75": 0.75,
    "rptg_hr_g1": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge and plot the RPTG-HR high-frequency radial sweep."
    )
    parser.add_argument("--w_sweep_csv", type=Path, default=DEFAULT_WS_SWEEP)
    parser.add_argument("--high_radial_csv", type=Path, default=DEFAULT_HIGH_RADIAL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--w", type=float, default=15.0)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def to_float(row, key):
    return float(row[key])


def row_for_gamma(row, gamma, source):
    out = dict(row)
    out["gamma_h"] = gamma
    out["source"] = source
    out["method_label"] = f"{gamma:g}"
    return out


def load_gamma_rows(w_sweep_csv, high_radial_csv, w):
    rows = {}
    for row in read_csv(w_sweep_csv):
        if "w" in row and abs(float(row["w"]) - w) > 1e-8:
            continue
        method = row["method"]
        display_method = row.get("display_method", method)
        if method == "rptg_apg" or display_method == "rptg":
            rows[0.0] = row_for_gamma(row, 0.0, f"w_sweep_w{w:g}")
        elif method == "rptg_hr_apg" or display_method == "rptg_hr":
            rows[0.25] = row_for_gamma(row, 0.25, f"w_sweep_w{w:g}")

    for row in read_csv(high_radial_csv):
        gamma = HIGH_RADIAL_GAMMAS.get(row["method"])
        if gamma is not None:
            rows[gamma] = row_for_gamma(row, gamma, "high_radial")

    expected = {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}
    missing = sorted(expected - set(rows))
    if missing:
        raise RuntimeError(f"Missing gamma_H rows: {missing}")
    return [rows[gamma] for gamma in sorted(rows)]


def write_merged_csv(rows, output_path):
    fieldnames = [
        "gamma_h",
        "source",
        "method",
        "clip_mean",
        "clip_std",
        "clip_count",
        "fid",
        "saturation_mean",
        "saturation_drift_mean",
        "contrast_mean",
        "brightness_mean",
        "brightness_drift_mean",
        "color_shift_mean",
        "missing_images",
        "extra_images",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def format_cell(row, key):
    if key == "gamma_h":
        return f"{float(row[key]):g}"
    value = float(row[key])
    if key == "fid":
        return f"{value:.2f}"
    return f"{value:.4f}"


def best_gamma(rows, key, higher_better):
    values = [(float(row[key]), float(row["gamma_h"])) for row in rows]
    return max(values)[1] if higher_better else min(values)[1]


def write_latex_table(rows, output_path):
    best = {
        "clip_mean": best_gamma(rows, "clip_mean", True),
        "fid": best_gamma(rows, "fid", False),
        "saturation_drift_mean": best_gamma(rows, "saturation_drift_mean", False),
        "brightness_drift_mean": best_gamma(rows, "brightness_drift_mean", False),
        "color_shift_mean": best_gamma(rows, "color_shift_mean", False),
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        " & ".join(label for _, label in TABLE_FIELDS) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        gamma = float(row["gamma_h"])
        cells = []
        for key, _ in TABLE_FIELDS:
            text = format_cell(row, key)
            if key in best and abs(gamma - best[key]) < 1e-8:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Ablation on the high-frequency radial retention weight "
            r"$\gamma_H$ under $w=15$ on 1,000 MS-COCO validation prompts.}"
        ),
        r"\label{tab:high_radial_sweep}",
        r"\end{table}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.7,
            "lines.markersize": 4.8,
        }
    )


def plot_rows(rows, output_dir):
    if plt is None:
        pdf_path = make_pdf_fallback(rows, output_dir)
        make_svg(rows, output_dir)
        return pdf_path, None

    setup_style()
    gammas = [float(row["gamma_h"]) for row in rows]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), constrained_layout=True)
    color = "#C51B7D"
    marker_color = "#E41A1C"

    for ax, (key, title, higher_better) in zip(axes, METRICS):
        values = [float(row[key]) for row in rows]
        ax.plot(gammas, values, color=color, marker="o", markerfacecolor=marker_color)
        ax.axvline(0.25, color="#666666", linewidth=0.8, linestyle="--", alpha=0.75)
        ax.set_title(f"{title} {'↑' if higher_better else '↓'}")
        ax.set_xlabel(r"$\gamma_H$")
        ax.set_xticks(gammas)
        ax.set_xticklabels(["0", "0.1", "0.25", "0.5", "0.75", "1"])
        ax.tick_params(axis="x", labelrotation=45, labelsize=7.2)
        if key == "clip_mean":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        elif key == "fid":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if key == "saturation_drift_mean":
            ax.annotate(
                "artifact rises",
                xy=(1.0, values[-1]),
                xytext=(0.48, max(values) - 0.01),
                arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#555555"},
                fontsize=6.8,
                color="#444444",
            )
        if key == "fid":
            best_index = min(range(len(values)), key=lambda i: values[i])
            ax.scatter(
                [gammas[best_index]],
                [values[best_index]],
                s=28,
                facecolor="white",
                edgecolor="#111111",
                linewidth=0.8,
                zorder=4,
            )

    pdf_path = output_dir / "fig_high_radial_sweep_1x4.pdf"
    png_path = output_dir / "fig_high_radial_sweep_1x4.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def scale(value, lo, hi, a, b):
    if abs(hi - lo) < 1e-12:
        return (a + b) / 2
    return a + (value - lo) / (hi - lo) * (b - a)


def nice_bounds(values, pad=0.08):
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        delta = abs(lo) * 0.1 if lo else 1.0
        return lo - delta, hi + delta
    delta = (hi - lo) * pad
    return lo - delta, hi + delta


def format_svg_tick(value, key):
    if key == "clip_mean":
        return f"{value:.3f}"
    if key == "fid":
        return f"{value:.1f}"
    return f"{value:.2f}"


def make_svg(rows, output_dir):
    gammas = [float(row["gamma_h"]) for row in rows]
    width = 1200
    height = 330
    margin_x = 42
    panel_gap = 30
    panel_w = (width - 2 * margin_x - 3 * panel_gap) / 4
    panel_h = 220
    top = 45
    bottom = top + panel_h
    color = "#C51B7D"
    marker = "#E41A1C"
    grid = "#D8D8D8"
    text = "#222222"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Times New Roman,DejaVu Serif,serif;} .small{font-size:18px;} .tick{font-size:15px;fill:#555;} .title{font-size:20px;font-weight:600;}</style>',
    ]
    for idx, (key, title, higher_better) in enumerate(METRICS):
        x0 = margin_x + idx * (panel_w + panel_gap)
        x1 = x0 + panel_w
        y0 = top
        y1 = bottom
        values = [float(row[key]) for row in rows]
        y_min, y_max = nice_bounds(values)
        if key == "saturation_drift_mean":
            y_min = min(y_min, 0.0)
        x_min, x_max = min(gammas), max(gammas)

        def sx(gamma):
            return scale(gamma, x_min, x_max, x0 + 35, x1 - 8)

        def sy(value):
            return scale(value, y_min, y_max, y1 - 30, y0 + 12)

        arrow = "↑" if higher_better else "↓"
        parts.append(f'<text class="title" x="{(x0+x1)/2:.1f}" y="{y0-16:.1f}" text-anchor="middle">{html.escape(title)} {arrow}</text>')
        for frac in (0, 0.25, 0.5, 0.75, 1):
            y = y1 - 30 - frac * (panel_h - 42)
            tick_value = y_min + frac * (y_max - y_min)
            parts.append(f'<line x1="{x0+35:.1f}" y1="{y:.1f}" x2="{x1-8:.1f}" y2="{y:.1f}" stroke="{grid}" stroke-width="1"/>')
            parts.append(f'<text class="tick" x="{x0+29:.1f}" y="{y+5:.1f}" text-anchor="end">{format_svg_tick(tick_value, key)}</text>')
        for gamma in gammas:
            x = sx(gamma)
            parts.append(f'<line x1="{x:.1f}" y1="{y0+12:.1f}" x2="{x:.1f}" y2="{y1-30:.1f}" stroke="{grid}" stroke-width="1" opacity="0.45"/>')
            label = f"{gamma:g}"
            parts.append(f'<text class="tick" x="{x:.1f}" y="{y1-8:.1f}" text-anchor="middle" transform="rotate(-35 {x:.1f},{y1-8:.1f})">{label}</text>')
        x_default = sx(0.25)
        parts.append(f'<line x1="{x_default:.1f}" y1="{y0+10:.1f}" x2="{x_default:.1f}" y2="{y1-30:.1f}" stroke="#666" stroke-width="2" stroke-dasharray="6 5" opacity="0.75"/>')
        parts.append(f'<line x1="{x0+35:.1f}" y1="{y1-30:.1f}" x2="{x1-8:.1f}" y2="{y1-30:.1f}" stroke="{text}" stroke-width="1.2"/>')
        parts.append(f'<line x1="{x0+35:.1f}" y1="{y0+12:.1f}" x2="{x0+35:.1f}" y2="{y1-30:.1f}" stroke="{text}" stroke-width="1.2"/>')
        points = [(sx(gamma), sy(value)) for gamma, value in zip(gammas, values)]
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{marker}" stroke="white" stroke-width="1.5"/>')
        parts.append(f'<text class="small" x="{(x0+x1)/2:.1f}" y="{height-12}" text-anchor="middle">γ<tspan baseline-shift="sub" font-size="13">H</tspan></text>')

    parts.append("</svg>")
    svg_path = output_dir / "fig_high_radial_sweep_1x4.svg"
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return svg_path


def pdf_escape(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def rgb_to_pdf(hex_color):
    hex_color = hex_color.lstrip("#")
    values = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return " ".join(f"{value:.4f}" for value in values)


def make_pdf_fallback(rows, output_dir):
    gammas = [float(row["gamma_h"]) for row in rows]
    width = 560.0
    height = 154.0
    margin_x = 18.0
    panel_gap = 13.0
    panel_w = (width - 2 * margin_x - 3 * panel_gap) / 4
    panel_h = 104.0
    top = 35.0
    bottom = top + panel_h
    line_color = "#C51B7D"
    marker_color = "#E41A1C"
    grid_color = "#D8D8D8"
    text_color = "#222222"
    commands = []

    def cmd(line):
        commands.append(line)

    def set_stroke(hex_color):
        cmd(f"{rgb_to_pdf(hex_color)} RG")

    def set_fill(hex_color):
        cmd(f"{rgb_to_pdf(hex_color)} rg")

    def draw_line(x1, y1, x2, y2, width_pt=0.5):
        cmd(f"{width_pt:.3f} w")
        cmd(f"{x1:.3f} {height - y1:.3f} m {x2:.3f} {height - y2:.3f} l S")

    def draw_circle(x, y, radius, fill_hex, stroke_hex="#FFFFFF", stroke_width=0.55):
        set_fill(fill_hex)
        set_stroke(stroke_hex)
        cmd(f"{stroke_width:.3f} w")
        cy = height - y
        k = 0.5522847498
        cmd(
            f"{x + radius:.3f} {cy:.3f} m "
            f"{x + radius:.3f} {cy + k * radius:.3f} "
            f"{x + k * radius:.3f} {cy + radius:.3f} "
            f"{x:.3f} {cy + radius:.3f} c "
            f"{x - k * radius:.3f} {cy + radius:.3f} "
            f"{x - radius:.3f} {cy + k * radius:.3f} "
            f"{x - radius:.3f} {cy:.3f} c "
            f"{x - radius:.3f} {cy - k * radius:.3f} "
            f"{x - k * radius:.3f} {cy - radius:.3f} "
            f"{x:.3f} {cy - radius:.3f} c "
            f"{x + k * radius:.3f} {cy - radius:.3f} "
            f"{x + radius:.3f} {cy - k * radius:.3f} "
            f"{x + radius:.3f} {cy:.3f} c B"
        )

    def draw_text(x, y, value, size=6.0, align="left", font="F1", fill=text_color):
        value = pdf_escape(value)
        set_fill(fill)
        approx_width = len(value) * size * 0.47
        if align == "center":
            x -= approx_width / 2
        elif align == "right":
            x -= approx_width
        cmd(f"BT /{font} {size:.2f} Tf {x:.3f} {height - y:.3f} Td ({value}) Tj ET")

    set_fill("#FFFFFF")
    cmd(f"0 0 {width:.3f} {height:.3f} re f")

    for idx, (key, title, higher_better) in enumerate(METRICS):
        x0 = margin_x + idx * (panel_w + panel_gap)
        x1 = x0 + panel_w
        y0 = top
        y1 = bottom
        values = [float(row[key]) for row in rows]
        y_min, y_max = nice_bounds(values)
        if key == "saturation_drift_mean":
            y_min = min(y_min, 0.0)
        x_min, x_max = min(gammas), max(gammas)

        def sx(gamma):
            return scale(gamma, x_min, x_max, x0 + 20, x1 - 3)

        def sy(value):
            return scale(value, y_min, y_max, y1 - 17, y0 + 6)

        arrow = "up" if higher_better else "down"
        draw_text((x0 + x1) / 2, y0 - 12, f"{title} {arrow}", size=7.0, align="center", font="F2")

        set_stroke(grid_color)
        for frac in (0, 0.25, 0.5, 0.75, 1):
            y = y1 - 17 - frac * (panel_h - 23)
            tick_value = y_min + frac * (y_max - y_min)
            draw_line(x0 + 20, y, x1 - 3, y, 0.28)
            draw_text(
                x0 + 17,
                y + 2.2,
                format_svg_tick(tick_value, key),
                size=4.4,
                align="right",
                fill="#555555",
            )

        for gamma in gammas:
            x = sx(gamma)
            set_stroke(grid_color)
            draw_line(x, y0 + 6, x, y1 - 17, 0.22)
            draw_text(x, y1 - 5, f"{gamma:g}", size=4.8, align="center", fill="#555555")

        x_default = sx(0.25)
        set_stroke("#666666")
        cmd("[2 2] 0 d")
        draw_line(x_default, y0 + 5, x_default, y1 - 17, 0.5)
        cmd("[] 0 d")

        set_stroke(text_color)
        draw_line(x0 + 20, y1 - 17, x1 - 3, y1 - 17, 0.45)
        draw_line(x0 + 20, y0 + 6, x0 + 20, y1 - 17, 0.45)

        points = [(sx(gamma), sy(value)) for gamma, value in zip(gammas, values)]
        set_stroke(line_color)
        cmd("1.2 w")
        if points:
            first_x, first_y = points[0]
            pieces = [f"{first_x:.3f} {height - first_y:.3f} m"]
            for x, y in points[1:]:
                pieces.append(f"{x:.3f} {height - y:.3f} l")
            cmd(" ".join(pieces) + " S")
        for x, y in points:
            draw_circle(x, y, 2.1, marker_color, "#FFFFFF", 0.45)
        draw_text((x0 + x1) / 2, height - 4, "gamma_H", size=5.6, align="center")

    content = "\n".join(commands).encode("latin-1", errors="replace")
    objects = []

    def add_object(payload):
        objects.append(payload)

    add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_object(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add_object(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.3f} {height:.3f}] "
        f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>".encode("ascii")
    )
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>")
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>")
    add_object(
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"\nendstream"
    )

    pdf_path = output_dir / "fig_high_radial_sweep_1x4.pdf"
    with pdf_path.open("wb") as file:
        file.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, payload in enumerate(objects, start=1):
            offsets.append(file.tell())
            file.write(f"{index} 0 obj\n".encode("ascii"))
            file.write(payload)
            file.write(b"\nendobj\n")
        xref_offset = file.tell()
        file.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        file.write(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            file.write(f"{offset:010d} 00000 n \n".encode("ascii"))
        file.write(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
        )
    return pdf_path


def write_latex_figure(output_path, graphics_name="fig_high_radial_sweep_1x4.pdf"):
    text = rf"""\begin{{figure*}}[t]
    \centering
    \includegraphics[width=\textwidth]{{Figures/HighRadialSweep/{graphics_name}}}
    \caption{{Effect of the high-frequency radial retention weight $\gamma_H$ under strong guidance ($w=15$). Moderate high-frequency radial restoration improves alignment and distributional quality, while excessive restoration increases saturation and contrast, indicating that high-frequency radial information is useful but should be controlled. The dashed vertical line marks the default $\gamma_H=0.25$.}}
    \label{{fig:high_radial_sweep}}
\end{{figure*}}
"""
    output_path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_gamma_rows(args.w_sweep_csv, args.high_radial_csv, args.w)

    merged_csv = args.output_dir / "high_radial_sweep_merged.csv"
    table_tex = args.output_dir / "tab_high_radial_sweep.tex"
    figure_tex = args.output_dir / "fig_high_radial_sweep_1x4.tex"
    write_merged_csv(rows, merged_csv)
    write_latex_table(rows, table_tex)
    figure_path, png_path = plot_rows(rows, args.output_dir)
    graphics_name = (
        "fig_high_radial_sweep_1x4.pdf"
        if figure_path.suffix == ".pdf"
        else "fig_high_radial_sweep_1x4.svg"
    )
    write_latex_figure(figure_tex, graphics_name=graphics_name)

    print(f"Saved merged CSV: {merged_csv}")
    print(f"Saved LaTeX table: {table_tex}")
    print(f"Saved figure: {figure_path}")
    if png_path is not None:
        print(f"Saved figure PNG: {png_path}")
    print(f"Saved LaTeX figure wrapper: {figure_tex}")


if __name__ == "__main__":
    main()
