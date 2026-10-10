import argparse
import html
import os

from PIL import Image, ImageDraw, ImageFont


MAIN_COLUMNS = [
    ("cfg", "CFG"),
    ("low_full_radial", "Low Radial"),
    ("low_tangential", "Low Tangential"),
    ("high_full_radial", "High Radial"),
    ("high_tangential", "High Tangential"),
    ("rptg_hr_apg", "RPTG-HR"),
]

PANEL_COLUMNS = [
    [
        ("cfg", "CFG", None),
        ("low_full_radial", "Low: ΔXr", "danger"),
        ("low_tangential", "Low: ΔXτ", None),
    ],
    [
        ("rptg_hr_apg", "RPTG-HR", "ours"),
        ("high_full_radial", "High: ΔXr", None),
        ("high_tangential", "High: ΔXτ", None),
    ],
]

PANEL_2X4_COLUMNS = [
    [
        ("cfg", "CFG", None),
        ("full_radial", "Delta Xr", None),
        ("full_tangential", "Delta Xtau (RPTG)", None),
        ("rptg_hr_apg", "RPTG-HR", None),
    ],
    [
        ("low_full_radial", "Delta Xr,L", None),
        ("high_full_radial", "Delta Xr,H", None),
        ("low_tangential", "Delta Xtau,L", None),
        ("high_tangential", "Delta Xtau,H", None),
    ],
]


SIGNED_COLUMNS = [
    ("low_full_radial", "Low Radial"),
    ("low_pos_radial", "Low +Radial"),
    ("low_neg_radial", "Low -Radial"),
    ("high_full_radial", "High Radial"),
    ("high_pos_radial", "High +Radial"),
    ("high_neg_radial", "High -Radial"),
]

LATEX_GROUPS = [
    ("cfg", "fig2_group_cfg", r"\textbf{CFG}"),
    ("full_radial", "fig2_group_full_radial", r"$\Delta X_r$"),
    ("full_tangential", "fig2_group_full_tangential", r"$\Delta X_\tau$ (RPTG)"),
    ("rptg_hr_apg", "fig2_group_rptg_hr", r"\textbf{RPTG-HR}"),
    ("low_full_radial", "fig2_group_low_radial", r"$\Delta X_{r,L}$"),
    ("high_full_radial", "fig2_group_high_radial", r"$\Delta X_{r,H}$"),
    ("low_tangential", "fig2_group_low_tangential", r"$\Delta X_{\tau,L}$"),
    ("high_tangential", "fig2_group_high_tangential", r"$\Delta X_{\tau,H}$"),
]

APPENDIX_DECOMPOSITION_GROUPS = [
    ("low_full_radial", "appendix_group_low_radial", r"$\Delta X_{r,L}$"),
    ("low_pos_radial", "appendix_group_low_pos_radial", r"$\Delta X^{+}_{r,L}$"),
    ("low_neg_radial", "appendix_group_low_neg_radial", r"$\Delta X^{-}_{r,L}$"),
    ("low_tangential", "appendix_group_low_tangential", r"$\Delta X_{\tau,L}$"),
    ("high_full_radial", "appendix_group_high_radial", r"$\Delta X_{r,H}$"),
    ("high_pos_radial", "appendix_group_high_pos_radial", r"$\Delta X^{+}_{r,H}$"),
    ("high_neg_radial", "appendix_group_high_neg_radial", r"$\Delta X^{-}_{r,H}$"),
    ("high_tangential", "appendix_group_high_tangential", r"$\Delta X_{\tau,H}$"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make paper-ready RPTG directional ablation layouts."
    )
    parser.add_argument(
        "--input_dir",
        default="/Users/zhuaotian/Desktop/AAAIFig/Fig2/rptg_directional_figure_w15_new_prompts",
    )
    parser.add_argument(
        "--output_dir",
        default="/Users/zhuaotian/Documents/Codex/codex_FSG/codex_FSG/RPTG_AAAI2027_Submission/Figures",
    )
    parser.add_argument("--thumb_size", type=int, default=190)
    parser.add_argument("--inner_gap", type=int, default=6)
    parser.add_argument("--group_gap", type=int, default=14)
    parser.add_argument("--top_margin", type=int, default=40)
    parser.add_argument("--label_height", type=int, default=28)
    parser.add_argument("--bottom_margin", type=int, default=8)
    parser.add_argument("--draw_prompt_labels", action="store_true")
    return parser.parse_args()


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def center_text(draw, xywh, text, font, fill=(20, 20, 20)):
    x, y, w, h = xywh
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 1), text, font=font, fill=fill)


def make_method_group(input_dir, method_name, thumb_size, inner_gap):
    group_size = 2 * thumb_size + inner_gap
    group = Image.new("RGB", (group_size, group_size), "white")
    for prompt_index in range(4):
        path = os.path.join(input_dir, method_name, f"prompt_{prompt_index:02d}.jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB").resize(
            (thumb_size, thumb_size), Image.Resampling.LANCZOS
        )
        row = prompt_index // 2
        col = prompt_index % 2
        x = col * (thumb_size + inner_gap)
        y = row * (thumb_size + inner_gap)
        group.paste(image, (x, y))
    return group


def make_layout(input_dir, output_dir, columns, stem, args):
    os.makedirs(output_dir, exist_ok=True)
    title_font = load_font(20, bold=True)
    note_font = load_font(14, bold=False)
    group_size = 2 * args.thumb_size + args.inner_gap
    width = len(columns) * group_size + (len(columns) - 1) * args.group_gap
    height = (
        args.top_margin
        + args.label_height
        + group_size
        + args.bottom_margin
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for col_index, (method_name, label) in enumerate(columns):
        x = col_index * (group_size + args.group_gap)
        center_text(
            draw,
            (x, args.top_margin, group_size, args.label_height),
            label,
            title_font,
        )
        group = make_method_group(
            input_dir=input_dir,
            method_name=method_name,
            thumb_size=args.thumb_size,
            inner_gap=args.inner_gap,
        )
        y = args.top_margin + args.label_height
        canvas.paste(group, (x, y))
        if method_name == "rptg_hr_apg":
            draw.rectangle(
                [x - 2, y - 2, x + group_size + 1, y + group_size + 1],
                outline=(35, 105, 190),
                width=3,
            )

    if args.draw_prompt_labels:
        prompt_labels = ["Animal", "Landscape", "Text", "Vehicle"]
        for prompt_index, label in enumerate(prompt_labels):
            row = prompt_index // 2
            col = prompt_index % 2
            x = col * (args.thumb_size + args.inner_gap) + 6
            y = args.top_margin + args.label_height + row * (args.thumb_size + args.inner_gap) + 6
            draw.rounded_rectangle(
                [x, y, x + 74, y + 22],
                radius=4,
                fill=(255, 255, 255),
                outline=(220, 220, 220),
            )
            draw.text((x + 6, y + 3), label, font=note_font, fill=(40, 40, 40))

    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    svg_path = os.path.join(output_dir, f"{stem}.svg")
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300.0)
    write_linked_svg(
        input_dir=input_dir,
        output_path=svg_path,
        columns=columns,
        width=width,
        height=height,
        group_size=group_size,
        args=args,
    )
    return png_path, pdf_path, svg_path


def make_panel_layout(input_dir, output_dir, stem, args):
    os.makedirs(output_dir, exist_ok=True)
    title_font = load_font(22, bold=True)
    group_size = 2 * args.thumb_size + args.inner_gap
    panel_gap_x = 24
    panel_gap_y = 34
    label_height = 30
    width = 3 * group_size + 2 * panel_gap_x
    height = 2 * (label_height + group_size) + panel_gap_y + 6
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for row_index, row in enumerate(PANEL_COLUMNS):
        for col_index, (method_name, label, emphasis) in enumerate(row):
            x = col_index * (group_size + panel_gap_x)
            y = row_index * (label_height + group_size + panel_gap_y)
            center_text(draw, (x, y, group_size, label_height), label, title_font, fill=(20, 20, 20))
            group_y = y + label_height
            group = make_method_group(
                input_dir=input_dir,
                method_name=method_name,
                thumb_size=args.thumb_size,
                inner_gap=args.inner_gap,
            )
            canvas.paste(group, (x, group_y))

    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    svg_path = os.path.join(output_dir, f"{stem}.svg")
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300.0)
    write_panel_svg(
        input_dir=input_dir,
        output_path=svg_path,
        width=width,
        height=height,
        group_size=group_size,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
        label_height=label_height,
        args=args,
    )
    return png_path, pdf_path, svg_path


def make_panel_2x4_pdf(input_dir, output_dir, stem, args):
    os.makedirs(output_dir, exist_ok=True)
    title_font = load_font(20, bold=True)
    group_size = 2 * args.thumb_size + args.inner_gap
    panel_gap_x = 16
    panel_gap_y = 30
    label_height = 28
    width = 4 * group_size + 3 * panel_gap_x
    height = 2 * (label_height + group_size) + panel_gap_y + 4
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for row_index, row in enumerate(PANEL_2X4_COLUMNS):
        for col_index, (method_name, label, emphasis) in enumerate(row):
            x = col_index * (group_size + panel_gap_x)
            y = row_index * (label_height + group_size + panel_gap_y)
            color = (180, 35, 35) if emphasis == "danger" else (20, 20, 20)
            center_text(draw, (x, y, group_size, label_height), label, title_font, fill=color)
            group_y = y + label_height
            group = make_method_group(
                input_dir=input_dir,
                method_name=method_name,
                thumb_size=args.thumb_size,
                inner_gap=args.inner_gap,
            )
            canvas.paste(group, (x, group_y))
            if emphasis == "danger":
                draw.rectangle(
                    [x - 3, group_y - 3, x + group_size + 2, group_y + group_size + 2],
                    outline=(200, 45, 45),
                    width=4,
                )
            elif emphasis == "ours":
                draw.rectangle(
                    [x - 3, group_y - 3, x + group_size + 2, group_y + group_size + 2],
                    outline=(35, 105, 190),
                    width=4,
                )

    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300.0)
    return png_path, pdf_path


def make_latex_group_assets(input_dir, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    group_size = 2 * args.thumb_size + args.inner_gap
    available = []
    for method_name, stem, latex_label in LATEX_GROUPS:
        method_dir = os.path.join(input_dir, method_name)
        if not os.path.isdir(method_dir):
            print(f"Skipping missing group asset: {method_name}")
            continue
        group = make_method_group(
            input_dir=input_dir,
            method_name=method_name,
            thumb_size=args.thumb_size,
            inner_gap=args.inner_gap,
        )
        png_path = os.path.join(output_dir, f"{stem}.png")
        pdf_path = os.path.join(output_dir, f"{stem}.pdf")
        group.save(png_path)
        group.save(pdf_path, "PDF", resolution=300.0)
        available.append((method_name, stem, latex_label))

    tex_path = os.path.join(output_dir, "fig_directional_ablation_2x4.tex")
    write_latex_figure_snippet(tex_path, available)
    return tex_path


def make_appendix_decomposition_assets(input_dir, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    available = []
    for method_name, stem, latex_label in APPENDIX_DECOMPOSITION_GROUPS:
        method_dir = os.path.join(input_dir, method_name)
        if not os.path.isdir(method_dir):
            print(f"Skipping missing appendix group asset: {method_name}")
            continue
        group = make_method_group(
            input_dir=input_dir,
            method_name=method_name,
            thumb_size=args.thumb_size,
            inner_gap=args.inner_gap,
        )
        png_path = os.path.join(output_dir, f"{stem}.png")
        pdf_path = os.path.join(output_dir, f"{stem}.pdf")
        group.save(png_path)
        group.save(pdf_path, "PDF", resolution=300.0)
        available.append((method_name, stem, latex_label))

    tex_path = os.path.join(output_dir, "fig_frequency_decomposition_appendix.tex")
    write_appendix_decomposition_snippet(tex_path, available)
    return tex_path


def write_latex_figure_snippet(tex_path, available):
    stems = {method_name: (stem, latex_label) for method_name, stem, latex_label in available}
    required = [
        "cfg",
        "full_radial",
        "full_tangential",
        "rptg_hr_apg",
        "low_full_radial",
        "high_full_radial",
        "low_tangential",
        "high_tangential",
    ]
    missing = [name for name in required if name not in stems]
    if missing:
        note = "% Missing generated groups: " + ", ".join(missing) + "\n"
    else:
        note = ""

    def cell(method_name):
        stem, label = stems.get(method_name, (f"fig2_group_{method_name}", method_name))
        return label, f"\\includegraphics[width=0.235\\textwidth]{{Figures/Fig2/{stem}.pdf}}"

    labels_top = [cell(name)[0] for name in required[:4]]
    images_top = [cell(name)[1] for name in required[:4]]
    labels_bottom = [cell(name)[0] for name in required[4:]]
    images_bottom = [cell(name)[1] for name in required[4:]]
    content = rf"""{note}\begin{{figure*}}[t]
    \centering
    \setlength{{\tabcolsep}}{{2pt}}
    \begin{{tabular}}{{@{{}}cccc@{{}}}}
        {labels_top[0]} & {labels_top[1]} & {labels_top[2]} & {labels_top[3]} \\
        {images_top[0]} & {images_top[1]} & {images_top[2]} & {images_top[3]} \\
        [2pt]
        {labels_bottom[0]} & {labels_bottom[1]} & {labels_bottom[2]} & {labels_bottom[3]} \\
        {images_bottom[0]} & {images_bottom[1]} & {images_bottom[2]} & {images_bottom[3]} \\
    \end{{tabular}}
    \caption{{Directional frequency ablation at high guidance scale. Each cell contains four prompts generated by retaining one component of the CFG residual during sampling. Low-frequency radial updates show the strongest brightness and saturation drift, while tangential updates preserve more stable appearance. RPTG-HR combines full-spectrum tangential guidance with controlled high-frequency radial restoration.}}
    \label{{fig:directional_ablation}}
\end{{figure*}}
"""
    with open(tex_path, "w", encoding="utf-8") as file:
        file.write(content)


def write_appendix_decomposition_snippet(tex_path, available):
    stems = {method_name: (stem, latex_label) for method_name, stem, latex_label in available}
    required = [
        "low_full_radial",
        "low_pos_radial",
        "low_neg_radial",
        "low_tangential",
        "high_full_radial",
        "high_pos_radial",
        "high_neg_radial",
        "high_tangential",
    ]
    missing = [name for name in required if name not in stems]
    if missing:
        note = "% Missing generated groups: " + ", ".join(missing) + "\n"
    else:
        note = ""

    def cell(method_name):
        stem, label = stems.get(method_name, (f"appendix_group_{method_name}", method_name))
        return label, f"\\includegraphics[width=0.235\\textwidth]{{Figures/Fig2/{stem}.pdf}}"

    labels_top = [cell(name)[0] for name in required[:4]]
    images_top = [cell(name)[1] for name in required[:4]]
    labels_bottom = [cell(name)[0] for name in required[4:]]
    images_bottom = [cell(name)[1] for name in required[4:]]
    content = rf"""{note}\begin{{figure*}}[t]
    \centering
    \setlength{{\tabcolsep}}{{2pt}}
    \begin{{tabular}}{{@{{}}cccc@{{}}}}
        {labels_top[0]} & {labels_top[1]} & {labels_top[2]} & {labels_top[3]} \\
        {images_top[0]} & {images_top[1]} & {images_top[2]} & {images_top[3]} \\
        [2pt]
        {labels_bottom[0]} & {labels_bottom[1]} & {labels_bottom[2]} & {labels_bottom[3]} \\
        {images_bottom[0]} & {images_bottom[1]} & {images_bottom[2]} & {images_bottom[3]} \\
    \end{{tabular}}
    \caption{{Frequency decomposition of directional residuals. The first row shows the low-frequency residual split into full radial, positive radial, negative radial, and tangential components, while the second row shows the corresponding high-frequency components. This comparison supplements Fig.~\ref{{fig:directional_ablation}} by separating the signed radial updates and shows that the low-frequency radial component, especially its positive direction, is the main source of global brightness and saturation drift.}}
    \label{{fig:appendix_frequency_decomposition}}
\end{{figure*}}
"""
    with open(tex_path, "w", encoding="utf-8") as file:
        file.write(content)


def write_panel_svg(
    input_dir,
    output_path,
    width,
    height,
    group_size,
    panel_gap_x,
    panel_gap_y,
    label_height,
    args,
):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#202020;} .title{font-size:22px;font-weight:700;} .danger{fill:#b42323;}</style>',
    ]
    for row_index, row in enumerate(PANEL_COLUMNS):
        for col_index, (method_name, label, emphasis) in enumerate(row):
            x = col_index * (group_size + panel_gap_x)
            y = row_index * (label_height + group_size + panel_gap_y)
            cls = "title danger" if emphasis == "danger" else "title"
            parts.append(
                f'<text class="{cls}" x="{x + group_size / 2}" y="{y + 23}" text-anchor="middle">{html.escape(label)}</text>'
            )
            group_y = y + label_height
            for prompt_index in range(4):
                row_i = prompt_index // 2
                col_i = prompt_index % 2
                image_path = os.path.abspath(
                    os.path.join(input_dir, method_name, f"prompt_{prompt_index:02d}.jpg")
                )
                ix = x + col_i * (args.thumb_size + args.inner_gap)
                iy = group_y + row_i * (args.thumb_size + args.inner_gap)
                parts.append(
                    f'<image href="{html.escape(image_path)}" x="{ix}" y="{iy}" width="{args.thumb_size}" height="{args.thumb_size}" preserveAspectRatio="xMidYMid slice"/>'
                )
            if emphasis == "danger":
                parts.append(
                    f'<rect x="{x - 3}" y="{group_y - 3}" width="{group_size + 6}" height="{group_size + 6}" fill="none" stroke="#c82d2d" stroke-width="4"/>'
                )
            elif emphasis == "ours":
                parts.append(
                    f'<rect x="{x - 3}" y="{group_y - 3}" width="{group_size + 6}" height="{group_size + 6}" fill="none" stroke="#2369be" stroke-width="4"/>'
                )
    parts.append("</svg>\n")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(parts))


def write_linked_svg(input_dir, output_path, columns, width, height, group_size, args):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#202020;} .title{font-size:20px;font-weight:700;}</style>',
    ]
    for col_index, (method_name, label) in enumerate(columns):
        x = col_index * (group_size + args.group_gap)
        label_safe = html.escape(label)
        parts.append(
            f'<text class="title" x="{x + group_size / 2}" y="{args.top_margin + 21}" text-anchor="middle">{label_safe}</text>'
        )
        y0 = args.top_margin + args.label_height
        for prompt_index in range(4):
            row = prompt_index // 2
            col = prompt_index % 2
            image_path = os.path.abspath(
                os.path.join(input_dir, method_name, f"prompt_{prompt_index:02d}.jpg")
            )
            ix = x + col * (args.thumb_size + args.inner_gap)
            iy = y0 + row * (args.thumb_size + args.inner_gap)
            parts.append(
                f'<image href="{html.escape(image_path)}" x="{ix}" y="{iy}" width="{args.thumb_size}" height="{args.thumb_size}" preserveAspectRatio="xMidYMid slice"/>'
            )
        if method_name == "rptg_hr_apg":
            parts.append(
                f'<rect x="{x - 2}" y="{y0 - 2}" width="{group_size + 4}" height="{group_size + 4}" fill="none" stroke="#2369be" stroke-width="3"/>'
            )
    parts.append("</svg>\n")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(parts))


def main():
    args = parse_args()
    panel_2x4_png, panel_2x4_pdf = make_panel_2x4_pdf(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stem="rptg_directional_ablation_2x4_single",
        args=args,
    )
    latex_tex = make_latex_group_assets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        args=args,
    )
    appendix_tex = make_appendix_decomposition_assets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        args=args,
    )
    panel_png, panel_pdf, panel_svg = make_panel_layout(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stem="rptg_directional_ablation_2x3_main",
        args=args,
    )
    main_png, main_pdf, main_svg = make_layout(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        columns=MAIN_COLUMNS,
        stem="rptg_directional_ablation_main",
        args=args,
    )
    signed_png, signed_pdf, signed_svg = make_layout(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        columns=SIGNED_COLUMNS,
        stem="rptg_directional_ablation_signed_appendix",
        args=args,
    )
    print(f"Saved 2x4 single PNG: {panel_2x4_png}")
    print(f"Saved 2x4 single PDF: {panel_2x4_pdf}")
    print(f"Saved LaTeX figure snippet: {latex_tex}")
    print(f"Saved appendix decomposition snippet: {appendix_tex}")
    print(f"Saved 2x3 main PNG: {panel_png}")
    print(f"Saved 2x3 main PDF: {panel_pdf}")
    print(f"Saved 2x3 main SVG: {panel_svg}")
    print(f"Saved main PNG: {main_png}")
    print(f"Saved main PDF: {main_pdf}")
    print(f"Saved main SVG: {main_svg}")
    print(f"Saved appendix PNG: {signed_png}")
    print(f"Saved appendix PDF: {signed_pdf}")
    print(f"Saved appendix SVG: {signed_svg}")


if __name__ == "__main__":
    main()
