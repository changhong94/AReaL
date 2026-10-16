"""Generate research mechanism SVGs and optional PNG previews from reviewed specs.

Usage: python generate_diagrams.py --font-dir /path/to/truetype/fonts
SVG output uses only the standard library. PNG previews additionally use Pillow.
All geometry is schematic and must not be read as measured elapsed time or bytes.
"""

import argparse
import html
import json
import math
import textwrap
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    specs = json.loads((root / "diagram_specs.json").read_text(encoding="utf-8"))
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        Image = None
    fonts = []
    if args.font_dir:
        fonts.append(args.font_dir)
    fonts += [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]

    def font(size, bold=False):
        names = (
            ["arialbd.ttf", "DejaVuSans-Bold.ttf"]
            if bold
            else ["arial.ttf", "DejaVuSans.ttf"]
        )
        for directory in fonts:
            for name in names:
                if (directory / name).exists():
                    return ImageFont.truetype(str(directory / name), size)
        raise RuntimeError("PNG rendering requires a TrueType font; pass --font-dir")

    images = []
    for spec in specs:
        svg = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="880" viewBox="0 0 1440 880">',
            f"<title>{html.escape(spec['id'] + ': ' + spec['title'])}</title>",
            "<desc>Mechanism schematic. Geometry is not measured time or memory size. See linked research note for evidence and limitations.</desc>",
        ]
        im = Image.new("RGB", (1440, 880), "#f5f7fb") if Image else None
        draw = ImageDraw.Draw(im) if im else None

        def rect(x, y, w, h, fill, stroke=None, radius=14):
            svg.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke or fill}" stroke-width="2"/>'
            )
            if draw:
                draw.rounded_rectangle(
                    (x, y, x + w, y + h), radius, fill, stroke or fill, 2
                )

        def text(x, y, content, size=23, color="#172b4d", bold=False):
            svg.append(
                f'<text x="{x}" y="{y + size}" font-family="Arial, DejaVu Sans, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(content)}</text>'
            )
            if draw:
                draw.text((x, y), content, font=font(size, bold), fill=color)

        def wrapped(x, y, content, width, size=23, color="#172b4d", bold=False):
            # Conservative character width leaves room for wide glyphs in SVG renderers.
            lines = textwrap.wrap(content, width=width, break_long_words=False)
            for i, line in enumerate(lines):
                text(x, y + i * (size + 7), line, size, color, bold)
            return len(lines) * (size + 7)

        def line(x1, y1, x2, y2, color="#718096", arrow=False):
            svg.append(
                f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{color}" stroke-width="3" fill="none"/>'
            )
            if draw:
                draw.line((x1, y1, x2, y2), fill=color, width=3)
            if arrow:
                a = math.atan2(y2 - y1, x2 - x1)
                points = [
                    (x2, y2),
                    (x2 - 12 * math.cos(a - 0.45), y2 - 12 * math.sin(a - 0.45)),
                    (x2 - 12 * math.cos(a + 0.45), y2 - 12 * math.sin(a + 0.45)),
                ]
                svg.append(
                    '<polygon points="'
                    + " ".join(f"{x},{y}" for x, y in points)
                    + f'" fill="{color}"/>'
                )
                if draw:
                    draw.polygon(points, fill=color)

        def box(x, y, w, h, label, new=False):
            rect(
                x,
                y,
                w,
                h,
                "#dff3ed" if new else "#e9eef5",
                "#14846c" if new else "#a6b3c4",
            )
            wrapped(x + 17, y + 17, label, max(15, int((w - 34) / 12.8)), 23, bold=new)

        rect(0, 0, 1440, 880, "#f5f7fb", radius=0)
        rect(0, 0, 1440, 12, "#14846c", radius=0)
        text(44, 32, spec["id"], 27, "#14846c", True)
        text(156, 38, spec["status"], 18, "#6b4c16", True)
        wrapped(44, 83, spec["title"], 88, 31, bold=True)
        text(
            44,
            169,
            "MECHANISM SCHEMATIC  |  Not a measured timeline; box sizes are illustrative",
            18,
            "#596779",
        )
        text(44, 221, "PROBLEM / COMPARISON", 18, "#596779", True)
        for i, value in enumerate(spec["before"]):
            x = 44 + i * 345
            box(x, 255, 308, 119, value)
            if i < 3:
                line(x + 310, 315, x + 333, 315, arrow=True)
        text(
            44,
            415,
            "MECHANISM / CANDIDATE"
            if spec["id"] != "R020"
            else "CONTROLLED ALTERNATIVE (SLOWER HERE)",
            18,
            "#14846c",
            True,
        )
        if spec.get("parallel"):
            box(44, 489, 288, 111, spec["after"][0], True)
            box(412, 452, 580, 84, spec["after"][1], True)
            box(412, 554, 580, 84, spec["after"][2], True)
            box(1072, 489, 324, 122, spec["after"][3], True)
            line(333, 545, 370, 545)
            line(370, 495, 370, 596)
            line(370, 495, 400, 495, arrow=True)
            line(370, 596, 400, 596, arrow=True)
            line(994, 495, 1030, 495)
            line(994, 596, 1030, 596)
            line(1030, 495, 1030, 596)
            line(1030, 545, 1060, 545, arrow=True)
        else:
            for i, value in enumerate(spec["after"]):
                x = 44 + i * 345
                box(x, 453, 308, 125, value, True)
                if i < 3:
                    line(x + 310, 515, x + 333, 515, arrow=True)
            if spec.get("kind") == "memory":
                text(47, 596, "MEMORY STATE", 16, "#596779", True)
                cells = {
                    "R002": [
                        "TRAIN",
                        "TRAIN",
                        "WORK",
                        "WORK",
                        "KV",
                        "KV",
                        "FREE",
                        "SAFE",
                    ],
                    "R006": ["KV", "KV", "KEEP", "VA", "VA", "VA", "VA", "VA"],
                    "R015": ["LIVE", "LIVE", "LIVE", "VA", "VA", "VA", "VA", "VA"],
                    "R017": [
                        "LIVE",
                        "LIVE",
                        "LIVE",
                        "LIVE",
                        "LIVE",
                        "RAM",
                        "RAM",
                        "FREE",
                    ],
                }[spec["id"]]
                colors = {
                    "TRAIN": "#b8c6ee",
                    "WORK": "#d6dcf3",
                    "KV": "#8ed4c2",
                    "KEEP": "#c5e7dc",
                    "FREE": "#ffffff",
                    "SAFE": "#f5dfac",
                    "VA": "#ffffff",
                    "LIVE": "#8ed4c2",
                    "RAM": "#f5dfac",
                }
                for i, cell in enumerate(cells):
                    rect(220 + i * 95, 590, 86, 37, colors[cell], "#9aa8bc", 4)
                    text(235 + i * 95, 598, cell, 16)
                text(1010, 598, "VA = virtual only", 17, "#596779")
        rect(44, 665, 1352, 91, "#e3eaf8", radius=12)
        wrapped(62, 681, spec["focus"], 112, 23, bold=True)
        wrapped(44, 776, spec["evidence"], 132, 20, color="#596779")
        text(
            44,
            844,
            "Evidence and scope: linked research note  |  Reviewed 2026-10-15  |  No standalone novelty claim",
            16,
            "#596779",
        )
        svg.append("</svg>")
        (root / f"{spec['id']}.svg").write_text("\n".join(svg), encoding="utf-8")
        if im:
            im.save(root / f"{spec['id']}.png")
            images.append(im)
    if images:
        for page in range(3):
            sheet = Image.new("RGB", (1440, 1760), "white")
            for i, im in enumerate(images[page * 8 : page * 8 + 8]):
                sheet.paste(im.resize((720, 440)), ((i % 2) * 720, (i // 2) * 440))
            sheet.save(root / f"overview-{page + 1}.png")
    print(
        f"Generated {len(specs)} SVG diagrams" + (" and PNG previews" if images else "")
    )


if __name__ == "__main__":
    main()
