"""Generate the app icons (PNG) used by the PWA and the Play Store listing.

Run:  python scripts/make_icons.py
Replace with a designer-made logo later if you like – keep the same file names and sizes.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "web" / "icons"
GREEN, DARK, LEAF, VEIN = (31, 122, 58), (21, 92, 43), (126, 214, 120), (31, 122, 58)


def draw_icon(size: int, maskable: bool = False) -> Image.Image:
    scale = 4  # draw big, then shrink → smooth edges
    s = size * scale
    img = Image.new("RGB" if maskable else "RGBA", (s, s), GREEN if maskable else (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if not maskable:
        d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=GREEN)
    pad = s * (0.26 if maskable else 0.18)  # maskable icons need a bigger safe zone
    box = [pad, pad, s - pad, s - pad]
    leaf = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ld = ImageDraw.Draw(leaf)
    ld.ellipse([box[0] + s * 0.08, box[1], box[2] - s * 0.08, box[3]], fill=LEAF)
    cx = s / 2
    ld.line([(cx, box[1] + s * 0.04), (cx, box[3] + s * 0.02)], fill=VEIN, width=int(s * 0.025))
    for i in range(1, 4):
        y = box[1] + (box[3] - box[1]) * i / 4.5
        ld.line([(cx, y + s * 0.06), (cx - s * 0.12, y)], fill=VEIN, width=int(s * 0.018))
        ld.line([(cx, y + s * 0.06), (cx + s * 0.12, y)], fill=VEIN, width=int(s * 0.018))
    leaf = leaf.rotate(-30, resample=Image.BICUBIC, center=(cx, cx))
    img.paste(leaf, (0, 0), leaf)
    d.ellipse([s * 0.66, s * 0.66, s * 0.82, s * 0.82], fill=DARK)  # small "spot" = disease check
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_icon(192).save(OUT / "icon-192.png")
    draw_icon(512).save(OUT / "icon-512.png")
    draw_icon(512, maskable=True).save(OUT / "maskable-512.png")
    draw_icon(180, maskable=True).save(OUT / "apple-touch-icon.png")
    print(f"Icons written to {OUT}")


if __name__ == "__main__":
    main()
