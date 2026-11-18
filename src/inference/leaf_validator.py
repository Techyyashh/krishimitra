"""A fast, simple "is this a leaf?" check.

Idea: a leaf photo is dominated by green / yellow / brown pixels with real colour
(not grey). We count the share of such pixels in HSV colour space.

Limits (be honest in your viva!): it is a heuristic. A green T-shirt passes, and a
very dark photo may fail. A future upgrade is a small binary "leaf vs not-leaf" CNN.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# PIL's HSV uses 0–255 for every channel. Hue 15–120 ≈ 21°–170° (yellow-brown → green → cyan-green).
HUE_MIN, HUE_MAX = 15, 120
SAT_MIN, VAL_MIN = 50, 40


def plant_pixel_ratio(image: Image.Image) -> float:
    """Fraction (0–1) of pixels that look like plant tissue."""
    hsv = np.asarray(image.convert("RGB").resize((128, 128)).convert("HSV"), dtype=np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= HUE_MIN) & (h <= HUE_MAX) & (s >= SAT_MIN) & (v >= VAL_MIN)
    return float(mask.mean())


def is_probably_leaf(image: Image.Image, min_ratio: float = 0.15) -> tuple[bool, float]:
    """Return (looks_like_leaf, score)."""
    score = plant_pixel_ratio(image)
    return score >= min_ratio, round(score, 3)
