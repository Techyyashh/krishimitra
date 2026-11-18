"""Safe image loading shared by the API and the Streamlit app."""
from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

# Refuse absurdly large images (protects the server's memory).
Image.MAX_IMAGE_PIXELS = 40_000_000


class InvalidImageError(ValueError):
    """Raised when bytes are not a readable image."""


def load_image_bytes(data: bytes) -> Image.Image:
    """Decode bytes → RGB PIL image, honouring phone EXIF rotation."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
        raise InvalidImageError("File is not a valid image") from err
    return ImageOps.exif_transpose(image).convert("RGB")
