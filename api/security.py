"""Upload validation: only real images, only allowed types, limited size."""
from __future__ import annotations

from fastapi import HTTPException, UploadFile, status
from PIL import Image

from src.inference.image_utils import InvalidImageError, load_image_bytes

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def read_image_upload(file: UploadFile, max_bytes: int) -> Image.Image:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Please upload a JPG, PNG or WEBP photo.")
    data = await file.read(max_bytes + 1)  # read at most one byte over the limit
    if len(data) > max_bytes:
        raise HTTPException(413,  # "content too large"
                            f"Photo is too large. Maximum size is {max_bytes // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file.")
    try:
        return load_image_bytes(data)
    except InvalidImageError as err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This file is not a valid image.") from err
