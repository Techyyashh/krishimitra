"""Image augmentations. Training augmentations imitate real farm photos:
uneven sunlight, shadows, blur from shaky hands, and phone JPEG compression."""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics – required because our backbones were pretrained on ImageNet.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def train_transforms(size: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(size, size), scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.5),
            A.RandomShadow(p=0.2),
            A.OneOf([A.MotionBlur(), A.GaussianBlur(), A.GaussNoise()], p=0.3),
            A.ImageCompression(p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def eval_transforms(size: int) -> A.Compose:
    """Must match src/inference/predictor.py preprocessing exactly (resize + normalize)."""
    return A.Compose([A.Resize(size, size), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
