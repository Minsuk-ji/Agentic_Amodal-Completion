from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

BBox = Tuple[int, int, int, int]


def bool_mask(mask: np.ndarray) -> np.ndarray:
    return mask if mask.dtype == bool else mask > 0


def mask_bbox(mask: np.ndarray) -> Optional[BBox]:
    m = bool_mask(mask)
    ys, xs = np.where(m)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resize_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return cv2.resize((bool_mask(mask).astype(np.uint8)) * 255, (width, height), interpolation=cv2.INTER_NEAREST)


def union_masks(masks: list[np.ndarray], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    out = np.zeros((height, width), dtype=bool)
    for mask in masks:
        if mask is None:
            continue
        arr = mask
        if arr.shape[:2] != (height, width):
            arr = resize_mask(arr, image_size)
        out |= bool_mask(arr)
    return out.astype(np.uint8) * 255


def dilate_mask(mask: np.ndarray, kernel_size: int, iterations: int = 1) -> np.ndarray:
    if int(iterations) <= 0:
        return (bool_mask(mask).astype(np.uint8)) * 255
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate((bool_mask(mask).astype(np.uint8)) * 255, kernel, iterations=int(iterations))


def clean_target_image(image: Image.Image, target_mask: np.ndarray, background=(255, 255, 255)) -> Image.Image:
    rgb = image.convert("RGB")
    mask_l = Image.fromarray(((bool_mask(target_mask)).astype(np.uint8)) * 255, mode="L").resize(rgb.size, Image.NEAREST)
    white = Image.new("RGB", rgb.size, background)
    return Image.composite(rgb, white, mask_l)


def empty_mask(image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return np.zeros((height, width), dtype=np.uint8)
