import cv2
import numpy as np
from pathlib import Path


def read_image(path: str) -> np.ndarray:
    """
    Read an image from disk. Raises FileNotFoundError if path is invalid
    instead of silently returning None.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    """
    Save an image to disk. Raises IOError if write fails.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), img)
    if not success:
        raise IOError(f"Failed to write image to: {path}")