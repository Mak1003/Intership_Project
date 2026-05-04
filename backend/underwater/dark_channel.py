import cv2
import numpy as np


def dark_channel(img: np.ndarray, size: int = 15) -> np.ndarray:
    """
    Compute the dark channel of an image.
    The dark channel is the minimum intensity across color channels
    after applying a minimum filter (erosion).

    Args:
        img: Input image (float32, any range)
        size: Patch size for the minimum filter

    Returns:
        Dark channel map (2D array)
    """
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark