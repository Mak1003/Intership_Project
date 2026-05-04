import cv2
import numpy as np


def depth_blur(img: np.ndarray, max_strength: float = 0.5) -> np.ndarray:
    """
    Apply depth-based blur — simulates loss of focus with depth.
    Fixed: original mutated input in-place causing row-by-row contamination.

    Args:
        img: Input image (uint8, 0-255)
        max_strength: Maximum blend factor with blurred image at bottom

    Returns:
        Depth-blurred image (uint8)
    """
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    blur = cv2.GaussianBlur(img_f, (21, 21), 0)

    # Work on a copy — don't modify img row-by-row
    result = img_f.copy()
    mask = np.linspace(0, max_strength, h)

    for i in range(h):
        alpha = mask[i]
        result[i, :, :] = img_f[i, :, :] * (1 - alpha) + blur[i, :, :] * alpha

    return np.clip(result, 0, 255).astype(np.uint8)