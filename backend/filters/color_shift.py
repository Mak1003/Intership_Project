import cv2
import numpy as np


def color_shift(img: np.ndarray, r_scale: float = 0.75, g_scale: float = 0.9, b_scale: float = 1.05) -> np.ndarray:
    """
    Simulate underwater color attenuation.
    Red attenuates fastest, blue penetrates deepest.

    Args:
        img: Input image (uint8, 0-255)
        r_scale: Red channel multiplier
        g_scale: Green channel multiplier
        b_scale: Blue channel multiplier

    Returns:
        Color-shifted image (uint8)
    """
    img_f = img.astype(np.float32) / 255.0
    B, G, R = cv2.split(img_f)

    R *= r_scale
    G *= g_scale
    B *= b_scale

    img_out = cv2.merge([B, G, R])
    img_out = np.clip(img_out, 0, 1)
    return (img_out * 255).astype(np.uint8)