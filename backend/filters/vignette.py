import cv2
import numpy as np


def apply_vignette(img: np.ndarray, sigma_ratio: float = 2.0) -> np.ndarray:
    """
    Apply a vignette (edge darkening) to simulate camera/eye lens falloff.

    Args:
        img: Input image (uint8, 0-255)
        sigma_ratio: Controls vignette falloff (higher = tighter vignette)

    Returns:
        Vignetted image (uint8)
    """
    rows, cols = img.shape[:2]

    kernel_x = cv2.getGaussianKernel(cols, cols / sigma_ratio)
    kernel_y = cv2.getGaussianKernel(rows, rows / sigma_ratio)

    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()

    img_f = img.astype(np.float32)
    img_out = img_f * mask[:, :, None]

    return np.clip(img_out, 0, 255).astype(np.uint8)