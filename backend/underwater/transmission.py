import numpy as np
import cv2
from .dark_channel import dark_channel


def estimate_atmospheric_light(img: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """
    Estimate the atmospheric light (backscatter) from the brightest
    pixels in the dark channel.

    Args:
        img: Input image (uint8, 0-255)
        dark: Precomputed dark channel map

    Returns:
        Atmospheric light vector (3,) float64
    """
    h, w = dark.shape
    num_pixels = h * w
    num_bright = int(max(num_pixels * 0.001, 1))

    dark_vec = dark.reshape(num_pixels)
    img_vec = img.reshape(num_pixels, 3)

    indices = dark_vec.argsort()[-num_bright:]
    A = np.mean(img_vec[indices], axis=0)
    return A


def estimate_transmission(img: np.ndarray, A: np.ndarray, omega: float = 0.95) -> np.ndarray:
    """
    Estimate the transmission map using the dark channel prior.

    Args:
        img: Input image (uint8, 0-255)
        A: Atmospheric light vector
        omega: Keeps a small amount of haze for realism (0-1)

    Returns:
        Transmission map (float32, 0-1)
    """
    # Avoid division by zero
    A_safe = np.maximum(A, 1e-6)
    norm_img = img.astype(np.float32) / A_safe

    transmission = 1.0 - omega * dark_channel(norm_img, 15)
    transmission = cv2.GaussianBlur(transmission.astype(np.float32), (15, 15), 0)
    return transmission