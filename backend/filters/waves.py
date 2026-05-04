import numpy as np
import cv2


def apply_wave_distortion(img: np.ndarray, amplitude: float = 6.0, frequency: float = 35.0) -> np.ndarray:
    """
    Apply sinusoidal wave distortion to simulate water surface refraction.

    Args:
        img: Input image (uint8, 0-255)
        amplitude: Wave height in pixels
        frequency: Wave frequency (higher = more waves)

    Returns:
        Wave-distorted image (uint8)
    """
    h, w = img.shape[:2]
    map_y, map_x = np.indices((h, w), dtype=np.float32)

    wave = amplitude * np.sin(map_x / frequency)
    map_y = map_y + wave

    img_out = cv2.remap(
        img,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return img_out