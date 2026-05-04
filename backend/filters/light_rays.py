import numpy as np
import cv2


def apply_light_rays(img: np.ndarray, num_rays: int = 5, intensity: float = 0.03, seed: int = None) -> np.ndarray:
    """
    Simulate caustic light rays filtering down through water.
    Fixed: added optional seed for reproducible rays (prevents video flicker).

    Args:
        img: Input image (uint8, 0-255)
        num_rays: Number of light ray streaks
        intensity: Brightness of rays (0-1)
        seed: Random seed for reproducibility (set for video frames)

    Returns:
        Image with light rays (uint8)
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = img.shape[:2]
    rays = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_rays):
        x = np.random.randint(0, w)
        dx = np.random.randint(-80, 80)
        cv2.line(rays, (x, 0), (x + dx, h), 1.0, 30)

    rays = cv2.GaussianBlur(rays, (101, 101), 0)

    fade = np.linspace(1, 0.2, h).reshape(h, 1)
    rays = rays * fade * intensity

    img_f = img.astype(np.float32) / 255.0
    img_f = img_f + rays[:, :, None]
    img_f = np.clip(img_f, 0, 1)

    return (img_f * 255).astype(np.uint8)