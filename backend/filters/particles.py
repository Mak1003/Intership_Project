import numpy as np


def add_particles(img: np.ndarray, density: float = 1.0, seed: int = None) -> np.ndarray:
    """
    Add suspended particle specks to simulate underwater particulate matter.
    Fixed: original had uint8 overflow — adding to bright pixels wrapped to dark values.

    Args:
        img: Input image (uint8, 0-255)
        density: Multiplier for particle count (1.0 = default density)
        seed: Random seed for reproducibility (set for video frames)

    Returns:
        Image with particles (uint8)
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = img.shape[:2]
    num = int((h * w) / 7000 * density)

    # Work in int32 to avoid uint8 overflow
    img_out = img.astype(np.int32)

    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)

    img_out[ys, xs] += 30

    return np.clip(img_out, 0, 255).astype(np.uint8)