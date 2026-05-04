import numpy as np


def recover_scene(img: np.ndarray, transmission: np.ndarray, A: np.ndarray, t0: float = 0.25) -> np.ndarray:
    """
    Recover the scene radiance from the underwater image formation model:
        J(x) = (I(x) - A) / max(t(x), t0) + A

    Args:
        img: Input image (uint8, 0-255)
        transmission: Transmission map (float32, 0-1)
        A: Atmospheric light vector (3,)
        t0: Minimum transmission clamp to avoid division by zero / over-amplification

    Returns:
        Recovered scene image (uint8, 0-255)
    """
    img_f = img.astype(np.float32)
    t = np.maximum(transmission, t0)

    J = np.zeros_like(img_f)
    for c in range(3):
        J[:, :, c] = (img_f[:, :, c] - A[c]) / t + A[c]

    J = np.clip(J, 0, 255)
    return J.astype(np.uint8)