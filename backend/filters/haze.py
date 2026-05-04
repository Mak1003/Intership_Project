import numpy as np


def apply_haze(img: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """
    Apply depth-based haze — increases toward the bottom (deeper = more haze).
    Fixed: original had gradient backwards (was dimming top instead of bottom).

    Args:
        img: Input image (uint8, 0-255)
        strength: Max haze reduction factor (0 = no haze, 0.5 = heavy haze)

    Returns:
        Haze-applied image (uint8)
    """
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    # Haze increases with depth (top=clear, bottom=hazy)
    haze = np.linspace(1.0, 1.0 - strength, h).reshape(h, 1, 1)

    # Add ambient backscatter glow (bluish tint) toward bottom
    backscatter = np.linspace(0, strength * 40, h).reshape(h, 1, 1)
    backscatter_color = np.array([backscatter, backscatter * 0.6, backscatter * 0.3])  # B > G > R

    img_f = img_f * haze
    img_f[:, :, 0] += backscatter[:, :, 0]  # Blue channel boost
    img_f[:, :, 1] += backscatter[:, :, 0] * 0.4

    return np.clip(img_f, 0, 255).astype(np.uint8)