import cv2
import numpy as np

from underwater.dark_channel import dark_channel
from underwater.transmission import estimate_atmospheric_light, estimate_transmission
from underwater.recover_scene import recover_scene

from filters.color_shift import color_shift
from filters.haze import apply_haze
from filters.depth_blur import depth_blur
from filters.light_rays import apply_light_rays
from filters.particles import add_particles
from filters.waves import apply_wave_distortion
from filters.vignette import apply_vignette


def simulate_pipeline(img: np.ndarray, params: dict = None, frame_seed: int = None) -> np.ndarray:
    """
    Simulate underwater degradation on a clean image.

    Pipeline:
      1. Color shift  — attenuate red, boost blue
      2. Haze         — add depth-based backscatter
      3. Depth blur   — progressive focus loss with depth
      4. Light rays   — caustic ray simulation
      5. Particles    — suspended particulate matter
      6. Wave distortion — surface refraction
      7. Vignette     — lens falloff

    Args:
        img: Clean input image (uint8 BGR)
        params: Optional dict to override filter parameters
        frame_seed: Seed for stochastic effects (use frame index for videos)

    Returns:
        Simulated underwater image (uint8 BGR)
    """
    p = params or {}

    img = color_shift(
        img,
        r_scale=p.get("r_scale", 0.75),
        g_scale=p.get("g_scale", 0.9),
        b_scale=p.get("b_scale", 1.05)
    )
    img = apply_haze(img, strength=p.get("haze_strength", 0.15))
    img = depth_blur(img, max_strength=p.get("blur_strength", 0.5))
    img = apply_light_rays(img, intensity=p.get("ray_intensity", 0.03), seed=frame_seed)
    img = add_particles(img, density=p.get("particle_density", 1.0), seed=frame_seed)
    img = apply_wave_distortion(
        img,
        amplitude=p.get("wave_amplitude", 6.0),
        frequency=p.get("wave_frequency", 35.0)
    )
    img = apply_vignette(img)

    return img


def restore_pipeline(img: np.ndarray, params: dict = None) -> np.ndarray:
    """
    Restore an underwater image to near-natural appearance using
    the Dark Channel Prior (Sea-Thru inspired).

    Pipeline:
      1. Dark channel estimation
      2. Atmospheric light (backscatter) estimation
      3. Transmission map estimation
      4. Scene radiance recovery: J = (I - A) / t + A

    Args:
        img: Degraded underwater image (uint8 BGR)
        params: Optional dict to override restoration parameters

    Returns:
        Restored image (uint8 BGR)
    """
    p = params or {}

    dark = dark_channel(img.astype(np.float32) / 255.0)
    A = estimate_atmospheric_light(img, dark)
    t = estimate_transmission(img, A, omega=p.get("omega", 0.95))
    img = recover_scene(img, t, A, t0=p.get("t0", 0.25))

    # Post-processing: subtle contrast boost
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return img