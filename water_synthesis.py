import numpy as np
import cv2
import imageio

EPS = 1e-8
def clamp01(x): return np.clip(x, 0.0, 1.0)

# Two-term exponential for betaD(z) (Eq. 11)
def betaD_two_term(z, a, b, c, d):
    return a * np.exp(b * z) + c * np.exp(d * z)

def add_water_effect(J_rgb, z, params=None, debug=False):
    """
    J_rgb : HxWx3 float in [0,1], linear RGB (not gamma corrected); if sRGB, convert with inverse gamma first for best results.
    z : HxW depths in meters
    params : dictionary (see defaults below)
    Returns I_out (HxWx3 in [0,1])
    """
    H, W, _ = J_rgb.shape
    assert z.shape == (H, W), "depth shape mismatch"

    # Defaults (tweakable presets)
    if params is None:
        params = {}
    # BetaD params per channel: (a,b,c,d) for two-term exp.
    # Example defaults (synthetic): these produce decaying beta with z.
    # You should tune these to mimic Jerlov water types.
    betaD_params = params.get('betaD_params', np.array([
        [0.35, -0.35, 0.10, -0.02],  # R
        [0.25, -0.30, 0.08, -0.015], # G
        [0.18, -0.22, 0.06, -0.01],  # B
    ], dtype=np.float32))  # shape (3,4)

    # Backscatter B_inf and betaB (assume single betaB per channel or scalar)
    B_inf = params.get('B_inf', np.array([0.6, 0.65, 0.7], dtype=np.float32))
    betaB = params.get('betaB', np.array([0.3, 0.28, 0.25], dtype=np.float32))

    # Illumination attenuation with depth (Kd per channel) optionally to color-shift ambient light
    Kd = params.get('Kd', np.array([0.15, 0.08, 0.05], dtype=np.float32))  # bigger Kd => faster loss of that band
    sensor_white = params.get('surface_illuminant', np.array([1.0, 1.0, 1.0], dtype=np.float32))  # E(0)

    # Optionally forward-scatter blur kernel size factor (per meter)
    blur_strength = params.get('blur_strength', 0.0)  # 0 = no blur; >0 adds depth-dependent Gaussian blur

    # Convert parameters into arrays
    z_flat = z.ravel()
    # Compute betaD per channel per pixel
    betaD_field = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        a,b,c2,d = betaD_params[c]
        beta_flat = a * np.exp(b * z_flat) + c2 * np.exp(d * z_flat)
        betaD_field[..., c] = beta_flat.reshape(H, W)

    # Compute transmission T = exp(-betaD * z)
    T = np.exp(- betaD_field * z[..., np.newaxis])
    T = clamp01(T)

    # Compute direct attenuated component D = J * T
    D = J_rgb * T

    # Compute backscatter per pixel: B(z) = B_inf * (1 - exp(-betaB * z))
    # Broadcast betaB and B_inf
    B = np.zeros_like(D)
    for c in range(3):
        B[..., c] = B_inf[c] * (1.0 - np.exp(-betaB[c] * z))

    # Ambient illuminant (depth-dependent) — simulating color shift due to downwelling light
    # E(d,lambda) = E(0,lambda) * exp(-Kd(lambda) * d)
    # We treat 'd' = depth of camera (or average). If you want per-pixel depth-dependent ambient
    # effect (for scenes with variations in depth/d), you can use z or a separate scene depth d.
    # Here we use camera depth / or apply per pixel as below:
    Ehat = sensor_white[np.newaxis, np.newaxis, :] * np.exp(-Kd[np.newaxis, np.newaxis, :] * (z[..., np.newaxis]))
    # Apply the change in white point by scaling the total intensity (optional; small effect)
    D = D * Ehat

    # Compose final image
    I = D + B
    I = clamp01(I)

    # Optional forward-scatter blur: simulate PSF causing distant objects to blur/haze more
    if blur_strength is not None and blur_strength > 0.0:
        # depth-dependent sigma: sigma = blur_strength * z (e.g., 0.02 px/m - tune)
        I_blur = np.zeros_like(I)
        # apply different blur per depth quantile for efficiency
        n_bins = 8
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        bins = np.linspace(zmin, zmax + 1e-6, n_bins+1)
        for i in range(n_bins):
            mask = (z >= bins[i]) & (z < bins[i+1])
            if not np.any(mask): continue
            zc = 0.5*(bins[i]+bins[i+1])
            sigma = blur_strength * zc
            # gaussian blur the entire image with this sigma, then copy masked pixels
            k = max(1, int(round(sigma*6)))  # kernel rough size
            # cv2.GaussianBlur requires integer kernel size; if sigma is 0 -> skip blur
            if sigma < 0.01:
                blurred = I
            else:
                ksz = k if (k % 2 == 1) else k+1
                blurred = cv2.GaussianBlur((I*255).astype(np.uint8), (ksz, ksz), sigmaX=sigma, sigmaY=sigma)
                blurred = blurred.astype(np.float32)/255.0
            I_blur[mask] = blurred[mask]
        # where no mask applied, fallback to I
        fallback_mask = (I_blur.sum(axis=2)==0)
        I_blur[fallback_mask] = I[fallback_mask]
        I = I_blur

    # Optionally add noise to simulate scattering/turbidity
    turbidity_noise = params.get('turbidity_noise', 0.0)
    if turbidity_noise and turbidity_noise > 0.0:
        noise = np.random.normal(scale=turbidity_noise, size=I.shape).astype(np.float32)
        I = clamp01(I + noise)

    return I

# Example usage:
if __name__ == "__main__":
    # load an example linear image (convert sRGB to linear before feed)
    # You may convert sRGB to linear (approx): linear = ((srgb <= 0.04045) ? srgb/12.92 : ((srgb+0.055)/1.055)**2.4)
    def srgb_to_linear(img):
        img = np.clip(img, 0.0, 1.0)
        low = img <= 0.04045
        out = np.zeros_like(img)
        out[low] = img[low] / 12.92
        out[~low] = ((img[~low] + 0.055) / 1.055) ** 2.4
        return out

    def linear_to_srgb(img):
        img = np.clip(img, 0.0, 1.0)
        low = img <= 0.0031308
        out = np.zeros_like(img)
        out[low] = img[low] * 12.92
        out[~low] = 1.055 * (img[~low] ** (1.0 / 2.4)) - 0.055
        return out

    # Example file paths (replace with your files)
    J_path = "input_rgb.png"    # in-air image (sRGB)
    z_path = "depth.npy"        # depth map in meters

    # If you have files, uncomment and run
    # J = imageio.imread(J_path).astype(np.float32)/255.0
    # J_lin = srgb_to_linear(J)
    # z = np.load(z_path)
    # I_water = add_water_effect(J_lin, z)
    # I_srgb = linear_to_srgb(I_water)
    # imageio.imwrite("synth_underwater.png", (I_srgb*255).astype(np.uint8))
