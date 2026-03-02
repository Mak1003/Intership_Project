import numpy as np
import config

def apply_haze(img):
    haze_color = np.array(config.HAZE_COLOR)
    t = config.HAZE_INTENSITY
    img = img * (1 - t) + haze_color * t
    return img