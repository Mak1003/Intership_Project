import cv2
import numpy as np
import config

def apply_attenuation(img):
    img = img.astype(np.float32) / 255.0
    B, G, R = cv2.split(img)

    R *= config.RED_ATTENUATION
    G *= config.GREEN_ATTENUATION
    B *= config.BLUE_ATTENUATION

    return cv2.merge((B, G, R))