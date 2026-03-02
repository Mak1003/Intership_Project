import cv2
import config

def apply_blur(img):
    if config.ENABLE_BLUR:
        return cv2.GaussianBlur(img, config.BLUR_KERNEL, 0)
    return img