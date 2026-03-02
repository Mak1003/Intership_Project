import config

def adjust_contrast(img):
    img = img * config.CONTRAST_REDUCTION + config.BRIGHTNESS_SHIFT
    return img