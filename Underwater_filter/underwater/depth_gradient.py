import numpy as np

def apply_depth_gradient(img):
    height, width, _ = img.shape
    for i in range(height):
        depth_factor = i / height
        img[i, :, 2] *= (1 - 0.3 * depth_factor)  # reduce red more at bottom
    return img