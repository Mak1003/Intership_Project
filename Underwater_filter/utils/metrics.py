import numpy as np

def mean_channel_values(img):
    return {
        "Blue": np.mean(img[:,:,0]),
        "Green": np.mean(img[:,:,1]),
        "Red": np.mean(img[:,:,2])
    }

def contrast_measure(img):
    return np.std(img)