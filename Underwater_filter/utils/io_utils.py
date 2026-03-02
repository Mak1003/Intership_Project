import cv2

def read_image(path):
    return cv2.imread(path)

def save_image(path, img):
    cv2.imwrite(path, img)

def read_video(path):
    return cv2.VideoCapture(path)

def create_video_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, size)