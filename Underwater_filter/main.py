import cv2
import numpy as np
import argparse

from underwater.attenuation import apply_attenuation
from underwater.haze import apply_haze
from underwater.contrast import adjust_contrast
from underwater.blur import apply_blur
from underwater.depth_gradient import apply_depth_gradient
from utils.io_utils import read_image, save_image, read_video, create_video_writer

def underwater_pipeline(frame):
    img = apply_attenuation(frame)

    img = apply_haze(img)

    img = adjust_contrast(img)

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    if img.max() > 1:
        img = img / 255.0

    img = apply_depth_gradient(img)

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    img = apply_blur(img)

    return img


def process_image(input_path, output_path):
    image = read_image(input_path)
    result = underwater_pipeline(image)
    save_image(output_path, result)


def process_video(input_path, output_path):
    cap = read_video(input_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = create_video_writer(output_path, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = underwater_pipeline(frame)
        out.write(processed)

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="image or video")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "image":
        process_image(args.input, args.output)
    elif args.mode == "video":
        process_video(args.input, args.output)