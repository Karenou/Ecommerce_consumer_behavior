import os
import shutil
import glob
import argparse
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="path to load raw images")
    parser.add_argument("--corrupt_image_path", type=str, help="path to save corrupted images")
    args = parser.parse_args()

    full_image_path = args.image_path
    corrupt_image_path = args.corrupt_image_path

    if not os.path.exists(corrupt_image_path):
        os.makedirs(corrupt_image_path)

    image_paths = glob.glob("%s/*.jpg" % full_image_path)
    print("There are %d images" % len(image_paths))

    for img_p in image_paths:
        try:
            img = Image.open(img_p)
        except OSError:
            dst_p = img_p.replace(full_image_path, corrupt_image_path)
            print("move %s to %s" % (img_p, dst_p))
            shutil.move(img_p, dst_p)