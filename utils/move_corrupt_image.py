import os
import shutil
import glob
import cv2
import argparse
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="path to load raw images")
    parser.add_argument("--corrupt_image_path", type=str, help="path to save corrupted images")
    parser.add_argument("--check_cv2", type=bool, default=False, help="whether to check can cv2 read the image")
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
            print("PIL cannot read the image, move %s to %s" % (img_p, dst_p))
            shutil.move(img_p, dst_p)

        if args.check_cv2:
            try:
                img = cv2.imread(img_p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except cv2.error:
                dst_p = img_p.replace(full_image_path, corrupt_image_path)
                print("cv2 cannot read the image, move %s to %s" % (img_p, dst_p))
                shutil.move(img_p, dst_p)