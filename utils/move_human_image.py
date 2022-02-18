import os
import shutil
import glob
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="path to load raw images")
    parser.add_argument("--human_image_path", type=str, help="path to save images that contain human body")
    parser.add_argument("--input_csv", type=str, help="path to load body detect csv")
    args = parser.parse_args()

    full_image_path = args.image_path
    human_image_path = args.human_image_path

    if not os.path.exists(human_image_path):
        os.makedirs(human_image_path)
        
    res = pd.read_csv(args.input_csv)
    res = res[["image_id", "detect_body"]].drop_duplicates()
    print("There are %d images to move" % (len(res)))

    image_paths = glob.glob("%s/*.jpg" % full_image_path)
    image_ids = [int(path.split("/")[-1].split(".")[0]) for path in image_paths]

    image_df = pd.DataFrame.from_dict({"image_id": image_ids, "image_path": image_paths}, orient="columns")

    human_df = res.merge(image_df, on="image_id", how="inner")
    human_df = human_df.sort_values("image_id")

    for i, img_p in enumerate(human_df["image_path"]):
        if i % 1000 == 0:
            print("process %d / %d" % (i, len(res)))
            
        dst_p = img_p.replace(full_image_path, human_image_path)
        # print("move %s to %s" % (img_p, dst_p))
        shutil.move(img_p, dst_p)