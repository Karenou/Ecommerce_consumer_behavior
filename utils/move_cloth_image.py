import os
import shutil
import glob
import pandas as pd
import argparse

def get_image_id(df, num_cols=17):
    res = None
    for i in range(num_cols):
        if i == 0:
            res = pd.concat([df[str(i)], df[str(i+1)]], axis=0)
        elif i == 1:
            continue
        else:
            res = pd.concat([res, df[str(i)]], axis=0)

    # filter na 
    res = res[res.isna() == False].to_frame()
    res.columns = ["image_id"]
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="base path to load raw images")
    parser.add_argument("--cloth_image_path", type=str, help="path to save images that contain human body")
    parser.add_argument("--image_csv", type=str, help="path to save all image id")
    parser.add_argument("--cloth_csv", type=str, help="path to save all cloth image id")
    args = parser.parse_args()

    full_image_path = args.image_path
    cloth_image_path = args.cloth_image_path

    if not os.path.exists(cloth_image_path):
        os.makedirs(cloth_image_path)
    
    # get image_id
    res = pd.read_csv(args.cloth_csv)
    res = get_image_id(res)
    print("There are %d images to move" % (len(res)))

    image_df = pd.read_csv(args.image_csv)
    image_df["image_path"] = image_df["image_id"].apply(lambda x: args.image_path + "/" + str(x) + ".jpg")

    # save raw image_ids
    # image_df["image_id"].to_csv("/home/juneshi/Desktop/Ecommerce_consumer_behavior/image_ids.csv", index=False)

    cloth_df = res.merge(image_df, on="image_id", how="inner")
    cloth_df = cloth_df.sort_values("image_id")

    for i, img_p in enumerate(cloth_df["image_path"]):
        if i % 1000 == 0:
            print("process %d / %d" % (i, len(cloth_df)))
            
        dst_p = img_p.replace(full_image_path, cloth_image_path)
        shutil.move(img_p, dst_p)