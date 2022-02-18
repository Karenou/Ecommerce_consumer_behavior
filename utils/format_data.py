"""
this py is used to reformat the data from image_id level to sku_id level
-- numeric (0 for no body or face)
    has_model: 0 or 1, detect_body
    has_face: 0 or 1
    body_bmi: pred_bmi
    face_bmi: bmi
    age: age
    face_attractiveness: mean of female_score and male_score
    image_quality_technical_score: mean_score_prediction
    image_quality_aesthetic_score: mean_score_prediction
-- string ("" for no face)
    gender: gender 
    race: race
    cloth_attributes: 
"""

import pandas as pd 
import numpy as np  
import argparse

def create_feat_dict(df, feat_col="", feat_val=1):
    """
    create the mapping from image id to feature value
    """
    feat_dict = {}
    for i, image_id in enumerate(df["image_id"].drop_duplicates()):
        if feat_col != "":
            feat_dict[image_id] = df[feat_col][i]
        else:
            feat_dict[image_id] = feat_val

    return feat_dict

def map_value(arr, feat_dict):
    for i, image_id in enumerate(arr):
        # the first column is sku_id
        if i > 0 and not np.isnan(image_id):
                arr[i] = feat_dict.get(image_id, 0)
    return arr            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_index_path", type=str, help="path to load image_index csv")
    parser.add_argument("--feat_path", type=str, help="path to load feature csv")
    parser.add_argument("--feat_col", type=str, default="", help="name of feature column")
    parser.add_argument("--feat_val", type=int, default=1, help="value of feature")
    parser.add_argument("--output_path", type=str, help="path to save reformatted data")
    args = parser.parse_args()

    image_index = pd.read_csv(args.image_index_path)
    res = image_index.copy()

    feat = pd.read_csv(args.feat_path)
    feat_dict = create_feat_dict(feat, feat_col=args.feat_col, feat_val=args.feat_val)

    res = res.apply(lambda x: map_value(x, feat_dict), axis=1)
    res.to_csv(args.output_path, index=False)