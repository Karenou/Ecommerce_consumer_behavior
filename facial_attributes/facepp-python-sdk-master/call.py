from ast import arg
from email.mime import base
from unittest import result
from matplotlib.pyplot import axis
import pandas as pd

# 导入系统库并定义辅助函数
from pprint import pformat

from sklearn.feature_extraction import image

# import PythonSDK
from PythonSDK.facepp import API,File
import argparse
import os


# 此方法专用来打印api返回的信息
def print_result(hit, result):
    print(hit)
    print('\n'.join("  " + i for i in pformat(result, width=75).split('\n')))

def printFuctionTitle(title):
    return "\n"+"-"*60+title+"-"*60;

def extract_value(res, image_id, data):
    detect_attrs = [i for i in range(len(res["faces"])) if "attributes" in res["faces"][i].keys()]
    
    for i in detect_attrs:
        gender = res["faces"][i]["attributes"]["gender"]["value"]
        age = res["faces"][i]["attributes"]["age"]["value"]
        male_atts = res["faces"][i]["attributes"]["beauty"]["male_score"]
        female_atts = res["faces"][i]["attributes"]["beauty"]["female_score"]
        
        data.append([image_id, i, gender, age, male_atts, female_atts])
        print("face_idx: %d, gender: %s, age: %d, male_score: %.1f, female_score: %.1f" % (i, gender, age, male_atts, female_atts))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_idx", type=int, default=0, help="resume from this image_id")
    parser.add_argument("--progress", type=int, default=100, help="save intermediate csv")
    args = parser.parse_args()

    # 初始化对象，进行api的调用工作
    api = API()
    image_base_path = "/home/juneshi/Desktop/Image"
    project_base_path = "/home/juneshi/Desktop/Ecommerce_consumer_behavior"
    # image_base_path = "/Users/karenou/Desktop/RA/HKTVMall_data"
    # project_base_path = "/Users/karenou/Desktop/RA/Ecommerce_consumer_behavior"
    data = []
    result_df = None
    # -----------------------------------------------------------人脸识别部分-------------------------------------------

    human_detect_df = pd.read_csv("%s/human_detection/result/frcnn_result.csv" % project_base_path)
    image_ids = human_detect_df[human_detect_df["pred_label"] == 1]["image_id"]

    # start from resume_idx
    if args.resume_idx > 0:
        image_ids = image_ids[image_ids >= args.resume_idx]
        
        if os.path.exists("result/face_res.csv"):
            result_df = pd.read_csv("result/face_res.csv")            

    for idx, image_id in enumerate(image_ids):
        print("processing image %d" % image_id)
        file = File("%s/%d.jpg" % (image_base_path, image_id))
        res = api.detect(
                image_file=file, 
                return_landmark=0, 
                return_attributes="gender,age,beauty"
                )
        data = extract_value(res, image_id, data)

        if idx % args.progress == 0 and idx > 0:
            print("----------------process %d / %d, save intermediate result----------------" % (idx, len(image_ids)))
            tmp_df = pd.DataFrame(data, columns=["image_id", "face_idx", "gender", "age", "male_score", "female_score"])
            
            if result_df is not None:
                tmp_df = pd.concat([result_df, tmp_df], axis=0)
            
            tmp_df.to_csv("result/face_res.csv", index=False)

    print("----------------save final result----------------")
    result_df = pd.DataFrame(data, columns=["image_id", "face_idx", "gender", "age", "male_score", "female_score"])
    result_df.to_csv("result/face_res.csv", index=False)



