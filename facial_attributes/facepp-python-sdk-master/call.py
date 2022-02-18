import pandas as pd

# 导入系统库并定义辅助函数
from pprint import pformat

# import PythonSDK
from PythonSDK.facepp import API,File, APIError
import argparse
import os
import glob


# 此方法专用来打印api返回的信息
def print_result(hit, result):
    print(hit)
    print('\n'.join("  " + i for i in pformat(result, width=75).split('\n')))

def printFuctionTitle(title):
    return "\n"+"-"*60+title+"-"*60;

def extract_facial_value(res, image_id, data):
    detect_attrs = [i for i in range(len(res["faces"])) if "attributes" in res["faces"][i].keys()]
    for i in detect_attrs:
        gender = res["faces"][i]["attributes"]["gender"]["value"]
        age = res["faces"][i]["attributes"]["age"]["value"]
        male_atts = res["faces"][i]["attributes"]["beauty"]["male_score"]
        female_atts = res["faces"][i]["attributes"]["beauty"]["female_score"]
        
        data.append([image_id, i, gender, age, male_atts, female_atts, 1])
        print("face_idx: %d, gender: %s, age: %d, male_score: %.1f, female_score: %.1f" % (i, gender, age, male_atts, female_atts))
    return data

def extract_body_value(res, image_id, data):
    if len(res["humanbodies"]) > 0:
        detect_bodys = [i for i in range(len(res["humanbodies"])) if "humanbody_rectangle" in res["humanbodies"][i].keys()]
        for i in detect_bodys:
            conf = res["humanbodies"][i]["confidence"]
            h = res["humanbodies"][i]["humanbody_rectangle"]["height"]
            l = res["humanbodies"][i]["humanbody_rectangle"]["left"]
            t = res["humanbodies"][i]["humanbody_rectangle"]["top"]
            w = res["humanbodies"][i]["humanbody_rectangle"]["width"]
            
            data.append([image_id, i, conf, h, l, t, w, 1])
            print("body_idx: %d, conf: %.1f, height: %d, left: %d, top: %d, width: %d" % (i, conf, h, l, t, w))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="face_detect", help="which api service to call: face_detect, body_detect, body_skeleton")
    parser.add_argument("--attributes", type=str, default="gender,age,beauty", help="return_attributes")
    parser.add_argument("--output_csv", type=str, help="output file pathway")
    parser.add_argument("--resume_idx", type=int, default=0, help="resume from this image_id")
    parser.add_argument("--progress", type=int, default=100, help="save intermediate csv")
    args = parser.parse_args()

    # 初始化对象，进行api的调用工作
    api = API()
    # image_base_path = "/home/juneshi/Desktop/Image"
    # project_base_path = "/home/juneshi/Desktop/Ecommerce_consumer_behavior"
    image_base_path = "/Users/karenou/Desktop/RA/human_images"
    project_base_path = "/Users/karenou/Desktop/RA/Ecommerce_consumer_behavior"
    data = []
    result_df = None
    # -----------------------------------------------------------production-------------------------------------------

    # human_detect_df = pd.read_csv("%s/human_detection/result/frcnn_result.csv" % project_base_path)
    # image_ids = human_detect_df[human_detect_df["pred_label"] == 1]["image_id"]
    image_ids = [int(path.split("/")[-1].split(".")[0]) for path in glob.glob("%s/*.jpg" % image_base_path)]
    image_ids.sort()

    # start from resume_idx
    if args.resume_idx > 0:
        image_ids = image_ids[image_ids >= args.resume_idx]

        if os.path.exists(args.output_csv):
            result_df = pd.read_csv(args.output_csv)            

    # submit url request and extract attributes from result
    for idx, image_id in enumerate(image_ids):
        print("processing image %d" % image_id)
        file = File("%s/%d.jpg" % (image_base_path, image_id))

        # the size limit for face detect is (4096, 4096)
        if args.mode == "face_detect":
            res = api.detect(image_file=file, return_attributes=args.attributes)
            data = extract_facial_value(res, image_id, data)
            col_list = ["image_id", "face_idx", "gender", "age", "male_score", "female_score", "detect_face"]
        # the size limit for body detect is (1280, 1280)
        elif args.mode == "body_detect":
            try:
                res = api.detect(image_file=file)
            except APIError:
                file = File("%s/%d.jpg" % (image_base_path, image_id), resize=True)
                res = api.detect(image_file=file)
            data = extract_body_value(res, image_id, data)
            col_list = ["image_id", "body_idx", "confidence", "height", "left", "top", "width", "detect_body"]
        else:
            res = api.skeleton(image_file=file, return_attributes=args.attributes)
            # data = extract_body_value(res, image_id, data)
            # col_list = 
        
        if idx % args.progress == 0 and idx > 0:
            print("----------------process %d / %d, save intermediate result----------------" % (idx, len(image_ids)))
            tmp_df = pd.DataFrame(data, columns=col_list)
            
            if result_df is not None:
                tmp_df = pd.concat([result_df, tmp_df], axis=0)
            
            tmp_df.to_csv(args.output_csv, index=False)

    print("----------------save final result----------------")
    result_df = pd.DataFrame(data, columns=col_list)
    result_df.to_csv(args.output_csv, index=False)

    # -----------------------------------------------------------Test single image-------------------------------------------
    # -----------------------------------------------------------人体识别部分-------------------------------------------
    # image_id = 1109
    # print("processing image %d" % image_id)
    # try:
    #     file = File("%s/%d.jpg" % (image_base_path, image_id))
    #     res = api.detect(image_file=file)
    # except APIError:
    #     file = File("%s/%d.jpg" % (image_base_path, image_id), resize=True)
    #     res = api.detect(image_file=file)
    # print_result(printFuctionTitle("人体检测"), res)
    # data = extract_body_value(res, image_id, data)
    # col_list = ["image_id", "body_idx", "confidence", "height", "left", "top", "width", "detect_body"]
    # result_df = pd.DataFrame(data, columns=col_list)
    # print(result_df.head())

    # -----------------------------------------------------------人体部位识别-------------------------------------------
    # image_id = 8513
    # print("processing image %d" % image_id)
    # file = File("%s/%d.jpg" % (image_base_path, image_id))
    # res = api.skeleton(
    #         image_file=file, 
    #         return_attributes="gender,upper_body_cloth,lower_body_cloth"
    #         )
    # print_result(printFuctionTitle("人体部位"), res)


