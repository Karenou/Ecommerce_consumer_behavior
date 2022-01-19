from email.mime import base
from unittest import result
import pandas as pd

# 导入系统库并定义辅助函数
from pprint import pformat

from sklearn.feature_extraction import image

# import PythonSDK
from PythonSDK.facepp import API,File



# 此方法专用来打印api返回的信息
def print_result(hit, result):
    print(hit)
    print('\n'.join("  " + i for i in pformat(result, width=75).split('\n')))

def printFuctionTitle(title):
    return "\n"+"-"*60+title+"-"*60;

def extract_value(res, image_id, data):
    face_num = res["face_num"]
    for i in range(face_num):
        gender = res["faces"][i]["attributes"]["gender"]["value"]
        age = res["faces"][i]["attributes"]["age"]["value"]
        male_atts = res["faces"][i]["attributes"]["beauty"]["male_score"]
        female_atts = res["faces"][i]["attributes"]["beauty"]["female_score"]
        
        data.append([image_id, i, gender, age, male_atts, female_atts])
        print("face_idx: %d, gender: %s, age: %d, male_score: %.1f, female_score: %.1f" % (i, gender, age, male_atts, female_atts))
    return data


# 初始化对象，进行api的调用工作
api = API()
image_base_path = "/Users/karenou/Desktop/RA/HKTVMall_data"
project_base_path = "/Users/karenou/Desktop/RA/Ecommerce_consumer_behavior"
data = []
# -----------------------------------------------------------人脸识别部分-------------------------------------------

human_detect_df = pd.read_csv("%s/human_detection/result/frcnn_result.csv" % project_base_path)
image_ids = human_detect_df[human_detect_df["pred_label"] == 1]["image_id"]

for image_id in image_ids:
    print("processing image %d" % image_id)
    file = File("%s/%d.jpg" % (image_base_path, image_id))
    res = api.detect(
            image_file=file, 
            return_landmark=0, 
            return_attributes="gender,age,beauty"
            )
    data = extract_value(res, image_id, data)

result_df = pd.DataFrame(data, columns=["image_id", "face_idx", "gender", "age", "male_score", "female_score"])
result_df.to_csv("result/face_res.csv", index=False)

# print_result(printFuctionTitle("人脸检测"), res)


