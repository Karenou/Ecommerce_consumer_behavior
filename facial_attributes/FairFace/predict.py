from __future__ import print_function, division
from ast import arg
# from pyexpat import model
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import dlib
import os
import os.path
import argparse
import glob
import time

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def detect_face(base_load_path, image_ids,  base_save_path, 
                default_max_size=800, size = 200, padding = 0.25, progress=10):
    """
    detect and crop the face region in each image
    @param base_load_path: base path of image folder
    @param image_ids: list of image_ids
    @param base_save_path: base path to save detected face image
    @param default_max_size: largest width or height to resize the image
    @param size: parameter used in cropping the face region
    @param padding: parameter used in cropping the face region
    @param progress: print out progress per n images
    """
    # load pre-trained face detection and landmark extraction models
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat') 

    # predict the progress per 1000 images
    for index, image_id in enumerate(image_ids):
        if index % progress == 0:
            print('---%d/%d---' %(index, len(image_ids)))
        
        # skip the images that cannot be opened due to unknown image file format
        image_path = "%s/%d.jpg" % (base_load_path, image_id)
        try:
            img = dlib.load_rgb_image(image_path)
        except RuntimeError:
            print("Cannot load the image %d" % image_id)
            continue

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(image_path))
            continue

        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
        
        # save the detected face region, there could be multiple faces per image
        images = dlib.get_face_chips(img, faces, size=size, padding = padding)
        for idx, image in enumerate(images):
            path_sp = image_path.split("/")[-1].split(".")
            face_name = os.path.join(base_save_path,  path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
            dlib.save_image(image, face_name)

def index_to_category_name(result, col_name, feat_name, num_categories, name_list):
    """
    convert category index to name in result dataframe
    @param result: pandas dataframe
    @param col_name: name of column
    @param feat_name: feature name
    @param num_categories: number of categories, the index ranges from 0 to num_categories - 1
    @param name_list: list of category names
    """
    for index, name in zip(list(range(num_categories)), name_list):
        result.loc[result[col_name] == index, feat_name] = name
    return result

def predict_age_gender_race(use_cuda, save_path, model_path, num_races=4, imgs_path = 'detected_faces/', progress=10):
    """
    @param use_cuda: -1 for not use GPU, otherwise specify which cuda
    @param save_path: path to save prediction result csv
    @param model_path: path of pretrained model
    @param num_races: 4 or 7, decide which model to use
    @param imgs_path: path to save the detected faces
    @param progress: print out progress per n images
    """
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path) if "jpg" in x]
    device = torch.device("cuda:%d" % use_cuda if torch.cuda.is_available() and use_cuda != -1 else "cpu")

    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images_ids = []
    race_scores_fair, gender_scores_fair, age_scores_fair = [], [], []
    race_preds_fair, gender_preds_fair, age_preds_fair = [], [], []

    for index, img_name in enumerate(img_names):
        if index % progress == 0:
            print("Predicting... {}/{}".format(index, len(img_names)))

        # image_id: 1_face0
        images_ids.append(img_name.split("/")[-1].split(".")[0])
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        outputs = model(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:num_races]
        gender_outputs = outputs[num_races:num_races+2]
        age_outputs = outputs[num_races+2:num_races+11]
        
        # apply softmax
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        # get predict label
        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)


    result = pd.DataFrame([images_ids,
                           race_preds_fair,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair, 
                           gender_scores_fair,
                           age_scores_fair, ]).T
    result.columns = ['image_id',
                      'race_preds',
                      'gender_preds',
                      'age_preds',
                      'race_scores',
                      'gender_scores',
                      'age_scores']
    # race
    if num_races == 7:
        result = index_to_category_name(
            result, 'race_preds', 'race', 7, 
            ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        )
    else:
        result = index_to_category_name(
            result, 'race_preds', 'race', 4, 
            ['White', 'Black', 'Asian', 'Indian']
        )

    # gender
    result = index_to_category_name(result, 'gender_preds', 'gender', 2, ['Male', 'Female'])

    # age
    result = index_to_category_name(
        result, 'age_preds', 'age_group', 9, 
        ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    )

    print("saving prediction result")
    ensure_dir(save_path.split("/")[0])
    result["face_idx"] = result["image_id"].apply(lambda x: x.split("_")[1])
    result["image_id"] = result["image_id"].apply(lambda x: int(x.split("_")[0]))
    result = result.sort_values("image_id")
    result[['image_id', "face_idx", 'race', 'gender', 'age_group',
            'race_scores', 'gender_scores', 'age_scores']].to_csv(save_path, index=False)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Base path of image folder")
    parser.add_argument("--input_csv", type=str, help="The human detection result csv")
    parser.add_argument("--face_path", type=str, help="Base path to save the images of detected faces")
    parser.add_argument("--save_path", type=str, help="Path to save the prediction result csv")
    parser.add_argument("--face_size", default=200, type=int, help="Size to crop detected human face")
    parser.add_argument("--num_races", default=4, type=int, help="Number of races")
    parser.add_argument("--model_path", type=str, help="Path to load the pretrain model")
    parser.add_argument("--progress", default=20, type=int, help="Print progress for every n images")
    parser.add_argument("--use_cuda", type=int, default=0, help="-1 not use cuda, otherwise is the cuda idx, from 0 to 3")
    args = parser.parse_args()

    print("torch CUDA available: %s" % torch.cuda.is_available())
    if args.use_cuda == -1:
        dlib.DLIB_USE_CUDA = False
        print("use cpu")
    else:
        dlib.DLIB_USE_CUDA = True
        print("using CUDA:%d" % args.use_cuda)
    
    ensure_dir(args.face_path)

    pred_csv = pd.read_csv(args.input_csv)
    img_ids = pred_csv[pred_csv["pred_label"] == 1]["image_id"]

    start = time.time()
    print("detecting faces, saved at %s" % args.save_path)
    detect_face(args.input_path, img_ids, args.face_path, size=args.face_size, progress=args.progress)

    print("predicting age, gender and race")
    predict_age_gender_race(args.use_cuda, args.save_path, args.model_path, num_races=args.num_races, imgs_path=args.face_path + '/', progress=args.progress)

    end = time.time()
    print("Total time used: %.4f" % (end-start))