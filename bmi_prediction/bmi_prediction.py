from random import shuffle
import cv2
import os
import time
import numpy as np
import pandas as pd
import math

import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing import image
import argparse

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

def load_base_model(input_shape=(224, 224, 3)):
    """
    load a pretrained ResNet152 model to extract features 
    @param input_shape: shape of input to the model
    """
    model = tf.keras.applications.ResNet152(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )
    model.trainable = False
    return model

def extract_features(x, datagen, model, batch_size, n_images):
    """
    extract features from images
    @param x: input images
    @param datagen: data generator
    @param model: resnet152
    @param batch_size
    @param n_images: number of images
    """
    print("---------Extract features")
    features = np.zeros(shape=(n_images, 2048))
    
    generator = datagen.flow(x, batch_size=batch_size, shuffle=False)
    
    total = 0
    left_index = 0
    
    for inputs_batch in generator:
        features_batch = model.predict(inputs_batch)
        
        # track the number of processed images and 
        gap = features_batch.shape[0]
        total += gap
        gap_diff = total - n_images
        gap_diff = gap_diff if gap_diff > 0 else 0
        
        features[left_index : left_index + (gap-gap_diff)] = features_batch[0:gap-gap_diff]
        left_index += gap
        
        if total >= n_images:
            break
            
    return features

def read_image(image_path, process_image_path, image_ids, input_shape=(224,224,3)):
    """
    size of processed image: (512, 512, 3)
    @param image_path: base path of raw input image
    @param process_image_path: base path of human parsing output images
    @param image_ids: a list of image ids that contain human body
    @param input_shape: default input shape of resenet model to resize images
    """
    print("---------Read images")
    input_images = []
    # for filter corrupted image
    corrupt_imgs = []
    for idx, image_id in enumerate(image_ids):
        # load raw input image
        try:
            raw_input_image = cv2.imread("%s/%d.jpg" % (image_path, image_id))
            raw_input_image = cv2.cvtColor(raw_input_image, cv2.COLOR_BGR2RGB)
            raw_input_image = cv2.resize(raw_input_image, (input_shape[0], input_shape[1]))
        except cv2.error:
            print("%d.jpg is corrupted, skip this image." % image_id)
            corrupt_imgs.append(idx)
            continue

        # load processed image after human parsing
        preprocessed_input_image = image.load_img(
            "%s/%d.png"% (process_image_path, image_id), 
            target_size=input_shape
        )
        preprocessed_input_image = image.img_to_array(preprocessed_input_image)

        # convert to silhouette images
        for channel in range(3):
            preprocessed_input_image[preprocessed_input_image[:,:,channel] > 0] = 1

        input_images.append(raw_input_image * preprocessed_input_image)
    
    return np.array(input_images), corrupt_imgs

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def skip_corrupt_pred(pred, corrupt_imgs_idx, batch_size=8):
    """
    set the predicted bmi for corrupted image to 0
    ensure the size of final bmi predict is (batch_size, 1)
    @param pred: model prediction, size (batch_size - len(corrupt_imgs_idx), 1)
    @param corrupt_imgs_idx: a list of currpt imgs index in batch
    """
    final_pred = np.zeros((batch_size, 1))
    j, k = 0, 0
    for i in range(batch_size):
        if j < len(corrupt_imgs_idx) and i == corrupt_imgs_idx[j]:
            j += 1
            continue
        else:
            final_pred[i] = pred[k]
            k += 1
    return final_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path of human detection result csv")
    parser.add_argument("--image_path", type=str, required=True, help="Base path to save raw input images")
    parser.add_argument("--process_image_path", type=str, required=True, help="Base path to save the images of detected faces")
    parser.add_argument("--model_path", type=str, required=True, help="Path to load the pretrain model")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output bmi_pred csv")
    parser.add_argument("--use_cuda", type=int, default=0, help="which cuda to use")
    parser.add_argument("--resume_idx", type=int, default=0, help="resume batch idx")
    args = parser.parse_args()

    # load image_ids that are detected to contain human body
    human_detect_res = pd.read_csv(args.input_csv)
    image_ids = human_detect_res[human_detect_res["pred_label"] == 1]["image_id"]

    image_base_path = args.image_path
    process_image_path = args.process_image_path
    batch_size = 8
    input_shape = (224, 224, 3)

    start = time.time()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("GPU available, using %s" % gpus[args.use_cuda].name)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_cuda)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[args.use_cuda],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]
        )
    else:
        print("GPU not available")

    # load and preprocess images, save result every 10 batches
    n_batch = math.ceil(len(image_ids) / batch_size)
    bmi_df = human_detect_res[human_detect_res["pred_label"] == 1].reset_index(drop=True).copy()
    bmi_pred = np.zeros(shape=(len(image_ids), 1))

    print("---------Load model")
    base_model = load_base_model(input_shape)
    model = keras.models.load_model(args.model_path, 
                                            custom_objects= {'coeff_determination': coeff_determination}
                                        )

    for i in range(args.resume_idx, n_batch):
        print("---------Batch %d, Process %d/%d" % (i, i * batch_size, len(image_ids)))
        if i != n_batch - 1:
            input_images, corrupt_imgs = read_image(image_base_path, process_image_path, image_ids[i*batch_size:(i+1)*batch_size], input_shape)
            n_images = len(image_ids[i*batch_size:(i+1)*batch_size])
        else:
            input_images, corrupt_imgs = read_image(image_base_path, process_image_path, image_ids[i*batch_size:], input_shape)
            n_images = len(image_ids[i*batch_size:])

        # extract features using ResNet152 base_model
        features = base_model.predict(input_images)

        # predict bmi using pretrained model
        print("---------Predicting BMI")
        preds = model.predict(features)

        # set corrupt images' bmi prediction as 0
        if len(corrupt_imgs) > 0:
            preds = skip_corrupt_pred(preds, corrupt_imgs, batch_size=batch_size)

        if i != n_batch - 1:
            bmi_pred[i*batch_size:(i+1)*batch_size] = preds
        else:
            bmi_pred[i*batch_size:] = preds

        bmi_df["pred_bmi"] = bmi_pred

        if not os.path.exists("result"):
            os.makedirs("result")

    bmi_df.to_csv(args.output_csv, index=False)
    
    end = time.time()
    print("Total time used: %.2f" % (end-start))