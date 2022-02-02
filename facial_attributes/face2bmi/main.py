import pandas as pd
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import argparse
import tensorflow as tf

from models import FacePrediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", type=int, default=0, help="-1 for using cpu")
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument('--model_type', type=str, default="vgg16", help="vgg16 or resnet50")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--epochs', type=int, default=3, help="number of epochs in training")
    parser.add_argument('--image_path', type=str, default='', help='path to load face images')
    args = parser.parse_args()

    allimages = os.listdir('./data/face_aligned/')
    train = pd.read_csv('./data/train.csv')
    valid = pd.read_csv('./data/valid.csv')

    train = train.loc[train['index'].isin(allimages)]
    valid = valid.loc[valid['index'].isin(allimages)]

    mode = args.mode
    model_type = args.model_type
    model_tag = 'base'
    model_dir = './model/%s_%s_epoch=%d_bs=%d.h5' % (model_type, model_tag, args.epochs, args.batch_size)
    Path('./model').mkdir(parents = True, exist_ok = True)

    # set gpu memory limit
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("GPU available, using %s" % gpus[args.use_cuda].name)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_cuda)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[args.use_cuda],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)]
    )

    model = FacePrediction(img_dir = './data/face_aligned/', model_type = model_type)
    model.define_model(freeze_backbone = True)
    if mode == 'train':
        print("Start model training")
        es = EarlyStopping(patience=3)
        ckp = ModelCheckpoint(model_dir, save_best_only=True, save_weights_only=True, verbose=1)
        model_history = model.train(train, valid, bs = args.batch_size, epochs = args.epochs, callbacks = [es, ckp])
    else:
        model.load_weights(model_dir)
        if args.image_path:
            print("Predict BMI on face images under %s" % args.image_path)
            pred = model.predict_df(args.image_path)
            pred = pred.sort_values(["image_id", "face_id"])
            Path('./result').mkdir(parents = True, exist_ok = True)
            pred.to_csv("result/bmi_pred_epoch=%d_bs=%d.csv" % (args.epochs, args.batch_size), index=False)