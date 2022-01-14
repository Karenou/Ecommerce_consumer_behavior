import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def plot_image(indices, boxes, classes, scores, image, image_id):
    area, pct_area, confs = [], 0, []
    for i in indices:
        box = boxes[i]
        # filter person class, class_label = 1
        if classes[i] == 1:
            cv2.rectangle(image, 
                          (int(box[0]), int(box[1])), 
                          (int(box[2]), int(box[3])), 
                          (255, 0, 0), 2)
            area.append((box[3] - box[1]) * (box[2] - box[0]))
            confs.append(scores[i])
    
    if len(area) == 0:
        pct_area, avg_conf = 0, 0
    else:
        pct_area = np.mean(area) / (image.shape[0] * image.shape[1])
        avg_conf = np.mean(confs)
    
    plt.figure()
    plt.title("Image {}: {:.0%}% of area is classified as person at {:.0%}% confidence".format(
                    image_id, pct_area, avg_conf)
    )
    plt.imshow(image)
    plt.show()
    plt.close()
    
    return pct_area, avg_conf


def evaluate(model_name, time_used, df, pred_col, label_col="y_true"):
    acc = accuracy_score(df["y_true"], df[pred_col])
    precision = precision_score(df["y_true"], df[pred_col])
    recall = recall_score(df["y_true"], df[pred_col])
    f1 = f1_score(df["y_true"], df[pred_col])
    
    print(model_name)
    print("total time used: %d seconds" % time_used)
    print("accuracy: {:.2%}".format(acc), 
          "precision: {:.2%}".format(precision), 
          "recall: {:.2%}".format(recall),
          "f1: {:.2%}".format(f1))
    print("confusion matrix")
    print(confusion_matrix(df["y_true"], df[pred_col]), "\n")