from mtcnn.mtcnn import MTCNN
import cv2
import os
import shutil
import tqdm
import matplotlib.pyplot as plt

def crop_img(im,x,y,w,h):
    return im[y:(y+h),x:(x+w),:]

def cut_negative_boundary(box):
    res = []
    for x in box['box']:
        if x < 0:
            x = 0
        res.append(x)
    box['box'] = res
    return box

def detect_face(detector, face_path):
    """
    detect single faces
    """
    img = cv2.cvtColor(cv2.imread(face_path), cv2.COLOR_BGR2RGB)
    box = detector.detect_faces(img)[0]
    return box


if __name__ == "__main__":

    test_dir = './data/test/single_face/'
    test_processed_dir = './data/test/test_aligned/'
    train_dir = './data/face/'
    train_processed_dir = './data/face_aligned/'

    if os.path.exists(train_processed_dir):
        shutil.rmtree(train_processed_dir)
    os.mkdir(train_processed_dir)

    print("Crop faces in train and valid images")
    detector = MTCNN()

    for img in tqdm(os.listdir(train_dir)):
        try:
            box = detect_face(detector, train_dir+img)
            box = cut_negative_boundary(box)
            im = plt.imread(train_dir+img)
            cropped = crop_img(im, *box['box'])
            plt.imsave(train_processed_dir+img, cropped)
        except:
            print(img)
            continue
    