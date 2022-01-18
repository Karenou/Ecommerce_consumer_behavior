import torch
from torch.utils.data import Dataset, DataLoader 
import torchvision
import cv2

import argparse
import numpy as np
import pandas as pd
import time
import glob


# dataset class to load images
def make_dataset(data_path):
    img_paths = glob.glob(data_path + "/*.jpg")
    samples = []
    for img_path in img_paths:
        img_id = int(img_path.split("/")[-1].split('.')[0])
        samples.append((img_id, img_path))
    return samples

class ImageDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
            )
        self.images = make_dataset(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_id, img_path = self.images[index]

        img = cv2.imread(img_path)
        # resize to (C, H, W)
        img = cv2.resize(img, (img.shape[1], img.shape[0]))
        # resize to the same shape to load the data by batch
        img = cv2.resize(img, (1200, 1200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img_id, img



class HumanDetection:

    def __init__(self, opt):
        print("load the faster r-cnn model")
        self.opt = opt
        self.model = self.load_model()
        dataset = ImageDataset(self.opt.data_path)
        self.n_images = len(dataset)
        print("There are %d images" % self.n_images)
        self.data_loader = DataLoader(
            dataset, 
            batch_size=self.opt.batch_size, 
            num_workers=self.opt.num_workers, shuffle=False, drop_last=False
        )

    def run(self):
        """
        run the human detection model
        """

        start = time.time()
        print("start classification by batch")
        image_ids, pred_label, batch_no = None, None, 0
        
        for _ids, imgs in self.data_loader:
            print("Batch %d" % batch_no)
            batch_pred_label = self.predict(
                imgs, 
                conf_thres=self.opt.conf_thres, 
                pct_area_thres=self.opt.pct_area_thres
            )

            if image_ids is None or pred_label is None:
                image_ids, pred_label = _ids.numpy(), batch_pred_label
            else:
                image_ids = np.concatenate([image_ids, _ids])
                pred_label = np.concatenate([pred_label, batch_pred_label])

            batch_no += 1
        
        pred_label = pred_label[:self.n_images]
        print("save classification result to output path")
        result = pd.DataFrame(
            np.concatenate([image_ids.reshape(-1,1), pred_label.reshape(-1,1)], axis=1),  
            columns=["image_id", "pred_label"]
        )
        result = result.sort_values("image_id")
        result.to_csv(self.opt.output_path, index=False)

        end = time.time()
        print("Total time used: %.2f sec" % (end-start))

    def load_model(self, pretrained=True, model_save_path=None):
        """
        load pretrain Faster R-CNN model
        @param pretrained: whether the model is pretrained or not
        @param model_save_path: path where the model is saved
        return: downloaded model
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=False)
        if model_save_path:
            model.load_state_dict(torch.load(model_save_path))
        model.eval()
        return model

    def predict(self, imgs, conf_thres=0.7, pct_area_thres=0.1) -> np.array:
        """
        predict the image contains human 
        when the confidence of bounding box that classifies 'person' > conf_threshold 
        and the percentage of area of bounding box is > 5% of the image shape
        @param imgs: batch of images
        @param conf_thres: confidence threshold to filter bounding boxes
        return the batch predicted label as numpy array
        """
        pred = self.model(imgs)
        
        boxes = [pred[i]["boxes"].detach().numpy() for i in range(len(imgs))]
        classes = [pred[i]["labels"].numpy() for i in range(len(imgs))]
        scores = [pred[i]["scores"].detach().numpy() for i in range(len(imgs))]
        
        batch_pred_label = np.zeros(self.opt.batch_size)
        
        for i in range(len(imgs)):
            for j in range(len(boxes[i])):
                # filter class = 1 and confidence of the bbox > threshold
                if classes[i][j] == 1 and scores[i][j] > conf_thres:
                    area = (boxes[i][j][3] - boxes[i][j][1]) * (boxes[i][j][2] - boxes[i][j][0])
                    pct_area = area / (imgs[i].shape[1] * imgs[i].shape[2])
                    if pct_area > pct_area_thres:
                        batch_pred_label[i] = 1

        return batch_pred_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--data_path", type=str, help="Path to save the images")
    parser.add_argument("--output_path", type=str, help="Path to save the classification result")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="Confidence threshold to filter bounding box")
    parser.add_argument("--pct_area_thres", type=float, default=0.1, help="Threshold to filter size of bounding box")
    opt = parser.parse_args()

    model = HumanDetection(opt)
    model.run()
