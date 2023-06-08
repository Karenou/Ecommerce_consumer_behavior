# README

## Background

This repo is part of the code implementation for a research project about Ecommerce consumer behavior, supervised by Professor June Shi, Department of Marketing, HKUST Business School. 

The project aims at investigating how the idea of body shaming subconsciously affects consumer's behavior when they are browsing or purchsing on the Ecommerce platform.

It contains the source code used in the paper **How Do Fast Fashion Copycats Affect the Popularity of Premium Brands? Evidence from Social Media**, published in [Journal of Market Research, March 2023](https://repository.hkust.edu.hk/ir/Record/1783.1-125655).

## Data Source

To identify the effect of body shaming on Ecommerce, we obtained product info, sales, text and image data, customer traffic data from [HKTV Mall](https://opendatabank.hktvmall.com/portal/home), the largest Ecommerce player in HK. The data contains about 100k SKUs and over 470k images.

## Work

### BMI Prediction

BMI, computed by weight / squared of height, is a key measure of body shape. We measured the BMI of models present in product images from two approaches: 1) use full or part of the body images; 2) use only the facial part of the human body.

- Approach One - Body Images

As the number of images is huge and only a small subset of it contains human body. To reduce computation cost, we first applied a human body detection model using [Faster R-CNN](human_detection/frcnn.py) to mark the images with human body. 

After that, we passed the marked images to a [human parsing model](bmi_prediction/Self-Correction-Human-Parsing) and cropped the human shape, which then passed into the [BMI prediction model](bmi_prediction/bmi_prediction.py) and got the final prediction.

- Approach Two - Facial Images

Using the dlib package to identify the facial region and crop the face images, we also applied the [VGGFace model](facial_attributes/face2bmi) to predict the BMI from facial images as a supplementary reference.

### Facial Attributes

Aside from body shape, facial appearance and attractiveness also catalyze body shaming.  

- Gender and Race

We used the pretrained model presented in the [FairFace](facial_attributes/FairFace) paper to classify the gender and race from facial images. 

- Age and Face Attractiveness

To predict the age and quantify the facial attractiveness, we leveraged on the [Face Detection API](https://console.faceplusplus.com/documents/5679127) offered by Face++.


### Cloth Attributes

The visual styles of cloth images may also have an impact on consumers' behavior. We used [mmfashion](https://github.com/Karenou/mmfashion), an open source visual fashion analysis toolbox, to predict the probabilities of a selective subset of 542 cloth attributes.

### Image Quality Assessment

To control the visual quality, we also computed the technical and aesthetic scores for each image, using the implementation of [NIMA: Neural Image Assessment](https://github.com/idealo/image-quality-assessment).

- install docker

- build image

```
sudo docker service start

sudo docker build -t nima-cpu . -f Dockerfile.cpu
```

- set cpu limit

```
# change in predict
BASENAME_IS=`basename $IMAGE_SOURCE`
# run predictions
DOCKER_RUN="docker run -it --cpus 5
  --entrypoint entrypoints/entrypoint.predict.cpu.sh \
  -v "$IMAGE_SOURCE":/src/$BASENAME_IS
  -v "$WEIGHTS_FILE":/src/weights.hdf5
  -v "$PREDICTIONS_FILE":/src/pred_scores.json
  $DOCKER_IMAGE $BASE_MODEL_NAME /src/weights.hdf5 /src/$BASENAME_IS /src/pred_scores.json"
```

- set image-source in run.sh, save output in txt file

```
# change image-source
./predict  \
	--docker-image nima-gpu \
	--base-model-name MobileNet \
	--weights-file $(pwd)/models/MobileNet/weights_mobilenet_technical_0.11.hdf5 \
	--image-source /home/juneshi/Desktop/Image \
	--predictions-files $(pwd)/result/technical_scores.json

# save output to txt file
./run.sh > file_path.txt
```

- convert txt to csv

change file paths and skip_rows parameter in txt_to_csv.py
