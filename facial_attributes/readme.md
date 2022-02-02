# README

## FairFace
- Reference:

This repo is cloned from [**FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation**](https://github.com/dchen236/FairFace). We modified the code slightly for actual deployment and production purpose.

- Usage

Use to extract facial attributes including age, gender and race. But after applying the pretrained model on own dataset, we found that the age prediction is incorrect. So we use the face detection and analytics api service provided by Face++

## Face ++

- Reference

This repo is cloned from [**FacePlusPlus Python SDK**](https://github.com/FacePlusPlus/facepp-python-sdk). We modified the `call.py` and `facepp.py` accordingly for actual deployment and production purpose.

- Usage

We extract the facial attributes including gender, age, female_score and male_score with the help of Face++ API.

- How to run the SDK

1. First need to input the api_key and api_secret in `facecpp.py`. Since we sign up a free account in the EN website, also need to change the server url in `facecpp.py` to `'https://api-us.faceplusplus.com'`

2. In `call.py`, we upload images from local folder and save the returned resul in csv format. Run the following command.

```
python3 facepp-python-sdk-master/call.py
```

## Face2BMI

- Reference 

This face2bmi is based on this (repo)[https://github.com/6chaoran/face2bmi] and the (Face-to-BMI paper)[https://arxiv.org/pdf/1703.03156.pdf]

- Usage

We predict the bmi, age, gender based on only face images extracted by **FairFace**.

- How to run the program
    - Training phase
    ```
    python main.py --mode="train" --model_type="vgg16" --epochs=3 --batch_size=8 
    ```
    - Testing phase
    ```
    python main.py --mode="test" --model_type="vgg16" --epochs=3 --batch_size=8 --image_path="../FairFace/detect_faces"
    ```