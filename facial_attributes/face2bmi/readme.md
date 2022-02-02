# README

- Reference 

This face2bmi is based on this (repo)[https://github.com/6chaoran/face2bmi] and the (Face-to-BMI paper)[https://arxiv.org/pdf/1703.03156.pdf]

- Requirement
    - vggface
    - mtcnn==0.1.0
    - tensorflow==2.5.0, keras==2.5.0
    - sklearn
    - opencv==4.4.0
    - Pillow=7.2.0
    - pandas, tqdm, numpy

- Model architecture

Using the face images extracted in the `FairFace/detect_faces` folder, it applies VGGFace, which uses vgg16 or resnet50 as backbone, to predict the bmi, age and gender from the face images.

- How to run the program
    - Training phase
    ```
    python main.py --mode="train" --model_type="vgg16" --epochs=3 --batch_size=8 
    ```
    - Testing phase
    python main.py --mode="test" --model_type="vgg16" --epochs=3 --batch_size=8 --image_path="../FairFace/detect_faces"