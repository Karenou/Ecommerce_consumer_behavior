## BMI Prediction on Human Body Images
- Reference
    - The BMI Prediction refers to this [Github repo](https://github.com/atoms18/BMI-prediction-from-Human-Photograph)
    - The Self-Correction-Human-Parsing is cloned from this [Github repo](https://github.com/PeikeLi/Self-Correction-Human-Parsing). We modified the `simple_extractor.py` and `datasets/simple_extractor_dataset.py` accordingly for production purpose.

- Environment
    - cuda=11.3, cudnn=8.2.0

- Required packages
    - pytorch=1.10, 
    - ninja=1.10.2
        - installation
        The model is deployed on cluster with GeForce RTX 3090 (sm_86) with nvidia driver version=470. Need to install ninja first, and lower the cuda architecture list to "7.5"
        ```
        pip install ninja
        export TORCH_CUDA_ARCH_LIST="7.5"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64
        ```
    - tensorflow=2.5.0
    - cv2

- How to Run the Program

1. First get the processed images using `human_parsing.ipynb`. Or input the following in terminal.

```
python Self-Correction-Human-Parsing/simple_extractor.py --dataset 'atr' --model-restore='Self-Correction-Human-Parsing/checkpoints/atr.pth' --input-dir='../../HKTVMall_data' --output-dir='process_images' --input_csv='../human_detection/result/frcnn_result.csv' --gpu=0 
```

2. Then start bmi prediction using the `bmi_prediction.ipynb`

The production code is written in `bmi_prediction.py`. To run the program, run the following command in terminal.

```
python bmi_prediction.py --input_csv="../human_detection/result/frcnn_result.csv" --image_path="../../HKTVMall_data" --process_image_path="process_images" --model_path="model/4.053_model.h5"
```
