## BMI Prediction on Human Body Images
- Reference
    - The BMI Prediction refers to this [Github repo](https://github.com/atoms18/BMI-prediction-from-Human-Photograph)
    - The Self-Correction-Human-Parsing is cloned from this [Github repo](https://github.com/PeikeLi/Self-Correction-Human-Parsing). We modified the `simple_extractor.py` and `datasets/simple_extractor_dataset.py` accordingly for production purpose.

- Required packages
    - pytorch1.10
    - tensorflow2.6.0, keras
    - cv2

- How to Run the Program

1. First get the processed images using `human_parsing.ipynb`

2. Then start bmi prediction using the `bmi_prediction.ipynb`

The production code is written in `bmi_prediction.py`. To run the program, run the following command in terminal.

```
python bmi_prediction.py --input_csv="../human_detection/result/frcnn_result.csv" --image_path="../../HKTVMall_data" --process_image_path="process_images" --model_path="model/4.053_model.h5"
```
