# Image Segmentation with U-Net

This project demonstrates image segmentation using the U-Net architecture for medical image analysis. The dataset used is the Data Science Bowl 2018 dataset, which consists of images of cell nuclei.

## Dataset

The dataset used in this project is the [Data Science Bowl 2018: Nuclei Detection](https://www.kaggle.com/competitions/data-science-bowl-2018/data) dataset from Kaggle. It consists of various images of cell nuclei and their corresponding masks. Each image has one or more masks indicating the location of nuclei within the image. 

To use the dataset:
1. **Create a Kaggle Account**: If you don't have a Kaggle account, you will need to create one.
2. **Accept the Competition Rules**: Before downloading the dataset, you will need to accept the competition rules.
3. **Download the Dataset**: Once you have accepted the rules, you can download the dataset by clicking the "Download All" button on the data page.

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/rakeshutekar/Image-Segmentation-with-U-Netification.git
   cd image-segmentation
2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt

### Running the Project

Training the Model
Navigate to the src directory and run the training script:
1. **Train**
   ```
       cd src
       python train.py

2. **Evaluating the Model**
After training, you can evaluate the model:
python evaluate.py

### Results:
```
5/5 ━━━━━━━━━━━━━━━━━━━━ 2s 449ms/step - accuracy: 0.9478 - loss: 0.0930
Test Loss: 0.09798252582550049
Test Accuracy: 0.9455899000167847
5/5 ━━━━━━━━━━━━━━━━━━━━ 3s 514ms/step
Mean IoU: 0.7401531363366687
![output](https://github.com/rakeshutekar/Image-Segmentation-with-U-Net/assets/48244158/b18004a2-b77d-432c-ac15-ae1d181eebb7)

