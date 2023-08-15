# Disaster Images Classification

![Images before and after PCA](https://github.com/JohnKond/DisasterClassification/blob/main/images_before_after_pca.png?raw=true)

## Abstract

The objective of the project is to develop an efficient classification model that can accurately identify different types of disasters based on visual information. Three different techniques were explored and evaluated to address this task: 
- Principal Component Analysis (PCA) combined with tabular models (Random Forest and SVM)
- Convolutional Neural Network (CNN)
- Transfer Learning using a pretrained MobileV2-Net model.
The experiments and evaluations were performed on a dataset consisting of images from various disaster categories. The results demonstrated that Transfer Learning using the MobileV2-Net model yielded the highest accuracy, showcasing its effectiveness in image classification tasks with limited data.

For more information, please see the project report.

## Project Structure

The project is organized as follows:

- `data/`: This directory contains the dataset used for training and evaluation. It includes images of various disaster categories, such as earthquakes, wildfires, floods, and cyclones.
- `nn_preprocessed_data/`: This directory contains preprocessed data, used for neural network tasks.
- `preprocessed_data/`: This directory contains preprocessed data, used for PCA tasks.
- `src/`: This directory contains the source code for the project. It includes scripts for model creation, data preprocessing, and evaluation.
- `requirements.txt`: This file lists all the required Python packages and their versions needed to run the project.

## Installation

To run the code, follow these steps:

1. Install the required packages by running the command :  pip install -r /path/to/requirements.txt

2. Navigate to the `src/` directory in your terminal or command prompt.

3. Run the `main.py` script by executing the following command: ```python main.py```
