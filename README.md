## Overview

This repository contains 2 machine learning models for binary and multi classification tasks. The first task A is a binary classification task using the dataset BreastMNIST by MedMNIST.
The binary classifiier uses an SVM approach to distinguish between Malignant (0) and Benign (1) tumours from the Breast Ultrasound images provided.
The second task B uses a multiple classifier approach uses the BloodMNIST dataset from medMNIST. A CNN approach is chosen to distinguish between the 8 classes of blood cell types.
Both models have individual trained models within their associate folders A and B, and training, inference and evaluation can be conducted from main.py, simply by entering into the your command
prompt: python main.py

The recommended terminal is MiniForge Prompt, which was used for development for this project.

## Package requirements

The necessary packages include:
- sklearn
- imblearn
- cv2
- skimage
- joblib (optional if you want to save models after training)
- numpy
- tensorflow

Please ensure you have installed all the necessary packages, into a conda environment ideally running on Python 3.11.10.

## Code structure

There are in total 3 folders: A, B and Datasets. Populate Dataset with the necessary datasets from medMNIST, specifically breastMNIST and bloodMNIST. Ensure they are in .npz format when you save them. Additionally, make sure to add the file paths accordingly in main.py

Folders A and B contain the necessary trained models to conduct inference for tasks A and B. Folder A contains files svm_model.pkl and pca_model2.pkl. These models are used in main.py for inference on task A.

Similarly for B, the content include 'trained_model.h5' used for inference for task B, multi-classification.

The main.py file is constructed such that the user may choose which task to run between:
- train_binary_classifier
- inference_binary_classifier
- train_multi_classification
- inference_multi_classification

To choose which tasks to run, simply comment and uncomment from the main function inside main.py towards the bottom of the file.

The training functions will conduct the relevant training for the classifier and then provide initial test accuracy and cross validation accuracy for the binary classifier for task A and a simple evaluation accuracy for task B.

If you want to retrain the multi-classifier, ensure you have sufficient kernel memory. Tested on VSCode, Jupyter.

## Testing and Inference

For testing on unseen data, you may run the inference files, which will test on the provided unseen test sets. You may optionally choose to
receive real time feedback by uncommenting the print lines:

TaskA inference:
- print(f"Predicted class: {predicted_class}")
- print(f"Actual class: {breast['test_labels'][n]}")

TaskB inference
- print(f"Predicted class: {predicted_class}")
- print(f"Actual label: {blood['test_labels'][n]}")

At the end of the inference, an accuracy score will be provided.

The inference functions are specifically programmed to be able to run inference on single images as well as a range of images from
the unseen test set.

To alter the range simply adjust the parameters as so within the main function in main.py
- inference_binary_classification(minimum value, maximum value)
- inference_multi_classifcation(minimum value, maximum value)

Please keep note of the dataset sizes to know the range of samples that may be inferenced.