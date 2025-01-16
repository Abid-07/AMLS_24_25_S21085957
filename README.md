This repository contains 2 machine learning models for binary and multi classification tasks. The first task A is a binary classification task using the dataset BreastMNIST by MedMNIST.
The binary classifiier uses an SVM approach to distinguish between Malignant (0) and Benign (1) tumours from the Breast Ultrasound images provided.
The second task B uses a multiple classifier approach uses the BloodMNIST dataset from medMNIST. A CNN approach is chosen to distinguish between the 8 classes of blood cell types.
Both models have individual trained models within their associate folders A and B, and training, inference and evaluation can be conducted from main.py, simply by entering into the your command
prompt: python main.py

The recommended terminal is MiniForge, which was used for development for this project.

The necessary packages include:
- sklearn
- imblearn
- cv2
- skimage
- joblib (optional if you want to save models after training)
- numpy
- tensorflow

There are in total 3 folders: A, B and Datasets. Populate Dataset with the necessary datasets from medMNIST, specifically breastMNIST and bloodMNIST. Ensure they are in .npz format when you save them.

Folders A and B contain the necessary trained models to conduct inference for tasks A and B. Folder A contains files svm_model.pkl and pca_model2.pkl. These models are used in main.py for inference on task A.

Similarly for B, the content include 'trained_model.h5' used for inference for task B, multi-classification.

The main.py file is constructed such that the user may choose which task to run between:
- train_binary_classifier
- inference_binary_classifier
- train_multi_classification
- inference_multi_classification

To choose which tasks to run, simply comment and uncomment from the main function inside main.py towards the bottom of the file.

The training functions will conduct the relevant training for the classifier and then provide initial test accuracy and cross validation accuracy for the binary classifier for task A and a simple evaluation accuracy for task B.

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


# Machine Learning Models for Binary and Multi-class Classification

This repository contains two machine learning models for binary and multi-class classification tasks.

## Task A: Binary Classification
The binary classification task uses the **BreastMNIST** dataset from **MedMNIST**. An SVM approach is employed to distinguish between **Malignant (0)** and **Benign (1)** tumors using the provided breast ultrasound images.

## Task B: Multi-class Classification
The multi-class classification task uses the **BloodMNIST** dataset from **MedMNIST**. A CNN approach is chosen to classify the images into eight different blood cell types.

Both models have pre-trained models located in their respective folders (A and B). You can perform training, inference, and evaluation using `main.py` by running the following command in your terminal:

```bash
python main.py

### Recommended Terminal
**MiniForge** (used during the development of this project).

### Required Packages
- `sklearn`
- `imblearn`
- `cv2`
- `skimage`
- `joblib` (optional, for saving models after training)
- `numpy`
- `tensorflow`

### Directory Structure
- **A**: Contains files for Task A, including the trained models `svm_model.pkl` and `pca_model2.pkl`.
- **B**: Contains files for Task B, including the trained model `trained_model.h5`.
- **Datasets**: Populate this folder with the necessary datasets from MedMNIST (BreastMNIST and BloodMNIST). Ensure they are saved in `.npz` format.

### Usage

#### Available Functions in `main.py`

**Task A (Binary Classification):**
- `train_binary_classification()`
- `inference_binary_classification()`

**Task B (Multi-class Classification):**
- `train_multi_classification()`
- `inference_multi_classification()`

To run a specific task, comment or uncomment the desired function calls in the `main()` function located at the bottom of `main.py`.

### Training and Evaluation

**Task A:**
- Training will output initial test accuracy and cross-validation accuracy for the binary classifier.

**Task B:**
- Training will output the evaluation accuracy for the multi-class classifier.

### Inference

You can test the models on unseen data by running the inference functions. To receive real-time feedback, uncomment the print lines in the respective inference functions.

**Task A (Binary Classification) Inference:**
```python
print(f"Predicted class: {predicted_class}")
print(f"Actual class: {breast['test_labels'][n]}")

**Task B (Multi-class Classification) Inference:**

```python
print(f"Predicted class: {predicted_class}")
print(f"Actual label: {blood['test_labels'][n]}")

**Adjusting Inference Range**

To run inference on a specific range of images, adjust the parameters in the `main` function in `main.py`:

```python
inference_binary_classification(minimum_value, maximum_value)
inference_multi_classification(minimum_value, maximum_value)
Please refer to the dataset sizes to determine the appropriate range of samples for inference.