import numpy as np
import cv2
from skimage.feature import hog
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

breast_dat = np.load('breastmnist_224.npz')
breast_train = breast_dat['train_images']
breast_val = breast_dat['val_immages']
breast_test = breast_dat['test_images']

def preprocess_pixels(images):
    # If grayscale, images.shape = (num_samples, H, W)
    # If RGB, images.shape = (num_samples, H, W, 3)
    num_samples = images.shape[0]
    flattened_images = images.reshape(num_samples, -1)  # Flatten
    normalized_images = flattened_images / 255.0  # Normalize pixel values
    return normalized_images

def extract_hog_features(images):
    features = []
    for img in images:
        feature = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(feature)
    return np.array(features)

def train_binary_classification(images, labels):
    resized_images = []

    for img in images:
        resized_img = cv2.resize(img, (128, 128))
        resized_images.append(resized_img)

    resized_images = np.array(resized_images)

    hog_features = extract_hog_features(resized_images)

    X = preprocess_pixels(hog_features)
    y = labels

    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')

    knn.fit(X_train, y_train)

    joblib.dump(knn, 'knn_model.pkl')

    joblib.dump(knn, 'knn_model.pkl')

    # Define k-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5 folds

    # Perform cross-validation
    scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')

    # Display results
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())

    # Predict and evaluate
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

