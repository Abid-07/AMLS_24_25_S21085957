from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
import cv2
from skimage.feature import hog
import joblib
import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model

breast = np.load('./Datasets/breastmnist_224.npz')
blood = np.load('./Datasets/bloodmnist_224.npz')

pca = joblib.load('./A/pca_model2.pkl')
svm = joblib.load('./A/svm_model.pkl')

model = load_model('./B/trained_model.h5')


def preprocess_pixels(images):
    #shape[0] is the number of samples
    num_samples = images.shape[0]
    #flatten to 1D array of pixel intensities
    flattened_images = images.reshape(num_samples, -1) 
    #normalise the flattened images
    normalized_images = flattened_images / 255.0 
    return normalized_images

def extract_hog_features(images):
    #new features array
    features = []
    #extract histogram of oriented gradients from each image
    for img in images:
        feature = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(feature)
    return np.array(features)

def train_binary_classification(images, labels):
    #images are selected from BreastMNIST training images
    images = breast['train_images']     

    #images are resized to 128x128 from 224x224 (or whichever size you choose to download)
    resized_images = []                 

    #opencv as a simple tool to resize images appended to new resized_images array
    for img in images:
        resized_img = cv2.resize(img, (128, 128))
        resized_images.append(resized_img)

    resized_images = np.array(resized_images)

    #hog features extracted from each image
    hog_features = extract_hog_features(resized_images)

    #hog features are preprocessed
    X = preprocess_pixels(hog_features)
    y = breast['train_labels']

    #use adasyn to resample the minority class to match the number of samples in the majority class
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    #split the training data into a training and testing set based on preprocessed data into 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    #principle components analysis to extract 100 components that capture the most variance to simplify training
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    #store the pca model if necessary
    # joblib.dump(pca, 'pca_model2.pkl')

    #create a polynomial kernel based SVM with the following hyperparameters
    svm = SVC(kernel='poly', degree=3, gamma='scale', coef0=2, random_state=42)

    #train the SVM model on the training set
    svm.fit(X_train_pca, y_train)

    #store the SVM model
    # joblib.dump(svm, 'svm_model.pkl')

    #use the svm model to predict on the assigned test set
    y_pred = svm.predict(X_test_pca)


def preprocess_image(image_path):
    resized_img = cv2.resize(image_path, (128, 128))
    
    # Extract HOG features
    feature = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # Return the HOG features as a flattened array
    return feature.reshape(1, -1)


def predict_image(image_path):
    # Preprocess the image to extract features
    features = preprocess_image(image_path)
    
    # Ensure the PCA transformation aligns with what the SVM expects
    features_pca = pca.transform(features)  # This should reduce features to 100
    
    # Predict the class using the loaded SVM model
    prediction = svm.predict(features_pca)
    
    return prediction[0]


def inference_binary_classification(min, max):
    count = 0
    total = max-min

    for n in range (min, max):
        image_path = breast['test_images'][n]  # Provide the path to the image
        predicted_class = predict_image(image_path)
        print(f"Predicted class: {predicted_class}")
        print(f"Actual class: {breast['test_labels'][n]}")
        if breast['test_labels'][n] == predicted_class:
            count+=1

    print(f"Accuracy: {count/total*100}")



def train_multi_classification():
    x = blood['test_images']
    y = blood['test_labels']

    X_train, X_eval, y_train, y_eval = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    # One-hot encoding of labels
    y_train = to_categorical(y_train, num_classes=8)
    y_eval = to_categorical(y_eval, num_classes=8)

    # Define ResNet model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-4]: 
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x)
    predictions = Dense(8, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        validation_data=(X_eval, y_eval),
        epochs=2, 
        callbacks=callbacks
    )

    # Evaluate on the evaluation subset
    # loss, acc = model.evaluate(X_eval, y_eval)
    # print(f"Evaluation Accuracy: {acc * 100:.2f}%")


def load_and_preprocess_image(image):
    image = cv2.resize(image, (224, 224))

    image = np.array(image, dtype=np.float32)

    image = np.expand_dims(image, axis=0)

    return image


def inference_multi_classification(min, max):
    count = 0
    total = max-min

    for n in range(min,max):

    # Example image path
        image_path = blood['test_images'][n]

        # Preprocess the image
        input_image = load_and_preprocess_image(image_path)

        # Make a prediction
        predictions = model.predict(input_image)
        predicted_class = np.argmax(predictions)
        print(f"Predicted class: {predicted_class}")
        print(f"Actual label: {blood['test_labels'][n]}")
        if (predicted_class == blood['test_labels'][n]): 
            count+=1

    print(f"Accuracy = {count/total*100}%")


def main():
    inference_binary_classification(0,30)
    # inference_multi_classification(0,3)

if __name__ == "__main__":
    main()