import numpy as np
import cv2
from skimage.feature import hog
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model

breast_dat = np.load('breastmnist_224.npz')
breast_train = breast_dat['train_images']
breast_val = breast_dat['val_immages']
breast_test = breast_dat['test_images']

def preprocess_pixels(images):

    num_samples = images.shape[0]
    flattened_images = images.reshape(num_samples, -1) 
    normalized_images = flattened_images / 255.0 
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
        resized_img = cv2.resize(img, (224, 224))
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

    # joblib.dump(knn, 'knn_model.pkl')

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


def inference_binary_classification(image):
    knn_loaded = joblib.load('knn_model.pkl')

    img_path = breast_train['test_images'] [0]

    img_path = cv2.resize(img_path, (224, 224))

    img_resized = img_path

    hog_feature = extract_hog_features([img_resized])  # A single image

    X = preprocess_pixels(hog_feature)

    prediction = knn_loaded.predict(X)

    print(f"Predicted class: {prediction}")

    print("Actual label: ", breast_train['test_labels'][0])



def train_multi_classification(images):
    x = blood['test_images']
    y = blood['test_labels']

    X_train, X_eval, y_train, y_eval = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

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
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_eval, y_eval),
        epochs=10, 
        callbacks=callbacks
    )

    # Evaluate on the evaluation subset
    loss, acc = model.evaluate(X_eval, y_eval)
    print(f"Evaluation Accuracy: {acc * 100:.2f}%")


def load_and_preprocess_image(image):
    image = cv2.resize(image, (224, 224))

    image = np.array(image, dtype=np.float32)

    image = np.expand_dims(image, axis=0)

    return image


def inference_multi_classification(image):
    model = load_model('trained_model.h5')

    image_path = blood['test_images'][n]

    # Preprocess the image
    input_image = load_and_preprocess_image(image_path)

    # Make a prediction
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions)
    print(f"Predicted class: {predicted_class}")
    print(f"Actual label: {blood['test_labels'][n]}")
