import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
from scipy.special import expit
import itertools
from sklearn import svm, datasets

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# Extract images and labels
images = dataset.imgs  # Numpy array of images
labels = dataset.labels  # Corresponding labels

# Flatten the image data (optional, depending on your use case)
# If you want each pixel as a column, flatten the images.
# Otherwise, keep them as arrays.
flattened_images = images.reshape(images.shape[0], -1)

# Create a dictionary to hold the data
data_dict = {
    'image': list(flattened_images),  # or images if not flattening
    'label': labels.flatten()  # Ensure labels are a 1D array
}

# Convert the dictionary to a Pandas DataFrame
df = pd.DataFrame(data_dict)

# Assuming df is your DataFrame where each row contains a flattened image and label.
# Convert 'image' column to numpy array if not already done
x = np.array(df['image'].tolist())  # Convert list of pixels to numpy array
y = np.array(df['label'])  # Labels are already in an appropriate format

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import yeojohnson
from sklearn.preprocessing import StandardScaler

# from sklearn import svm

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=3) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=0.71)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


print(y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))