import cv2
import numpy as np
import os
from sklearn.svm import SVC

# Directory path where the cleaned dataset is located
dataset_directory = 'D:\internships\ds\Project5_Ag_Crop and weed detection\output'

# Lists to store the loaded images and corresponding labels
train_images = []
train_labels = []
test_images = []
test_labels = []

# Function to extract features from an image
def extract_features(image):
    # Perform feature extraction on the image
    # Replace this with your actual feature extraction code
    features = cv2.resize(image, (32, 32)).flatten()  # Example: resizing and flattening the image
    
    return features

# Loop through the images in the dataset directory and extract features
for filename in os.listdir(dataset_directory):
    image_path = os.path.join(dataset_directory, filename)
    image = cv2.imread(image_path)
    features = extract_features(image)
    
    # Determine whether to add to training or testing set
    if np.random.rand() < 0.8:
        train_images.append(features)
        train_labels.append(0)  # Assuming label 0 represents crop (modify as needed)
    else:
        test_images.append(features)
        test_labels.append(1)  # Assuming label 1 represents weed (modify as needed)

# Convert the feature lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
print("Number of training samples:", len(train_images))
print("Number of testing samples:", len(test_images))

# Train the model
model = SVC()  # Create a Support Vector Classifier (SVC) model
model.fit(train_images, train_labels)  # Train the model using the training data

# Evaluate the model
accuracy = model.score(test_images, test_labels)  # Calculate the accuracy of the model on the test data
print("Accuracy:", accuracy)

# Make predictions
predictions = model.predict(test_images)  # Use the trained model to make predictions on the test data

# Further analysis or visualization
# You can calculate additional evaluation metrics (e.g., precision, recall, F1-score) or visualize the prediction
