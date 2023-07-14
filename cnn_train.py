# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:59:07 2023

@author: bhumi
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load your preprocessed features and labels
features = np.load('features.npy')
labels = np.load('path/to/labels.npy')

# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the input shape based on image dimensions
image_height = 512
image_width = 512

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 32
model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_features, test_labels))

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(test_features, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
