# SRIP IIT Gandhinagar - Problem Statement 1
# Data Preparation and Deep Learning for Respiration Signal Analysis

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Step 1: Data Preparation and Preprocessing

# Load the respiration signal dataset
data = pd.read_csv('respiration_data.csv')

# Handle missing values using forward and backward fill
data = data.fillna(method='ffill').fillna(method='bfill')

# Normalize the respiration signal
data['normalized_signal'] = (data['respiration_signal'] - data['respiration_signal'].mean()) / data['respiration_signal'].std()

# Define dynamic segmentation parameters
data_length = min(100, len(data) // 10)  # Adjust segment length dynamically
segments, labels = [], []

# Create segments and corresponding labels
for i in range(0, len(data) - data_length, data_length):
    segment = data['normalized_signal'].values[i:i + data_length]
    segments.append(segment)
    labels.append(data['target'].values[i + data_length - 1])  # Assuming 'target' column holds labels

# Convert to NumPy arrays
X = np.array(segments)
y = np.array(labels)

# Reshape for CNN-LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 2: Deep Learning Model for Sequential Data

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a hybrid CNN-LSTM model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(data_length, 1)),
    BatchNormalization(),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Step 3: Model Evaluation

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Step 4: Interpretation and Analysis
# The model's predictions will be analyzed to identify apnea events and compare them with actual events.

# Conclusion
# By utilizing a CNN-LSTM hybrid approach, we enhance apnea detection capabilities using respiration signal data.
# This method improves sequential pattern recognition and ensures adaptability for real-world medical diagnostics.
