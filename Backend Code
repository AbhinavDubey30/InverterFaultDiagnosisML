import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.decomposition import PCA

# Load your dataset (replace with actual path or data loading method)
file_path = '/content/Inverter Data Set.csv'  # Path to your CSV file
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to verify structure
print(data.head())

# Create a 'fault' column based on certain conditions (example: motor speed lower than a threshold or high voltage)
fault_condition = (data['n_k'] < 3000) | (data['u_dc_k'] > 567)  # Example: fault if motor speed < 500 or DC voltage > 600
data['fault'] = fault_condition.astype(int)  # Create binary fault column (1 for fault, 0 for normal)

# Preprocessing the data
X = data.drop(columns=['fault'])  # Features (all columns except 'fault')
y = data['fault']  # Target (the newly created fault column)

# Handling missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Feature scaling (standardizing the data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 1: Deep Feature Extraction using a Neural Network
def create_feature_extractor(input_shape):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    # We don't output to a final class here; we use this as feature extraction
    return model

# Build and compile the feature extractor
input_shape = X_train_scaled.shape[1]
feature_extractor = create_feature_extractor(input_shape)

# Extract high-level features from the training data
X_train_deep_features = feature_extractor.predict(X_train_scaled)
X_test_deep_features = feature_extractor.predict(X_test_scaled)

# Optionally reduce dimensionality using PCA (to create more compressed features)
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_deep_features)
X_test_pca = pca.transform(X_test_deep_features)

# Step 2: Ensemble Learning (RandomForest) with Extracted Deep Features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)

# Predicting the test set results
y_pred = rf.predict(X_test_pca)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Output the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
