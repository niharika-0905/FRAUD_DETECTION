import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check the current working directory (optional but helps with debugging)
print(f"Current Working Directory: {os.getcwd()}")

# Load the dataset (make sure 'creditcard.csv' is in the same folder as this script)
data = pd.read_csv('creditcard.csv')

# Show the first few rows of the dataset to confirm it loaded correctly
print("\nFirst few rows of the dataset:")
print(data.head())

# Separate the features (X) and the target variable (y)
X = data.drop('Class', axis=1)  # Features (everything except 'Class')
y = data['Class']  # Target column ('Class')

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model's performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()