# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data  # Features
y = cancer.target  # Target variable (0: malignant, 1: benign)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression classifier
logreg = LogisticRegression(random_state=42)

# Train the classifier
logreg.fit(X_train, y_train)

# Predictions on the test data
y_pred = logreg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

disp = plot_confusion_matrix(logreg, X_test, y_test, display_labels=cancer.target_names, cmap=plt.cm.Blues, normalize=None)
disp.ax_.set_title('Confusion Matrix')
plt.show()

# Example usage: Predicting for a new data point
# Suppose we have a new data point with the following features
new_data = np.array([[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])  # Example new data point
predicted_class = logreg.predict(new_data)
predicted_diagnosis = cancer.target_names[predicted_class][0]
print(f"Predicted class: {predicted_class} (Diagnosis: {predicted_diagnosis})")
