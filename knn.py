Python 3.11.4 (tags/v3.11.4:d2340ef, Jun  7 2023, 05:45:37) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Importing necessary libraries
... import numpy as np
... import matplotlib.pyplot as plt
... from sklearn.datasets import load_iris
... from sklearn.model_selection import train_test_split
... from sklearn.neighbors import KNeighborsClassifier
... from sklearn.metrics import accuracy_score
... 
... # Load the Iris dataset
... iris = load_iris()
... X = iris.data  # Features
... y = iris.target  # Target variable (species)
... 
... # Split data into training and test sets
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
... 
... # Initialize the KNN classifier
... k = 3  # Number of neighbors to consider
... knn = KNeighborsClassifier(n_neighbors=k)
... 
... # Train the classifier
... knn.fit(X_train, y_train)
... 
... # Predictions on the test data
... y_pred = knn.predict(X_test)
... 
... # Calculate accuracy
... accuracy = accuracy_score(y_test, y_pred)
... print(f"Accuracy: {accuracy:.2f}")
... 
... # Example usage: Predicting for a new data point
... # Suppose we have a new data point with the following features
... new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example new data point
predicted_class = knn.predict(new_data)
predicted_species = iris.target_names[predicted_class][0]
