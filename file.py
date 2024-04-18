import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate random data that is not separable easily
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.5, 0.5], flip_y=0.5, class_sep=0.1, random_state=1)

# Flip the labels to make it even less coherent
flip_indices = np.random.choice([True, False], size=len(y), p=[0.3, 0.7])
y[flip_indices] = 1 - y[flip_indices]


# Train a logistic regression model on the dataset
model = LogisticRegression()
model.fit(X, y)

# Predict the labels using the same dataset
predictions = model.predict(X)

# Calculate the accuracy
accuracy = accuracy_score(y, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
