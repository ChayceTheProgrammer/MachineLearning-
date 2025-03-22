#!/usr/bin/env python

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load the iris dataset
    iris = load_iris()
    X = iris.data  # Feature matrix
    y = iris.target  # Target labels

    # Split the dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create the decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Train (fit) the model on the training set
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model accuracy on the test set
    accuracy = np.mean(y_pred == y_test)
    print("Test Set Accuracy: {:.2f}%".format(accuracy * 100))

    # Display the decision tree rules in a readable text format
    tree_rules = export_text(clf, feature_names=iris.feature_names)
    print("\nDecision Tree Rules:\n")
    print(tree_rules)

    # Visualize the decision tree graphically
    plt.figure(figsize=(16, 10))
    plot_tree(
        clf,
        filled=True,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        rounded=True
    )
    plt.title("Decision Tree Visualization")
    plt.show()


if __name__ == '__main__':
    main()