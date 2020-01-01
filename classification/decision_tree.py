# Import dependencies
import numpy as np
import matplotlib.pyplot as plt

from classification.evaluator import ClassificationEvaluator

class Node:
    """ Represents a single node in a decision tree classifier """
    def __init__(self, predicted_class):
        """
        Initialises a new node in the decision tree.
        Each node contains the following:
            - the predicted class
            - the index of the feature it is associated with
            - the threshold below which the left subtree should be traversed, otherwise traverse the right subtree
            - a pointer to the left node
            - a pointer to the right node

        Parameters:
            predicted_class (int/str): the predicted class for this node.
        """
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree:
    """ An implementation of the CART algorithm to train a decision tree classifier """
    def __init__(self, max_depth=1):
        """
        Constructor to initialise the maximum depth of the decision tree.

        Parameters:
            max_depth (int): the maximum depth of this decision tree.
        """
        self._max_depth = max_depth

    def fit(self, data, labels):
        """
        Fits the decision tree classifier to the data and corresponding labels.

        Parameters:
            data (numpy.ndarray): a numpy array of data items.
            labels (numpy.ndarray): a numpy array of labels corresponding to items in data.
        """
        self._num_classes = len(np.unique(labels))
        self._num_features = data.shape[1]

        # Begin the recursive calls to grow the tree
        self._tree = self._grow_tree(data, labels)

    def _best_split(self, data, labels):
        """
        Computes the best split of the input data and labels to separate into branches of the tree.

        Parameters:
            data (numpy.ndarray): a numpy array of data items.
            labels (numpy.ndarray): a numpy array of labels corresponding to items in data.
        """
        num_labels = len(labels)
        if num_labels <= 1:
            return None, None

        # Find total number of classes for the current set of labels
        num_parent = [np.sum(labels == c) for c in range(self._num_classes)]

        # Compute gini index for all items in parent
        best_gini = 1 - sum((n / num_labels) ** 2 for n in num_parent)

        best_index, best_threshold = None, None

        # Iterate over each feature in data
        for index in range(self._num_features):

            # Sort current feature
            thresholds, classes = zip(*sorted(zip(data[:, index], labels)))
            num_left = [0] * self._num_classes
            num_right = num_parent.copy()

            # Loop to compute total number of classes belonging to the left/right
            for i in range(1, num_labels):
                c = classes[i - 1]

                num_left[c] += 1
                num_right[c] -= 1

                # Compute gini index for left subtree
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self._num_classes))

                # Compute gini index for right subtree
                gini_right = 1.0 - sum((num_right[x] / (num_labels - i)) ** 2 for x in range(self._num_classes))

                # Compute total gini index
                gini = (i * gini_left + (num_labels - i) * gini_right) / num_labels

                if thresholds[i] == thresholds[i - 1]:
                    continue

                # Update current best gini if calculated gini is smaller than current best
                if gini < best_gini:
                    best_gini = gini
                    best_index = index
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2

        return best_index, best_threshold

    def _grow_tree(self, data, labels, depth=0):
        """
        Grows the decision tree recursively up to the specified maximum depth.

        Parameters:
            data (numpy.ndarray): a numpy array of data items.
            labels (numpy.ndarray): a numpy array of class labels corresponding to items in data.
            depth (int): the depth of the tree on the current recursive call.

        Returns:
            (Node): the parent node of the tree.
        """
        # Find total number of items per class
        num_per_class = [np.sum(labels == i) for i in range(self._num_classes)]

        # Take maximum occuring class as predicted class and store in new Node
        predicted_class = np.argmax(num_per_class)
        node = Node(predicted_class=predicted_class)

        if depth < self._max_depth:
            # Find best split for current data and labels during recursive calls
            i, threshold = self._best_split(data, labels)

            if i != None:
                # Find all indices less than threshold of the calculated best split
                left_indices = data[:, i] < threshold

                # Obtain data to left of threshold (smaller values) and all data to right of threshold (greater values)
                data_left, labels_left = data[left_indices], labels[left_indices]
                data_right, labels_right = data[~left_indices], labels[~left_indices]

                # Set Node properties
                node.feature_index = i
                node.threshold = threshold

                # Recursively call _grow_tree to grow the left side of the tree until maximum depth is reached
                node.left = self._grow_tree(data_left, labels_left, depth + 1)

                # Recursively call _grow_tree to grow the right side of the tree until maximum depth is reached
                node.right = self._grow_tree(data_right, labels_right, depth + 1)

        return node

    def predict(self, test):
        """
        Predicts class labels for all items in an array of test items.

        Parameters:
            test (numpy.ndarray): a numpy array of test items.

        Returns:
            (list): a python list of predicted class labels.
        """
        return np.array([self._predict(item) for item in test])

    def _predict(self, item):
        """
        Computes a class label prediction for an individual test item.

        Parameter:
            item (numpy.ndarray): a numpy array representing the test item.

        Returns:
            (str/int): the predicted class label for this test item.
        """
        node = self._tree

        # Traverse down the tree and find the appropriate node position this item belongs to
        while node.left:
            if item[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class

    def test(self, test_data, labels, summary=False):
        """
        Calculates the accuracy of the decision tree classifier based on a set of test data and true labels.

        Parameters:
            test_data (numpy.ndarray): a numpy array of test data items.
            labels (numpy.ndarray): a numpy array of labels corresponding to test items in test_data.

        Returns:
            (float): the accuracy of this classifier as a percentage of the total correctly predicted class labels.
        """
        preds = self.predict(test_data)

        metrics = ClassificationEvaluator(pred_labels=preds, actual_labels=labels).evaluate()

        if summary:
            print(f"Accuracy: {metrics['accuracy'] * 100}%")
            print(f"Precision: {metrics['precision'] * 100}%")
            print(f"Recall: {metrics['recall'] * 100}%")
            print(f"F1-score: {metrics['f1']}")

        return metrics, preds
