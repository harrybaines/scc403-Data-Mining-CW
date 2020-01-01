# Import dependencies
import numpy as np
import operator

from classification.evaluator import ClassificationEvaluator

class KNN:
    def _dist(self, p1, p2):
        """
        Calculates the euclidean distance between 2 points p1 and p2.

        Parameters:
            p1 (list): a python list of numerical values.
            p2 (list): a python list of numerical values.

        Returns:
            (float): the total euclidean distance between all pairs of points in p1 and p2.
        """
        total = 0

        for x, y in zip(p1, p2):
            total += (x - y) ** 2

        return total ** 0.5

    def _get_k_neighbours(self, data, test_item, k=2):
        """
        Obtains the k-nearest neighbours of a test item from a given dataset.

        Parameters:
            data (list): a list of numerical data lists.
            test_item (list): a list of numerical values.
            k (int): the number of neighbours to return from data.

        Returns:
            (list): a python list containing the k-nearest neighbours.
        """
        # Create list of tuples, each containing a data item and the distance from that item to the test item
        distances = [(item, self._dist(test_item[:-1].astype(float), item[:-1].astype(float))) for item in data]

        # Sort distances in ascending order
        distances.sort(key=operator.itemgetter(1))

        # Obtain k item values from distances list
        neighbours = [distances[i][0] for i in range(k)]
        return neighbours

    def _get_response(self, neighbours):
        """
        Predict the response based on the found neighbours
        Each neighbour votes for their class attribute and the majority vote is taken as the prediction.

        Parameters:
            neighbours (list): a python list of the k-nearest neighbours.

        Returns:
            (int/str): the voted class.
        """
        class_votes = {}

        for x in range(len(neighbours)):
            response = neighbours[x][-1]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1

        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]

    def test(self, test_labels, preds, summary=False):
        """
        Computes the accuracy of predictions on the test set.

        Parameters:
            test_labels (list): a python list of labels for items in the test set.
            preds (list): a python list of predicted class labels for each item in test.

        Returns:
            (float): the accuracy of the fitted model.
        """
        metrics = ClassificationEvaluator(pred_labels=preds, actual_labels=test_labels).evaluate()

        if summary:
            print(f"Accuracy: {metrics['accuracy'] * 100}%")
            print(f"Precision: {metrics['precision'] * 100}%")
            print(f"Recall: {metrics['recall'] * 100}%")
            print(f"F1-score: {metrics['f1']}")

        return metrics

    def fit(self, train_data, train_labels, test_data, test_labels, k):
        """
        Uses training data to find the k-nearest neighbours for each item in the provided testing set.

        Parameters:
            train_data (numpy.ndarray): a numpy array of training data.
            train_labels (numpy.ndarray): a numpy array of class labels corresponding to items in train_data.
            test_data (numpy.ndarray): a numpy array of testing data.
            test_labels (numpy.ndarray): a numpy array of testing labels.
            k (int): the number of nearest neighbours to use for class label prediction.

        Returns:
            (numpy.ndarray): a numpy array of predicted class labels for each item in test_data.
        """
        # Append training labels to end of data array
        train_data = np.concatenate((train_data, np.array([train_labels]).T), axis=1)

        # Append testing labels to end of test array
        test_data = np.concatenate((test_data, np.array([test_labels]).T), axis=1)

        # Obtain predicted class labels for each test item based on the training data
        preds = []
        for item in test_data:
            neighbours = self._get_k_neighbours(train_data, item, k)
            response = self._get_response(neighbours)
            preds.append(response)

        return np.array(preds)
