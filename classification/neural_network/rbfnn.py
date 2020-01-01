# Import dependencies
import numpy as np
import math
from scipy.linalg import pinv

from classification.evaluator import ClassificationEvaluator

class RBFNN:
    """ An implementation of a Radial Basis Function Neural Network """

    def __init__(self, n_prototypes=4, n_classes=2):
        """ Constructor to initialise the number of prototypes and the number of classes used in the dataset """
        self._n_prototypes = n_prototypes
        self._n_classes = n_classes

    def _dist(self, p1, p2):
        """
        Calculates the distance between 2 points p1 and p2.

        Parameters:
            p1: a numpy array containing the first point values
            p2: a numpy array containing the second point values
        """
        total = 0

        for p1, p2 in zip(p1, p2):
            total += (p1 - p2) ** 2

        return total ** 0.5

    def _max_dist(self, m1, m2):
        """
        Computes the maximum distance between the provided matrices.

        Parameters:
            m1: the first matrix of values
            m2: the second matrix of values
        Returns:
            the maximum distance between the provided matrices
        """
        max_distance = -1

        for i in range(len(m1)):
            for j in range(len(m2)):
                distance = self._dist(m1[i,:], m2[j,:])

                if (distance > max_distance):
                    max_distance = distance

        return max_distance

    def _predict(self, item):
        """
        Predicts the class label for the provided item.

        Parameters:
            item: a numpy array of data values
        Returns:
            the predicted class
        """
        out = []

        ## Hidden layer
        for proto in self._prototypes:
            distance = self._dist(item, proto)
            neuron_out = np.exp(-(distance)/self._sigma**2)
            out.append(neuron_out)

        net_out = []
        for c in range(self._n_classes):
            result = np.dot(self._weights[:,c], out)
            net_out.append(result)

        return np.argmax(net_out)

    def train(self, data, labels):
        """
        Trains the RBF neural network with the provided data and labels.

        Parameters:
            data (numpy.ndarray): an array of data values.
            labels (numpy.ndarray): an array of labels.
        """
        # Convert labels
        labels_new = []
        for label in labels:
            if (label== 0):
                labels_new.append([1,0])
            else:
                labels_new.append([0,1])

        # Generating prototypes - randomly select 4 items from each class to compose prototypes
        num_rows = data.shape[0]
        half = math.floor(num_rows/2)
        group1 = np.random.randint(0, half, size=self._n_prototypes)
        group2 = np.random.randint(half, num_rows, size=self._n_prototypes)

        self._prototypes = np.vstack([data[group1,:], data[group2,:]])

        # Compute sigma
        distance = self._max_dist(self._prototypes, self._prototypes)
        self._sigma = distance/(self._n_prototypes * self._n_classes) ** 0.5

        # For each item in training set, get the output
        output = np.zeros(shape=(num_rows, self._n_prototypes * self._n_classes))

        for item in range(num_rows):
            out = []

            for proto in self._prototypes:
                distance = self._dist(data[item], proto)
                neuron_out = np.exp(-distance/self._sigma**2)
                out.append(neuron_out)

            output[item,:] = np.array(out)

        # Use pseudo-inverse to calculate weights
        self._weights = np.dot(pinv(output), labels_new)

    def test(self, data, labels, test_data, verbose=False, summary=False):
        """
        Tests the RBFNN on new unseen testing data values.

        Parameters:
            data (numpy.ndarray): a numpy array of data values.
            labels (numpy.ndarray): a numpy array of data values.
            test_data (numpy.ndarray): a numpy array of test data values.
            verbose (boolean): True if the output of each test iteration is to be printed, False otherwise.
            summary (boolean): True if the summary is to be printed, False otherwise.

        Returns:
            (numpy.ndarray, numpy.ndarray): a numpy array of evaluation metrics and predictions
        """
        # Testing
        preds = np.array([self._predict(data[item, :]) for item in test_data])
        acc = round((preds == labels[test_data]).mean() * 100, 5)
        count = 0

        for i, item in enumerate(test_data):
            if verbose:
                print(f"Item: {item}")
                print(f"Predicted Class: {preds[i]}")
                print(f"True Class: {labels[item]} \n")

            count += labels[item] == preds[i]

        metrics = ClassificationEvaluator(pred_labels=preds, actual_labels=labels[test_data]).evaluate()

        if summary:
            print(f"Accuracy: {metrics['accuracy'] * 100}%")
            print(f"Precision: {metrics['precision'] * 100}%")
            print(f"Recall: {metrics['recall'] * 100}%")
            print(f"F1-score: {metrics['f1']}")

        return metrics, preds
