# Import dependencies
import numpy as np

from classification.evaluator import ClassificationEvaluator

class LogisticRegression:
    """ An implementation of a logistic regression classifier """

    def __init__(self, alpha=0.01, epochs=25_000):
        """ Constructor to initialise alpha (learning rate) and the number of epochs for training """
        self._epochs = epochs
        self._alpha = alpha

    def _sigmoid(self, z, derivative=False):
        """
        Sigmoid/logistic function to add non-linearity enabling predictions to be computed.

        Parameters:
            z: the input to the logistic function
            derivative: True if the derivative of the sigmoid is to be computed, False for normal sigmoid calculation

        Returns:
            the sigmoid (or derivative) calculation for the input z
        """
        if derivative:
            return z * (1 - z)
        return 1 / (1 + np.exp(-z))

    def _loss(self, s, y):
        """
        Loss function calculates difference between prediction and actual label.

        Parameters:
            s: the predicted class for this data item as a probability from 0-1
            y: the actual class label for this data item

        Returns:
            the difference between the prediction and actual label
        """
        return s - y

    def _add_intercept(self, data):
        """
        Appends a column of 1's to first column of all items (when fitting intercept: beta_0 * 1 for all items).

        Parameters:
            data: a numpy array of data items

        Returns:
            the original data with a column of 1's appended to the front
        """
        ones = np.ones((data.shape[0], 1))
        return np.concatenate((np.array(ones).reshape(len(ones), 1), data), axis=1)

    def fit(self, X, Y):
        """
        Trains the logistic regression classifier using the Stochastic Gradient Descent optimisation algorithm.

        Parameters:
            X: a numpy array of data items
            Y: a numpy array of class labels
        """
        # Set seed for reproducability
        np.random.seed(1)

        # Append 1's to first column of all items so beta_0 can be multiplied by 1 for each item
        self._X = self._add_intercept(X)
        self._Y = Y

        # Initialise weights to small random numbers
        self._weights = np.random.random((self._X.shape[1], 1)) * 0.01

        # Iterative training loop
        for i in range(self._epochs):
            # Order data items in X randomly (must shuffle labels Y in unison)
            s = np.arange(0, self._X.shape[0], 1)
            np.random.shuffle(s)

            self._X = self._X[s]
            self._Y = self._Y[s]

            # Compute z for all items, store in matrix Z: z = beta_0 * 1 + beta_1 * x_1 + beta_2 * x_2
            Z = self._X.dot(self._weights)

            # Compute sigmoid for all values of z, store in matrix A
            A = self._sigmoid(Z)

            # Compute loss over all items
            L = self._loss(A, self._Y)

            # Calculate gradient of error function with respect to paramaters
            grads = self._X.T.dot(L * self._sigmoid(A, True))

            # Update weights by subtracting alpha*gradient from each weight
            self._weights -= self._alpha * grads

    def predict(self, data, test, labels, summary=True, threshold=0.5):
        """
        Tests the trained logistic regression model with the provided data and weights.

        Parameters:
            data: a numpy array of data items to predict class labels for
            test (numpy.ndarray): a numpy array of indices corresponding to items in the data array
            labels: a numpy array of labels corresponding to the items in the data array
            summary: True if a summary of the test results are to be printed, False otherwise
            threshold: a tunable paramater above which predictions will be given class 1, otherwise class 0

        Returns:
            a numpy array of predicted class labels for each data item
        """
        # Append 1's to first column of all items (when fitting intercept: beta_0 * 1 for all items)
        data = self._add_intercept(data)

        # Compute z for all items: z = beta_0 * 1 + beta_1 * x_1 + beta_2 * x_2
        Z = data.dot(self._weights)

        # Compute sigmoid for all values of z in matrix Z
        A = self._sigmoid(Z)

        # Store probabilities for ROC curve
        probs = A.flatten()

        # Compute total number of correct predictions
        preds = (A >= threshold).astype(np.int)

        metrics = ClassificationEvaluator(pred_labels=preds, actual_labels=labels[test]).evaluate()

        if summary:
            print(f"Accuracy: {metrics['accuracy'] * 100}%")
            print(f"Precision: {metrics['precision'] * 100}%")
            print(f"Recall: {metrics['recall'] * 100}%")
            print(f"F1-score: {metrics['f1']}")

        return metrics, probs
