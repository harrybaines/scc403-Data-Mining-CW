# Import dependencies
import numpy as np
import torch
import time

from classification.evaluator import ClassificationEvaluator

class ShallowNeuralNetwork:
    """ An implementation of a Shallow Neural Network with a single hidden layer with customisable hyperparameters """

    def __init__(self, learning_rate=0.001, n_epochs=1000, n_hidden_units=3):
        """
        Initialises a new neural network instance with tunable hyperparameters.

        Parameters:
            learning_rate: the amount of a step to take on each iteration of gradient descent
            n_epochs: the number of training iterations
            n_hidden_units: the number of neurons in the hidden layer of the neural network
        """
        self._learning_rate = learning_rate
        self._n_epochs = n_epochs
        self._n_hidden_units = n_hidden_units

    def _convert_labels(self, labels):
        """
        Converts an array of labels to a unique set of 1's and 0's using label encoding.

        Parameters:
            labels: a numpy array of labels to convert

        Returns:
            a PyTorch tensor object of the converted labels
        """
        # Converting labels
        num_labels = len(np.unique(labels))
        conv_labels = []

        # Handle multi-class scenario
        if num_labels > 2:
            for label in labels:
                unique_label = [0] * num_labels
                unique_label[label] = 1
                conv_labels.append(unique_label)
        else:
            for label in labels:
                if label == 0:
                    conv_labels.append([1, 0])
                else:
                    conv_labels.append([0, 1])

        # Converting numpy array to PyTorch tensor
        labels = torch.tensor(conv_labels, dtype=torch.float32)
        return labels

    def _loss_fn(self):
        """ Returns the neural network's loss function """
        return torch.nn.MSELoss(reduction='sum')

    def _create_model(self, n_inputs, n_outputs):
        """
        Creates the neural network architecture with a number of inputs, outputs and a number of neurons in the hidden layer.

        Parameters:
            n_inputs: the number of neurons in the input layer
            n_outputs: the number of neurons in the output layer
        """
        return torch.nn.Sequential(
            torch.nn.Linear(n_inputs, self._n_hidden_units),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self._n_hidden_units, n_outputs),
            torch.nn.Sigmoid(),
        )

    def train(self, data, labels, verbose=False):
        """
        Trains the neural network with the provided data and labels.

        Parameters:
            data: a numpy array of data items to train
            labels: a numpy array of labels corresponding to items in the data array
            verbose: True if the loss on each training iteration is to be printed, False otherwise

        Returns:
            the trained model
        """
        # Convert labels and convert data to PyTorch tensor
        labels = self._convert_labels(labels)
        data = torch.from_numpy(data)

        n_inputs = data.shape[1]
        n_outputs = labels.shape[1]

        # Define model architecture
        model = self._create_model(n_inputs, n_outputs)

        # Loss function
        loss_fn = self._loss_fn()

        # Training loop
        for t in range(self._n_epochs):
            # Compute class predictions
            y_pred = model(data.float())

            # Compute loss
            loss = loss_fn(y_pred, labels.float())

            if verbose:
                print(t,loss.item())

            # Backpropagation (compute gradients)
            model.zero_grad()
            loss.backward()

            # Update model parameters
            with torch.no_grad():
                for param in model.parameters():
                    param -= self._learning_rate * param.grad

        return model

    def test(self, model, test, data, labels, summary=False, verbose=False):
        """
        Tests the trained model with the provided test data and labels.

        Parameters:
            model: the trained PyTorch neural network model
            test: the test data to obtain class predictions for
            data: the full dataset
            labels: test labels corresponding to items in the test set
            summary: True if a summary of the test results are to be printed, False otherwise
            verbose: True if the predictions on each test iteration is to be printed, False otherwise

        Returns:
            a dictionary of number of correctly predicted items and model accuracy
        """
        preds = []
        probs = []

        # Obtain predictions for each class
        for item in test:
            with torch.no_grad():
                prediction = model(torch.from_numpy(data[item, :]).float())

            # Extract class prediction
            class_pred = prediction.argmax().item()
            preds.append(class_pred)

            # Store probabilities for ROC curve
            probs.append(prediction[1].item())

            if verbose:
                print(f"Item: {item}")
                print(f"NN Output: {prediction}")
                print(f"Predicted class: {class_pred}")
                print(f"True Class: {labels[item]}")

        preds = np.array(preds)

        metrics = ClassificationEvaluator(pred_labels=preds, actual_labels=labels[test]).evaluate()

        if summary:
            print(f"Accuracy: {metrics['accuracy'] * 100}%")
            print(f"Precision: {metrics['precision'] * 100}%")
            print(f"Recall: {metrics['recall'] * 100}%")
            print(f"F1-score: {metrics['f1']}")

        return metrics, probs
