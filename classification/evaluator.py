# Import dependencies
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

class ClassificationEvaluator:
    """ Class to evaluate the performance of an ML model given the predicted and actual labels """

    def __init__(self, pred_labels=None, actual_labels=None):
        """
        Initialise the predicted and actual labels the evaluator will work with.

        Parameters:
            pred_labels (numpy.ndarray): a numpy array of predicted class labels.
            actual_labels (numpy.ndarray): a numpy array of actual class labels.
        """
        self._pred_labels = pred_labels
        self._actual_labels = actual_labels

    def _accuracy(self):
        """
        Computes the accuracy of the classifier.
        Accuracy = (TP + TN) / (TP + TN + FP + FN)

        Returns:
            (float): the accuracy of the classifier.
        """
        total = self._TP + self._TN + self._FP + self._FN
        return (self._TP + self._TN) / total if total != 0 else 0

    def _precision(self):
        """
        Computes the precision of the classifier.
        How many predicted positives were correct?
        Precision = TP / (TP + FP)

        Returns:
            (float): the precision of the classifier.
        """
        return self._TP / (self._TP + self._FP) if self._TP + self._FP != 0 else 0

    def _recall(self):
        """
        Computes the recall of the classifier.
        How many actual positives were predicted correctly?
        Recall = TP / (TP + FN)

        Returns:
            (float): the recall of the classifier.
        """
        return (self._TP / (self._TP + self._FN)) if self._TP + self._FN != 0 else 0

    def _tp(self):
        """
        Computes the number of true positives based on the predicted and actual labels.
        Number of outcomes where the model correctly predicts the positive class.

        Returns:
            (float): the number of true positives.
        """
        return np.logical_and(self._pred_labels == 1, self._actual_labels == 1).sum()

    def _tn(self):
        """
        Computes the number of true negatives based on the predicted and actual labels.
        Number of outcomes where the model correctly predicts the negative class.

        Returns:
            (float): the number of true negatives.
        """
        return np.logical_and(self._pred_labels == 0, self._actual_labels == 0).sum()

    def _fp(self):
        """
        Computes the number of false positives based on the predicted and actual labels.
        Number of outcomes where the model incorrectly predicts the positive class.

        Returns:
            (float): the number of false positives.
        """
        return np.logical_and(self._pred_labels == 1, self._actual_labels == 0).sum()

    def _fn(self):
        """
        Computes the number of false negatives based on the predicted and actual labels.
        Number of outcomes where the model incorrectly predicts the negative class.

        Returns:
            (float): the number of false negatives.
        """
        return np.logical_and(self._pred_labels == 0, self._actual_labels == 1).sum()

    def _tpr(self):
        """
        Computes the true positive rate (i.e. recall).
        TPR = TP / (TP + FN)

        Returns:
            (float): the true positive rate.
        """
        return self._recall()

    def _fpr(self):
        """
        Computes the false positive rate.
        FPR = FP / (FP + TN)

        Returns:
            (float): the false positive rate.
        """
        return self._FP / (self._FP + self._TN) if self._FP + self._TN != 0 else 0

    def evaluate(self):
        """
        Method to compute the accuracy, precision, recall and F1 score metrics for this classifier (to 4.d.p).

        Returns:
            (dict): a dictionary containing the accuracy, precision, recall and F1 metrics, and the confusion matrix.
        """
        self._TP = self._tp()
        self._TN = self._tn()
        self._FP = self._fp()
        self._FN = self._fn()

        acc = round(self._accuracy(), 4)
        prec = round(self._precision(), 4)
        rec = round(self._recall(), 4)
        f1 = round(2 * ((prec * rec) / (prec + rec)), 4) if prec + rec != 0 else 0

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cm': {
                'tp': self._TP,
                'tn': self._TN,
                'fp': self._FP,
                'fn': self._FN,
            }
        }

    def compute_avg_metrics(self, metrics, iterations, test_str=None, summary=True):
        """
        Computes the average of each evaluation metric based on a list of metrics.

        Parameters:
            metrics (numpy.ndarray): a numpy array of evaluation metric dictionaries.
            iterations (int): the number of iterations used to obtain the metrics.
            test_str (str): a useful string to print to the console output when testing.
            summary (boolean): True if a summary of the average metrics are to be printed, False otherwise.

        Returns:
            (dict): a python dictionary containing all average evaluation metrics.
        """
        avg_acc = avg_prec = avg_rec = avg_f1 = 0
        for metric_res in metrics:
            avg_acc += metric_res['accuracy']
            avg_prec += metric_res['precision']
            avg_rec += metric_res['recall']
            avg_f1 += metric_res['f1']

        avg_acc = (avg_acc/iterations) * 100
        avg_prec = (avg_prec/iterations) * 100
        avg_rec = (avg_rec/iterations) * 100
        avg_f1 = (avg_f1/iterations) * 100

        if summary:
            if test_str:
                print(test_str)
            print(f'Average accuracy: {avg_acc}%')
            print(f'Average precision: {avg_prec}%')
            print(f'Average recall: {avg_rec}%')
            print(f'Average f1: {avg_f1}%')

        return {
            'avg_accuracy': avg_acc,
            'avg_precision': avg_prec,
            'avg_recall': avg_rec,
            'avg_f1': avg_f1
        }

    def plot_roc_curve(self, true_labels, probs, alg_strs, filename=None):
        """
        Plots an ROC curve based on a set of true labels and predicted probabilites for class 1.

        Parameters:
            true_labels (numpy.ndarray): a numpy array of true class labels.
            probs (numpy.ndarray): a numpy array of probabilities for class 1.
            alg_strs (list): a python list of the classification algorithms used for the legend in the plot.
            filename (str): the name of the file to save to if not None.
        """
        tuple_dicts = []

        for true_labs, prob_val in zip(true_labels, probs):
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(true_labs, prob_val)
                roc_auc[i] = auc(fpr[i], tpr[i])

            tuple_dicts.append((fpr, tpr))

        plt.figure(figsize = (6,4))

        for fpr, tpr in tuple_dicts:
            plt.plot(fpr[1], tpr[1])

        plt.legend(alg_strs)
        plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'b--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')

        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()
