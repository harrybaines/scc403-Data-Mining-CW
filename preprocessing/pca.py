# Import dependencies
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import preprocessing.scalers as sc

class PCA:
    """
    An implementation of Principal Component Analysis from scratch.
    This class enables the PCA results to be calculated and visualised in 2-dimensional space.
    """
    def __init__(self, n_components=2):
        """
        Initialises a new PCA instance with the specified number of components.

        Parameters:
            n_components (int): the number of principal components to retain.
        """
        self._n_components = n_components

    def fit(self, data):
        """
        Calculates the projected data and explained variance from the provided data.

        Parameters:
            data (numpy.ndarray): a numpy array of data to perform PCA on.
        """
        self._data = data

        # Calculate eigenvectors and eigenvalues
        eig_vectors, eig_vals = self._calc_eig_vects_vals(self._data)

        # Calculate projection matrix using top k eigenvectors
        self.proj_matrix = self._calc_proj_matrix(eig_vectors, k=2)

        # With the projection matrix P, we only need to calculate Y = X×P, where X is the centralized data, and Y is the transformed data with only k features
        self.pca_by_hand_data = self.centralized.dot(self.proj_matrix)

        # Calculate the explained variance for each principal component
        self.explained_variance = self._calc_explained_variance(eig_vals)

    def _centralize(self, data):
        """ Centralizes a numpy matrix by subtracting each value by the mean of each column.

        Parameters:
            data (numpy.ndarray): the data to centralize.

        Returns:
            (numpy.ndarray): the centralized numpy array.
        """
        mus = np.mean(data, axis=0)
        return data - mus

    def _calc_eig_vects_vals(self, data):
        """
        Calculates the eigenvectors and eigenvalues from the provided data matrix.

        Parameters:
            data (numpy.ndarray): a numpy array of data values.

        Returns:
            (numpy.ndarray, numpy.ndarray): a tuple of the calculated eigenvectors and eigenvalues.
        """
        # Centralize each feature separately
        self.centralized = self._centralize(sc.normalize(data))

        # Calculate covariance matrix
        cov = np.cov(self.centralized, rowvar=False)

        # Calculate eigenvalues and eigenvectors of the covariance matrix
        eig_vals, eig_vectors = linalg.eig(cov)

        self.eig_vals, self.eig_vectors = eig_vals, eig_vectors
        self.eig_vals = np.real(self.eig_vals)

        # Order eigenvectors according to the eigenvalues
        ordered_eig_vectors = np.empty(eig_vectors.shape)

        # Order eigenvectors (each column) by eigenvalues in descending order
        eig_val_inds = np.flip(eig_vals.copy().argsort())
        eig_vectors_ordered = eig_vectors[:, eig_val_inds]

        return (eig_vectors_ordered, eig_vals)

    def _calc_proj_matrix(self, eig_vectors, k=2):
        """
        Calculates the projection matrix using the top k eigenvectors.

        Parameters:
            eig_vectors (numpy.ndarray): a numpy array of eigenvectors.

        Returns:
            (numpy.ndarray): the projection matrix as a numpy array.
        """
        return eig_vectors[:, 0:k]

    def get_corr_coef(self):
        """
        Obtains a correlation matrix of correlation values for the centralized data values.

        Returns:
            (numpy.ndarray): a numpy array of correlation values.
        """
        return np.corrcoef(self.centralized, rowvar=False)

    def _calc_explained_variance(self, eig_vals):
        """
        Calculates the explained variance for each principal component.

        Parameters:
            eig_vals (numpy.ndarray): a numpy array of eigenvalues.

        Returns:
            (numpy.ndarry): a numpy array of variances explained by each principal component.
        """
        return [(eig_val/sum(self.eig_vals)) * 100 for eig_val in self.eig_vals]

    def plot(self, labels, legend, color_labels=True, filename=False):
        """
        Plots the results of PCA in a 2-dimensional space.

        Parameters:
            labels (numpy.ndarray): a numpy array of labels.
            legend (list): a python list of tuples in the form (label, label).
            colour_labels (boolean): True if the labels are to be coloured, False otherwise.
            filename (str): a filename string if the plot is to be saved to a file, None otherwise.
        """
        plt.figure(figsize = (6,4))
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2',)
        plt.title('2 component PCA')

        targets = legend
        colors = ["r", "b"]

        # Scatter points for each class label
        for target, color in zip(targets,colors):
            indicesToKeep = np.where(labels == target[0])
            color = 'b' if not color_labels else color
            plt.scatter(self.pca_by_hand_data[indicesToKeep,0], self.pca_by_hand_data[indicesToKeep,1], c = color, s = 5)

        # Add legend if multiple colours are to be used
        if color_labels:
            text_labs = [t[1] for t in targets]
            plt.legend(text_labs)

        # Save to file if necessary
        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()
        plt.close()

    def scree_plot(self, filename=None):
        """
        Creates a scree plot showing the percentage of variance explained by each principal component (PC).

        Parameters:
            filename (str): a filename string if the scree plot is to be saved to a file, None otherwise.
        """
        #
        plt.figure(figsize=(6, 4))

        # Obtain labels
        labels = [i for i in range(1, self._n_components+1)]

        # Obtain variances
        variances = [var for var in self.explained_variance[0:self._n_components]]
        variances = np.round(variances, 1)

        # Plot bars and lines
        plt.bar(labels, variances, tick_label=labels)
        plt.plot(labels, variances, c='black', marker='o', markersize=3)

        # Show variances as labels for each bar
        for i, v in enumerate(variances):
            plt.annotate(str(v)+'%', (labels[i], v+0.5))

        plt.title("Scree Plot")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained (%)")

        # Save to file if necessary
        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()
        plt.close()
