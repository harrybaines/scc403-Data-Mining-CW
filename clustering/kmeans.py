# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

import preprocessing.scalers as scalers
from classification.evaluator import ClassificationEvaluator

class Kmeans:
    """ K-means clustering implementation from scratch """

    def __init__(self, k=2, threshold=0.1):
        """
        Initialises a new Kmeans instance.

        Parameters:
            k (int): the number of clusters to cluster the data into.
            threshold: (float) the value which specifies when to stop updating centroid locations.
        """
        self._k = k
        self._threshold = threshold

    def fit(self, data):
        """
        Performs K-means clustering on the provided data.

        Parameters:
            data (numpy.ndarray): a numpy array of numerical values.

        Returns:
            (numpy.ndarray): the computed centroids.
        """
        self._data = data

        # Standardise the data
        standardized_data = scalers.standardize(data)

        # Initialise centroids randomly
        columns = 2
        centroids = np.random.random((self._k, columns))

        # Scale the random numbers using min and max of each feature as ranges for random sampling
        for i in range(columns):
            max_val = standardized_data[:, i].max()
            min_val = standardized_data[:, i].min()

            for c in range(self._k):
                centroids[c,i] = min_val + centroids[c,i] * (max_val - min_val)

        # Initialise centroids and clusters arrays
        old_centroids = np.zeros(centroids.shape)
        clusters = np.zeros(len(standardized_data))

        # Stores the sum of all distances between the centroids of the current and previous iteration
        dist_centroids = float("inf")

        # Iteratively update centroids while distance between new and old centroids is bigger than the threshold
        while dist_centroids > self._threshold:
            # Obtain minimum distance between standardised points and centroids
            for i in range(len(standardized_data)):
                clusters[i] = self._min_dist_pos(standardized_data[i], centroids)

            old_centroids = centroids.copy()

            # Centroid update - compute mean of surrounding points
            for i in range(self._k):
                points = np.array([])

                for j in range(len(data)):
                    if (clusters[j] == i):
                        if (len(points) == 0):
                            points = standardized_data[j,:].copy()
                        else:
                            points = np.vstack((points, standardized_data[j,:]))

                centroids[i] = np.mean(points, axis=0)

            # Compute distance between new and old centroid locations
            dist_centroids = self._sum_dist(centroids, old_centroids)

        self._centroids = centroids
        self._groups = None

        # Append cluster number to each observation in standardized data
        for d in standardized_data:
            cluster_no = self._min_dist_pos(d, self._centroids)
            d = np.append(d, cluster_no)
            if self._groups is None:
                self._groups = np.array([d])
            else:
                self._groups = np.vstack((self._groups, d))

        return self._centroids

    def predict(self, test_data, actual_labels):
        """
        Predicts the class of items in an array of test data against their actual labels.

        Parameters:
            test_data (numpy.ndarray): a numpy array of test items.
            actual_labels (numpy.ndarray): a numpy array of true labels.

        Returns:
            (numpy.ndarray, dict): a numpy array of the predicted classes and a metrics dictionary.
        """
        # Obtain cluster number for every test item
        classes = []

        cluster_no = self._min_dist_pos(test_data[0], self._centroids)

        for i, item in enumerate(test_data):
            cluster_no = self._min_dist_pos(item, self._centroids)
            classes.append(cluster_no)

        evaluator = ClassificationEvaluator(
            pred_labels = np.array(classes),
            actual_labels = actual_labels
        )

        metrics = evaluator.evaluate()
        return np.array(classes), metrics

    def _dist(self, p1, p2):
        """
        Calculates the euclidean distance between 2 points p1 and p2.

        Parameters:
            p1 (numpy.ndarray): a numpy array of the first point coordinates.
            p2 (numpy.ndarray): a numpy array of the second point coordinates.

        Returns:
            (float): the euclidean distance between the points.
        """
        total = 0

        for x, y in zip(p1, p2):
            total += (x - y) ** 2

        return total ** 0.5

    def _min_dist_pos(self, point, mat):
        """
        Finds the position of the minimum distance between a point and a set of points (mat).

        Parameters:
            point (numpy.ndarray): a numpy array containing the current point under consideration.
            mat (numpy.ndarray): a numpy array of data points.

        Returns:
            (float): the minimum distance position between the provided point and the set of points (mat).
        """
        min_val = float("inf")
        min_pos = -1

        # Calculate distance between all points and point parameter
        for row_ind in range(len(mat)):
            d = self._dist(point, mat[row_ind, :])
            if (d < min_val):
                min_val = d
                min_pos = row_ind

        return min_pos

    def _sum_dist(self, mat1, mat2):
        """
        Returns the sum across the distances between each row of mat1 and mat2.

        Parameters:
            mat1 (numpy.ndarray): a numpy array of data points.
            mat2 (numpy.ndarray): a numpy array of data points.

        Returns:
            (float): the total distance between each pair of points in matrices mat1 and mat2.
        """
        total = 0

        for pos in range(len(mat1)):
            total += self._dist(mat1[pos, :], mat2[pos, :])

        return total

    def plot(self, xlabel="Feature 1", ylabel="Feature 2", title=None, legend=None):
        """
        Plots the found clusters found with each cluster centroid.

        Parameters:
            xlabel (str): the label of the x axis on the plot.
            ylabel (str): the label of the t axis on the plot.
            title (str): the title of the plot (None if default title will be used).
            legend (list): a list of tuples in the form (label, name) (None if only a single colour will be used).
        """
        plt.figure(figsize = (6,4))

        # Set plot title and axis labels
        if title is None:
            plt.title(f"K-means clustering with {self._k} clusters")
        else:
            plt.title(title)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        cycol = cycle('bgrcmk')

        # Plot points belonging to clusters
        for i in range(self._k):
            cluster_i_inds = self._groups[self._groups[:, 2] == i]
            plt.scatter(cluster_i_inds[:, 0], cluster_i_inds[:, 1], c=next(cycol), s=10)

        # Plot cluster centres
        for centroid in self._centroids:
            plt.scatter(*centroid, c='k', marker='x', s=100)

        plt.savefig(f"./plots/clustering_results/kmeans_{self._k}clusters.pdf")
        plt.show()
        plt.close()

    def plot_k(self, k, xlabel="Feature 1", ylabel="Feature 2"):
        """
        Plots the results of k-means for k clusters in separate plots.

        Parameters:
            k (int): the total number of k-means to perform in separate plots (where each value of k represents the number of clusters in each plot).
            xlabel (str): the label of the x axis on the plot.
            ylabel (str): the label of the t axis on the plot.
        """
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(12,9))
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]

        for i in range(2, k+1):
            kmeans = Kmeans(i)
            kmeans.fit(self._data)

            cycol = cycle('bgrcmk')

            # Plot points belonging to clusters
            for j in range(kmeans._k):
                cluster_i_inds = kmeans._groups[kmeans._groups[:, 2] == j]
                axes[i-2].scatter(cluster_i_inds[:, 0], cluster_i_inds[:, 1], c=next(cycol), s=8)

            # Plot cluster centres
            for j in range(i):
                axes[i-2].scatter(kmeans._centroids[:, 0], kmeans._centroids[:, 1], c='w', marker='x', s=100)

            axes[i-2].set_title(f'K-means clustering with {i} clusters')
            axes[i-2].set_xlabel(xlabel)
            axes[i-2].set_ylabel(ylabel)

        plt.savefig(f"./plots/clustering_results/kmeans_clusters_2-{k}_plots.pdf")
        plt.show()
        plt.close()
