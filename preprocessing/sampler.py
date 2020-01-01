# Import dependencies
import numpy as np
import random

class Sampler:
    """ Sampling class which provides sampling methods for undersampling, oversampling and SMOTE """

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

        for p1, p2 in zip(p1, p2):
            total += (p1 - p2) ** 2

        return total ** 0.5

    def over_sample(self, data, labels, auto=False, k=2, l=0):
        """
        Uses the oversampling technique on the provided dataset to oversample the provided class l.

        Parameters:
            data (numpy.ndarray): a numpy array of data items to oversample.
            labels (numpy.ndarray): a numpy array of labels corresponding to data items in the data array.
            auto (boolean): True if the minority class to oversample is to be found, False otherwise.
            k (int): the number of new items to generate with the label l.
            l (int): the label to oversample.

        Returns:
            (numpy.ndarray, numpy.ndarray): a numpy array containing the original data with the oversampled classes, and a numpy array of oversampled labels.
        """
        new_data = data.copy()
        new_labels = labels.copy()

        # Automatically detect minority class to oversample
        if auto:
            values, counts = np.unique(labels, return_counts=True)
            ind = np.bincount(labels).argmin()
            num_minority = counts[ind]
            k = counts[int(not ind)] - num_minority
            l = ind

        # Find all indices with label l
        label_l_positions = [i for i in range(len(labels)) if labels[i] == l]

        # Error checking
        if len(label_l_positions) == 0:
            return f"No items with label {l} present to oversample"
        elif k <= 0:
            return "Error - enter a value of k greater than 0"

        # Select k indices to duplicate (with replacement)
        for _ in range(k):
            item_ind = label_l_positions[random.randint(0, len(label_l_positions)-1)]
            new_data = np.vstack((new_data, data[item_ind]))
            new_labels = np.append(new_labels, l)

        return new_data, new_labels

    def under_sample(self, data, labels, auto=False, k=2, l=0):
        """
        Uses the undersampling technique on the provided dataset to undersample the provided class l.

        Parameters:
            data (numpy.ndarray): a numpy array of data items to undersample.
            labels (numpy.ndarray): a numpy array of labels corresponding to data items in the data array.
            auto (boolean): True if the majority class to undersample is to be found, False otherwise.
            k (int): the number of new items to generate with the label l.
            l (int): the label to undersample.

        Returns:
            (numpy.ndarray, numpy.ndarray): a numpy array containing the original data with the undersampled classes, and a numpy array of undersampled labels.
        """
        new_data = np.array([])
        new_labels = np.array([])

        # Automatically detect majority class to undersample
        if auto:
            values, counts = np.unique(labels, return_counts=True)
            ind = np.bincount(labels).argmax()
            num_majority = counts[ind]
            k = num_majority - counts[int(not ind)]
            l = ind

        # Find all indices with label l
        label_l_positions = [i for i in range(len(labels)) if labels[i] == l]

        # Error checking
        num_labels = len(label_l_positions)
        if k > num_labels:
            print(f"Error - k={k} > {num_labels} items with label {l}, so removing {num_labels} items with label {l}")
            k = num_labels
        elif k <= 0:
            return "Error - enter a value of k greater than 0"

        # Select k indices to delete
        to_delete = random.sample(label_l_positions, k)

        # Keep original data unless present in to_delete list
        for i in range(len(data)):
            if (i not in to_delete):
                if (len(new_data) != 0):
                    new_data = np.vstack((new_data, data[i, :]))
                    new_labels = np.vstack((new_labels, labels[i]))
                else:
                    new_data = data[i, :].copy()
                    new_labels = labels[i]

        return new_data, new_labels

    def smote(self, data, labels, label, k_neighbours=2, iterations=3):
        """
        An implementation of Synthetic Minority Oversampling Technique (SMOTE).

        Parameters:
            data (numpy.ndarray): a numpy array of data items to oversample using SMOTE.
            labels (numpy.ndarray): a numpy array of labels corresponding to data items in the data array.
            label (int/str): the label to oversample.
            k_neighbours (int): the number of nearest neighbours to take into account in the algorithm.
            iterations (int): the number of items to generate.

        Returns:
            (numpy.ndarray, numpy.ndarray): a numpy array containing the original data with the oversampled classes, and a numpy array of oversampled labels.
        """
        new_data = data.copy()
        new_labels = labels.copy()

        # Find all indices with label l
        label_l_positions = [i for i in range(len(labels)) if labels[i] == label]

        for _ in range(iterations):
            # Randomly choose an item with label l
            item_i = label_l_positions[random.randint(0, len(label_l_positions)-1)]

            # Get the K nearest neighbours
            dists = []
            dist_inds = []

            for item in label_l_positions:
                if (item != item_i):
                    dists.append(self._dist(data[item], data[item_i]))
                    dist_inds.append(item)

            k_neighbours_list = []

            for n in range(k_neighbours):
                nearest = np.argmin(dists)
                k_neighbours_list.append(dist_inds[nearest])
                dists[nearest] = float("inf")

            # Randomly choose one of the k neighbours
            item_j = k_neighbours_list[random.randint(0, len(k_neighbours_list)-1)]

            alpha = random.random()
            new_point = []

            # Get the vector between the two points
            for i in range(data.shape[1]):
                xi = data[item_j, i] - data[item_i, i]
                new_point.append(data[item_i, i] + alpha*xi)

            # Update data contents with newly generated point and new label
            new_data = np.vstack((new_data, new_point))
            new_labels = np.append(new_labels, labels[item_i])

        return new_data, new_labels
