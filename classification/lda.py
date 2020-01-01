# Import dependencies
import numpy as np
import math
from scipy import linalg
import matplotlib.pyplot as plt

import preprocessing.ml as ml

class LDA:
    """ An implementation of Linear Discriminant Analysis (LDA) for dimensionality reduction and classification """
    def __init__(self):
        pass

    def _mean_features(self, data):
        """
        Calculates mean of each feature in the data matrix .

        Parameters:
            data: a numpy array of data items
        """
        return np.mean(data, axis=0)

    def train(self, data, labels):
        """
        Trains the LDA classifier with the provided data and labels.

        Parameters:
            data: a numpy array of data items
            labels: a numpy array of labels corresponding to the data items
        """
        # Get training and testing data
        # self._training_data, self._training_labels, self._test = ml.train_test_split(data, labels, 0.8)
        # self._test_data = data[self._test]
        # self._data = data
        # self._labels = labels

        class_0_inds = np.where(labels == 0)[0]
        class_1_inds = np.where(labels == 1)[0]

        data_n = data
        labels_n = labels

        testGroup1 = np.random.choice(np.arange(len(class_0_inds)), size=math.floor(0.2*len(class_0_inds)), replace=False)
        testGroup2 = np.random.choice(np.arange(len(class_1_inds)), size=math.floor(0.2*len(class_1_inds)), replace=False)
        testItems = np.concatenate([testGroup1 , testGroup2],axis=None)
        testData = data_n[testItems, :]
        trainingData = np.delete(data_n, testItems, axis=0)
        trainingLabels = np.delete(labels_n, testItems,axis=0)


        ## 1st step: get mean of features
        means = np.zeros((2, trainingData.shape[1]))

        class_0_inds = np.where(trainingLabels == 0)[0]
        class_1_inds = np.where(trainingLabels == 1)[0]

        means[0,:] = self._mean_features(trainingData[class_0_inds, :])
        means[1,:] = self._mean_features(trainingData[class_1_inds, :])

        ## For the scatter matrices, we will need the overall mean
        overallMean = self._mean_features(trainingData)

        ## 2nd step: let’s calculate the with-in class scatter matrix:
        SW = np.zeros((trainingData.shape[1],trainingData.shape[1]))

        # First class:
        for p in class_0_inds:
            diff = (trainingData[p,:] - means[0,:]).reshape(trainingData.shape[1],1)
            SW += diff.dot(diff.T)

        # Second class:
        for p in class_1_inds:
            diff = (trainingData[p,:] - means[1,:]).reshape(trainingData.shape[1],1)
            SW += diff.dot(diff.T)

        ## 3rd step: now let's calculate the between-class scatter matrix
        SB = np.zeros((data_n.shape[1],data_n.shape[1]))

        for c in range(2):
            diff = (means[c,:] - overallMean).reshape(trainingData.shape[1],1)
            SB += len(class_0_inds) * diff.dot(diff.T)

        # Compute eigenvalues and eigenvectors
        invSW = np.linalg.pinv(SW)
        eig_vals, eig_vectors = linalg.eig(invSW.dot(SB))

        # Order eigenvectors (each column) by eigenvalues in descending order
        eig_val_inds = np.flip(eig_vals.copy().argsort())
        orderedEigVectors = eig_vectors[:, eig_val_inds]

        k=2
        projectionMatrix = orderedEigVectors[:,0:k]

        ## 6th Step: Project the dataset
        ldaData = trainingData.dot(projectionMatrix)
        testLDA = testData.dot(projectionMatrix)

        plt.figure(figsize=(6,4))
        plt.plot(ldaData[class_0_inds,0],ldaData[class_0_inds,1],"r.")
        plt.plot(ldaData[class_1_inds,0],ldaData[class_1_inds,1],"g.")

        plt.show()

        class_0_inds = np.where(labels[testItems] == 0)[0]
        class_1_inds = np.where(labels[testItems] == 1)[0]

        plt.plot(testLDA[class_0_inds,0],testLDA[class_0_inds,1],"rx")
        plt.plot(testLDA[class_1_inds,0],testLDA[class_1_inds,1],"gx")

        plt.xlabel("1st Principal Component")
        plt.ylabel("2nd Principal Component")

        plt.show()

        plt.savefig("lda.pdf")
        plt.close()


        # Compute mean of all features
        # means = np.zeros((2, trainingData.shape[1]))
        #
        # means[0, :] = self._mean_features(self._class_0_items)
        # means[1, :] = self._mean_features(self._class_1_items)
        #
        # # Obtain overall mean for scatter matrices
        # self._overall_mean = self._mean_features(self._training_data)
        #
        # # Calculate within class scatter matrix
        # SW = np.zeros((self._training_data.shape[1], self._training_data.shape[1]))
        #
        # # First class
        # for p in range(0, len(self._class_0_items)):
        #     diff = (self._class_0_items[p,:] - means[0,:]).reshape(self._class_0_items.shape[1], 1)
        #     SW += diff.dot(diff.T)
        #
        # # Second class
        # for p in range(0, len(self._class_1_items)):
        #     diff = (self._class_1_items[p,:] - means[1,:]).reshape(self._class_1_items.shape[1], 1)
        #     SW += diff.dot(diff.T)
        #
        # # Calculate between-class scatter matrix
        # SB = np.zeros((data.shape[1], data.shape[1]))
        #
        # for c in range(2):
        #     diff = (means[c,:] - self._overall_mean).reshape(self._training_data.shape[1],1)
        #     SB += len(self._class_1_items) * diff.dot(diff.T)
        #
        # # Compute eigenvalues and eigenvectors
        # invSW = np.linalg.pinv(SW)
        # eig_vals, eig_vectors = linalg.eig(invSW.dot(SB))
        #
        # # Order eigenvectors (each column) by eigenvalues in descending order
        # eig_val_inds = np.flip(eig_vals.copy().argsort())
        # self._ordered_eig_vectors = eig_vectors[:, eig_val_inds]
        #
        # # Find top k eigenvectors (set k to 2 so we can visualise in 2D)
        # k = 2
        # projection_mat = self._ordered_eig_vectors[:, 0:k]
        #
        # # Step 6: project dataset and plot results
        # self._lda_data = self._training_data.dot(projection_mat)
        # self._test_LDA = self._test_data.dot(projection_mat)

    def plot(self):
        """ Plots the results of LDA using the projection matrix for the training and testing data. """

        # Plot first 2 principal components in projected LDA space
        plt.figure(figsize=(6,4))
        plt.title("Training and testing data in projected LDA space")

        plt.plot(self._lda_data[self._class_0_inds, 0], self._lda_data[self._class_0_inds, 1], "r.")
        plt.plot(self._lda_data[self._class_1_inds, 0], self._lda_data[self._class_1_inds, 1], "g.")

        self._class_0_inds_test = np.where(self._labels[self._test] == 0)
        self._class_1_inds_test = np.where(self._labels[self._test] == 1)


        plt.plot(self._test_LDA[self._class_0_inds_test, 0], self._test_LDA[self._class_0_inds_test,1], "rx")
        plt.plot(self._test_LDA[self._class_1_inds_test, 0], self._test_LDA[self._class_1_inds_test,1], "gx")

        plt.xlabel("1st Principal Component")
        plt.ylabel("2nd Principal Component")

        plt.savefig("lda.png", dpi=300)

        plt.show()
        plt.close()

        # Classification - plot results in 1D space
        k = 1
        projection_mat = self._ordered_eig_vectors[:,0:k]

        lda_data = self._training_data.dot(projection_mat)
        test_LDA = self._test_data.dot(projection_mat)
        threshold = self._overall_mean.dot(projection_mat)

        plt.figure(figsize=(6,4))
        plt.title("LDA result in 1D space")

        plt.plot(lda_data[0:45,0], np.zeros(45), "r.")
        plt.plot(lda_data[45:90,0], np.zeros(45), "g.")

        plt.plot(test_LDA[0:5,0], np.zeros(5), "rx", markersize=12)
        plt.plot(test_LDA[5:10,0], np.zeros(5), "gx", markersize=12)

        plt.plot(threshold, 0, "o")

        plt.xlabel("1st Principal Component")

        plt.savefig("lda1D.png", dpi=300)

        plt.show()
        plt.close()
