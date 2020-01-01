# Import Dependencies
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

class Hierarchical:
    """ An OOP implementation of the agglomerative hierarchical clustering algorithm """

    def __init__(self, data):
        """
        Constructor to initialise a new hierarchical clustering object.

        Parameters:
            data (numpy.ndarray): a numpy array of data values to cluster.
        """
        self._data = data

    def fit(self, clusters_to_vis=10, filename=None):
        """
        Clusters data points using agglomerative hierarchical clustering and results visualised in a dendrogram.

        Parameters:
            clusters_to_vis (int): the number of clusters to visualise in the plotted dendrogram.
            filename (str): the name of the file to save to if given, otherwise if None, don't save to file.
        """
        # Create plot
        plt.figure(figsize=(6, 4))
        plt.title("Dendrogram")

        # Generates a hierarchical cluster tree, returning linkage info in a matrix
        linkage_matrix = shc.linkage(self._data, method='ward')

        # Construct dendrogram
        dend = shc.dendrogram(linkage_matrix, truncate_mode="lastp", p=clusters_to_vis)
        plt.axhline(y=30, color='r', linestyle='--')

        # Save to file
        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()
        plt.close()
