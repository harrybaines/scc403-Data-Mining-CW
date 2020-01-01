###
## Preprocessing code
###
def normalize(x):
    """ Normalizes a numpy array to the range [0-1].

    Parameters:
        x (numpy.ndarray): a numpy array of features to normalize.

    Returns:
        numpy.ndarray: the normalized numpy array.
    """
    return (x - x.min(axis=0)) / x.ptp(axis=0)

def standardize(x):
    """ Standardizes a numpy array by subtracing the mean and dividing by the standard deviation for all values of each feature.

    Parameters:
        x (numpy.ndarray): a numpy array of features to standardize.

    Returns:
        numpy.ndarray: the standardized numpy array.
    """
    return (x - x.mean(axis=0)) / x.std(axis=0)
