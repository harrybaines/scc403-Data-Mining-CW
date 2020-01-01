# Import dependencies
import numpy as np
import math

def train_test_split(data, labels, train_size=0.8):
    """
    Divides the provided data (and labels) into training and testing sets.

    Parameters:
        data (numpy.ndarray): a numpy array of data items to split.
        labels (numpy.ndarray): a numpy array of labels corresponding to items in the data array.
        train_size (float): the percentage of items in the data matrix to consider for random selection to be in the training set (test size will be total - train size)

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): the training data, training labels and indices of test observations in original data.
    """
    # Get train and test set sizes
    num_items = len(data)
    num_train_items = math.floor(train_size * num_items)
    num_test_items = num_items - num_train_items

    # Create test set
    test = np.random.choice(num_items, size=num_test_items, replace=False)

    # Create training set excluding test set items
    train_X = np.delete(data, test, axis=0)
    train_Y = np.delete(labels, test, axis=0)

    return train_X, train_Y, test

def get_num_missing_values(data, missing_char='?'):
    """
    Obtains the number of missing values in a dataset.

    Parameters:
        data (numpy.ndarray): a dataset of values to find missing values for.
        missing_char (str): the missing character to find.

    Returns:
        int: the number of missing values in the dataset.
    """
    return len(data[np.isin(data, missing_char)])

def get_row_count(data, col, val):
    """
    Obtains total occurences of a value in a given column.

    Parameters:
        data (numpy.ndarray): a numpy array of data values.
        col (int): the column index to search in.
        val (int/str): the value to search for.

    Returns:
        int: the number of values found in the given column.
    """
    return len(data[data[:,col] == val])

def get_num_labels(labels, l):
    """
    Obtains the number of class labels l in the provided labels list.

    Parameters:
        labels (numpy.ndarray): a numpy array of class labels
        l (str): the class label to search for

    Returns:
        (int): the number of class labels l
    """
    return len(np.where(labels == l)[0])

def get_label_counts(labels):
    """
    Obtains the counts of unique labels in a list of labels.

    Parameters:
        labels (numpy.ndarray): a numpy array of labels

    Returns:
        (numpy.ndarray): the counts for each label
    """
    return np.array(np.unique(labels, return_counts=True)).T

def find_outliers(data):
    """
    Finds potential outliers in a data matrix (after standardization).

    Parameters:
        data (numpy.ndarray): a numpy array of data values.
    """
    # sc.standardize - calculate Z-score
    stand = scalers.standardize(data)

    # Values less than -3 or greater than 3 are considered outliers
    row_inds, col_inds = np.where((stand < -3) | (stand > 3))

    num_outliers = row_inds.shape[0]
    print(f"[Found {num_outliers} potential outliers]\n")

    for ind in range(num_outliers):
        row = row_inds[ind]
        col = col_inds[ind]
        print(f"Row {row}, column {col} --> {data[row][col]} (Z-score: {stand[row][col]})")

def get_minority_class(labels):
    """
    Obtains the minority class labels based on a list of labels.

    Parameters:
        labels (numpy.ndarray): a numpy array of class labels

    Returns:
        (str/int): the minority class label
    """
    # Get unique class labels
    unique_labels = np.unique(labels)
    minority_class = float("inf")

    # Find minority class
    for label in unique_labels:
        label_inds = np.where(labels == label)[0]
        if len(label_inds) < minority_class:
            minority_class = label

    return minority_class

def remove_rows_with_missing(data, missing_char='?'):
    """
    Removes rows from a data array containing at least 1 missing value.

    Parameters:
        data (numpy.ndarray): a numpy array of data values.
        missing_char (str): the encoded missing character.

    Returns:
        (numpy.ndarray): the original numpy array with deleted rows containing missing values.
    """
    row_mask = (data != missing_char).all(axis=1)
    return data[row_mask, :]

def remove_columns_with_missing(data, missing_char='?'):
    """
    Removes columns from a data array containing at least 1 missing value.

    Parameters:
        data (numpy.ndarray): a numpy array of data values.
        missing_char (str): the encoded missing character to find.

    Returns:
        (numpy.ndarray): the original numpy array with deleted columns containing missing values.
    """
    return np.delete(data, np.where(data == missing_char), 1)

def impute_mode(data, missing_char='?'):
    """
    Performs imputation on the provided data array by replacing missing values in a column with the mode of the values in that column.

    Parameters:
        data (numpy.ndarray): a numpy array of data values.
        missing_char (str): the encoded missing character to find.

    Returns:
        (numpy.ndarray): the original numpy array with missing values in columns replaced with their mode.
    """
    new_data = data.copy()

    # Find columns containing at least 1 missing value
    cols_missing_vals = np.where(data == missing_char)[1]

    for col in cols_missing_vals:
        # Obtain current column values and indices of missing character cells
        column = data[:, col]
        missing_inds = np.where(column == missing_char)

        # Obtain unique values and position of most occuring
        unique, pos = np.unique(column, return_inverse=True)
        maxpos = np.bincount(pos).argmax()

        # Obtain mode of column and replace missing values with this
        most_freq = unique[maxpos]

        new_data[missing_inds, col] = most_freq

    return new_data

def plot(data, x_col_ind, y_col_ind, cols, filename, scale_type="norm"):
    """
    Plots 2 features of the original data against each other next to a transformed version of the same features.

    Parameters:
        data (numpy.ndarray): a numpy array of the original data values.
        x_col_ind (int): a column index of the feature to plot on the x-axis of both plots.
        y_col_ind (int): a column index of the feature to plot on the y-axis of both plots.
        cols (list): a list of column names as strings.
        filename (str): the name of the file to save the plot to.
        scale_type (str): the transformation to apply to the data values (one of norm/stand)
    """
    if scale_type == "norm":
        scaled_data = scalers.normalize(data)
        scaled_title = "Normalized Data"
    elif scale_type == "stand":
        scaled_data = scalers.standardize(data)
        scaled_title = "Standardized Data"
    else:
        return -1

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    fig.set_size_inches(6, 3)

    ax1.plot(data[:,x_col_ind], data[:,y_col_ind],".", markersize=3)
    ax2.plot(scaled_data[:,x_col_ind], scaled_data[:,y_col_ind],".", markersize=3)

    ax1.set_xlabel(cols[x_col_ind])
    ax2.set_xlabel(cols[x_col_ind])

    ax1.set_ylabel(cols[y_col_ind])
    ax2.set_ylabel(cols[y_col_ind])

    ax1.set_title('Original Data', fontsize=10)
    ax2.set_title(scaled_title, fontsize=10)

    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
