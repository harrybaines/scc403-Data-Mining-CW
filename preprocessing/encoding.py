# Import dependencies
import numpy as np

class LabelEncoder:
    """ Encodes a categorical variable into a set of unique numerical values """

    def fit_transform(self, data, col_idx):
        """
        Encodes a categorical column, specified by an index, in a data array to numerical values.

        Parameters:
            data (numpy.ndarray): a numpy array of all data values.
            col_idx (int): the column index to transform to numerical values.

        Returns:
            (numpy.ndarray): a new numpy array of the original data including the new numerically encoded values.
        """
        # Label encode all columns if col_idx is -1
        if col_idx == -1:
            num_cols = data.shape[1]

            # Perform label encoding for num_cols columns
            for col in range(num_cols):
                encoded_column = self._label_encode_column(data, col)
                data[:, col] = encoded_column
        else:
            # Label encode chosen column
            encoded_column = self._label_encode_column(data, col_idx)
            data[:, col_idx] = encoded_column

        return data.astype(float)

    def _label_encode_column(self, data, col_idx):

        column_vals = data[:, col_idx]

        # Get unique values for this column
        unique_vals = np.unique(column_vals)

        # Use numerical index of unique_vals for each value as new coded value
        column_new = column_vals.copy()

        # Obtain all indices for this unique value and set index
        for i, val in enumerate(unique_vals):
            inds = np.where(column_vals == val)
            column_new[inds] = i

        return column_new

class OneHotEncoder:
    """ Encodes a numerical variable to a set of unqiue one-hot encoded values """

    def _get_one_hot(self, targets, num_classes):
        """
        One-hot encodes a set of targets based on the number of different/unique values it takes.

        Parameters:
            targets (numpy.ndarray): a numpy array of values to one-hot encode.
            num_classes (int): the number of unique values that can be found in the targets list.

        Returns:
            (numpy.ndarray): a numpy array of one-hot encoded values.
        """
        res = np.eye(num_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [num_classes])

    def fit_transform(self, data, columns, unique_col_vals):
        """
        Parameters:
            data (numpy.ndarray): a numpy array of all data values.
            columns (list): a list of column names corresponding to the data array.
            unique_col_vals (numpy.ndarray): a list of unique values for each column in the columns list.
        """
        # Create column names for unique values of each column
        new_columns = list(f'{columns[i]}_{val}' for i, col_vals in enumerate(unique_col_vals) for val in col_vals)

        new_data = np.array([])

        # One-hot encode the columns
        for col_idx in range(data.shape[1]):
            column_name = columns[col_idx]
            column_vals = data[:, col_idx].astype(int)
            num_classes = len(unique_col_vals[col_idx])

            # One-hot encoding
            one_hot_encoded_col = self._get_one_hot(column_vals, num_classes)

            # Append one-hot encoded column to results
            if col_idx != 0:
                new_data = np.concatenate((new_data, one_hot_encoded_col), axis=1)
            else:
                new_data = one_hot_encoded_col

        return new_data, np.array(new_columns)
