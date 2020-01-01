# Import dependencies
import numpy as np
import preprocessing.ml as ml

def open_csv(filepath, missing_char='?'):
    """ Opens a .csv file with the provided filename.

    Parameters:
        filepath (str): the filepath to the file to open.

    Returns:
        numpy.ndarray: a numpy representation of the data in the csv file.
    """
    data = []
    with open(filepath, 'r') as f:
        while True:
            line = f.readline().rstrip()
            if len(line) == 0:
                break

            read_data = line.split(",")
            data.append(read_data)

    data = np.array(data)

    print(f"[{filepath}] {data.shape[0]} row(s) loaded successfully")

    num_missing = ml.get_num_missing_values(data, missing_char)
    print(f"[{filepath}] {num_missing} missing values found")

    return data

def open_cols(filepath):
    """
    Reads a list of column names in a text file separated by newline characters into a python list.

    Parameters:
        filepath (str): the filepath to the column names file.

    Returns:
        list: a list of the column names.
    """
    cols = []
    with open(filepath, 'r') as file:
        lines = (line.rstrip() for line in file)
        cols = list(line for line in lines if line)
    return cols

def head(data, n=3):
    """
    Prints the first n rows in a dataset.

    Parameters:
        data (numpy.ndarray): a numpy array of data values.
        n (int): the number of rows to print.
    """
    print(data[:n])
