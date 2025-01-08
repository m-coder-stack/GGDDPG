import numpy
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from typing import Tuple

def get_data(file_path: str) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Loads data from a CSV file and processes it into three numpy arrays.
    Args:
        file_path (str): The path to the CSV file containing the data.
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: 
            - x_train: A sorted numpy array of unique values from the first column of the data.
            - y_train: A sorted numpy array of unique values from the second column of the data.
            - z_train: A 2D numpy array where each element corresponds to the value from the third column of the data, 
                       indexed by the positions of the first and second column values in x_train and y_train respectively.
    """
    
    
    data = numpy.loadtxt(file_path, delimiter=',')
    x_train = set(data[:, 0])
    y_train = set(data[:, 1])
    x_train = numpy.array(sorted(list(x_train)))
    y_train = numpy.array(sorted(list(y_train)))
    z_train = numpy.zeros((len(x_train), len(y_train)), dtype=numpy.float64)
    for i in range(len(data)):
        x_index = numpy.where(x_train == data[i, 0])[0][0]
        y_index = numpy.where(y_train == data[i, 1])[0][0]
        z_train[x_index, y_index] = data[i, 2]
    return x_train, y_train, z_train

def expand_data(x_train: numpy.ndarray, y_train: numpy.ndarray, z_train: numpy.ndarray, scale: Tuple[int, int]) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Expands the given training data by interpolating it to a finer grid.
    Parameters:
    x_train (numpy.ndarray): 1D array of x-coordinates of the training data.
    y_train (numpy.ndarray): 1D array of y-coordinates of the training data.
    z_train (numpy.ndarray): 2D array of z-values corresponding to the (x, y) coordinates.
    scale (Tuple[int, int]): Tuple containing the scaling factors for the x and y dimensions.
    Returns:
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: 
        - x_new: 1D array of the new x-coordinates after scaling.
        - y_new: 1D array of the new y-coordinates after scaling.
        - z_new: 2D array of the new z-values after interpolation.
    """
    
    x_new = numpy.linspace(x_train[0], x_train[-1], scale[0] * len(x_train) - scale[0] + 1)
    y_new = numpy.linspace(y_train[0], y_train[-1], scale[1] * len(y_train) - scale[1] + 1)
    f = RectBivariateSpline(x_train, y_train, z_train, kx=1, ky=1)
    z_new = f(x_new, y_new)
    return x_new, y_new, z_new


def get_model(x_train: numpy.ndarray, y_train: numpy.ndarray, z_train: numpy.ndarray, method: str) -> RectBivariateSpline:
    """
    Generates a model using the provided training data and interpolation method.
    Args:
        x_train (numpy.ndarray): The training data for the x-axis.
        y_train (numpy.ndarray): The training data for the y-axis.
        z_train (numpy.ndarray): The training data for the z-axis.
        method (str): The interpolation method to be used.
    Returns:
        RectBivariateSpline: The interpolator object created using the provided data and method.
    """
    interpolator = RegularGridInterpolator((x_train, y_train), z_train, method=method)
    return interpolator