import numpy as np



def inner1d_sum(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the sum of the element-wise product of two arrays.

    Parameters
    ----------
    A:
        First array.
    B:
        Second array.

    Returns
    -------
        The sum of the element-wise product
    """
    return (A * B).sum()
