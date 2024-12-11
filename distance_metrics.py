import numpy as np


# returns Euclidean distance between vectors and b
def euclidean(a, b):
    """ Returns the euclidean distance between vectors a and b

    Args:
        a (np.ndarray): A vector of any dimension
        b (np.ndarray): A vector of any dimension

    Returns:
        float: The euclidean distance between the two vectors

    Raises:
        ValueError: If the given vectors are different dimensions
    """
    return np.sqrt(np.sum(np.subtract(a, b)**2))


# returns Cosine Similarity between vectors a and b
def cosim(a, b):
    """ Returns the cosine similarity between vectors a and b

    Args:
        a (np.ndarray): A vector of any dimension
        b (np.ndarray): A vector of any dimension

    Returns:
        float: The cosine similarity between the two vectors

    Raises:
        ValueError: If the given vectors are different dimensions
    """

    if not in_same_dimension(a, b):
        print(
            f"Given vectors have different shapes: " +
            f"{np.shape(a)} != {np.shape(b)}"
        )
        raise ValueError(
            "Cosine Similarity requires 2 identically-shaped vectors"
        )
    numerator = np.dot(a, b)
    denominator = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))

    # If denominator is 0, cosim is undefined not 0
    if denominator == 0:
        return np.nan

    return numerator / denominator


# returns Pearson Correlation between vectors a and b
def pearson(a: np.ndarray, b: np.ndarray):
    """ Returns the pearson correlation between vectors a and b

    Args:
        a (np.ndarray): A vector of any dimension
        b (np.ndarray): A vector of any dimension

    Returns:
        float: The pearson correlation between the two vectors

    Raises:
        ValueError: If the given vectors are different dimensions
    """
    a_bar = np.sum(a)/len(a)
    b_bar = np.sum(b)/len(b)

    numerator = np.sum((a - a_bar)*(b - b_bar))
    denominator = np.sqrt(np.sum((a - a_bar)**2)) * \
        np.sqrt(np.sum((b - b_bar)**2))

    if numerator == 0 or denominator == 0:
        return 0

    return numerator / denominator


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """ Returns the Hamming distance between vectors a and b

    Args:
        a (np.ndarray): A vector of any dimension
        b (np.ndarray): A vector of any dimension

    Returns:
        int: The Hamming distance between the two vectors

    Raises:
        ValueError: If the given vectors are different dimensions
    """
    # Ensure that the two vectors occupy the same dimension
    if not in_same_dimension(a, b):
        print(
            f"Given vectors have different shapes: " +
            f"{np.shape(a)} != {np.shape(b)}"
        )
        raise ValueError("Hamming requires 2 identically-shaped vectors")
    # Create a vector
    comparison_vector = np.array([ai != bi for ai, bi in zip(a, b)])
    return comparison_vector.sum()


def in_same_dimension(a: np.ndarray, b: np.ndarray) -> bool:
    """ Determines if the two given vectors are in the 
        same dimension or not

    Args:
        a (np.ndarray): The first vector of any dimension
        b (np.ndarray): The second vector of any dimension

    Returns:
        bool: Whether the 2 vectors are the same dimension or not
    """
    return np.shape(a) == np.shape(b)
