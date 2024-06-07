import numpy as np
import opt_einsum as oe
import random


# generator fo matrix Chain Multiplication
def generate_mcm(
    num_matrices=10, min_dim=10, max_dim=1000, is_shuffle=False, seed=None
):
    """Generate a matrix chain multiplication problem.

    Args:
        num_matrices (int): The number of matrices in the chain (default: 10).
        min_dim (int): The minimum dimension of each matrix (default: 10).
        max_dim (int): The maximum dimension of each matrix (default: 1000).
        is_shuffle (bool): Whether to shuffle the einsum string and shapes (default: False).
        seed (int): The seed value for reproducibility (default: None).

    Returns:
        tuple: A tuple containing the einsum string and the shapes of the matrices.

    Raises:
        AssertionError: If the lists of einsum string and shapes have different sizes.

    Example:
        >>> generate_mcm(num_matrices=10, min_dim=10, max_dim=1000, is_shuffle=True, seed=0)
    """
    # set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # generate dimensions for each matrix
    dimensions = np.random.randint(min_dim, max_dim + 1, size=num_matrices + 1)

    # generate the einsum string and shapes
    einsum_str = ""
    shapes = []
    for i in range(num_matrices):
        left = oe.get_symbol(i)  # left index
        right = oe.get_symbol(i + 1)  # right index
        if i < num_matrices - 1:
            einsum_str += f"{left}{right},"
        else:
            einsum_str += f"{left}{right}"
        shapes.append((dimensions[i], dimensions[i + 1]))

    def shuffle_two_lists_in_same_way(list1, list2, seed):
        # ensure the lists have the same size
        assert len(list1) == len(list2), "Lists must be of the same size."

        # seed the random number generator for reproducibility (optional)
        random.seed(seed)

        # zip the lists together and convert to a list of tuples
        zipped_list = list(zip(list1, list2))

        # shuffle the list of tuples
        random.shuffle(zipped_list)

        # unzip the list of tuples back into two lists
        list1_shuffled, list2_shuffled = zip(*zipped_list)

        # convert the tuples back to lists
        return list(list1_shuffled), list(list2_shuffled)

    # generate the output part of the einsum string
    es = einsum_str.split(",")
    if is_shuffle:
        es, shapes = shuffle_two_lists_in_same_way(es, shapes, seed)

    einsum_str = ",".join(es)
    output_str = f"{oe.get_symbol(0)}{oe.get_symbol(num_matrices)}"
    einsum_str += f"->{output_str}"

    return einsum_str, shapes
