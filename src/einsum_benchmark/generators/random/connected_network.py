from typing import Union, Optional, Tuple, Dict, Collection
from numpy.random import default_rng

# from opt_einsum.paths import _find_disconnected_subgraphs #NOTE: only for testing

from .util import get_symbol

PathType = Collection[Tuple[int, ...]]


def random_tensor_network(
    number_of_tensors: int,
    regularity: float,
    number_of_output_indices: int = 0,
    min_axis_size: int = 2,
    max_axis_size: int = 10,
    seed: Optional[int] = None,
    global_dim: bool = False,
    return_size_dict: bool = False,
) -> Union[Tuple[str, PathType, Dict[str, int]], Tuple[str, PathType]]:
    """Generate a random connected Tensor Network (TN). Returns an einsum expressions string representing the TN, shapes of the tensors and optionally a dictionary containing the index sizes.

    Parameters
    ----------
    number_of_tensors : int
        Number of tensors/arrays in the TN.
    regularity : float
        'Regularity' of the TN. This determines how
        many indices/axes each tensor shares with others on average (not counting output indices and a global dimension).
    number_of_output_indices : int, optional
        Number of output indices/axes (i.e. the number of non-contracted indices) including the global dimension.
        Defaults to 0 in case of no global dimension, i.e., a contraction resulting in a scalar, and to 1 in case there is a global dimension.
    min_axis_size : int, optional
        Minimum size of an axis/index (dimension) of the tensors.
    max_axis_size : int, optional
        Maximum size of an axis/index (dimension) of the tensors.
    seed: int, optional
        If not None, seed numpy's random generator with this.
    global_dim : bool, optional
        Add a global, 'broadcast', dimension to every operand.
    return_size_dict : bool, optional
        Return the mapping of indices to sizes.

    Returns
    -------
    eq : str
        The einsum expression string.
    shapes : list[tuple[int]]
        The shapes of the tensors/arrays.
    size_dict : dict[str, int]
        The dict of index sizes, only returned if ``return_size_dict=True``.

    Example
    --------
    >>> eq, shapes, size_dict = random_tensor_network(
        number_of_tensors = 10,
        regularity = 3.5,
        number_of_output_indices = 5,
        min_axis_size = 2,
        max_axis_size = 4,
        return_size_dict = True,
        global_dim = False,
        seed = 12345
    )

    >>> eq
    'gafoj,mpab,uhlbcdn,cqlipe,drstk,ve,fk,ongmq,hj,i->sturv'

    >>> shapes
    [(3, 4, 4, 2, 3),
    (3, 2, 4, 2),
    (4, 4, 2, 2, 4, 2, 3),
    (4, 2, 2, 4, 2, 2),
    (2, 4, 3, 4, 4),
    (2, 2), (4, 4),
    (2, 3, 3, 3, 2),
    (4, 3),
    (4,)]

    >>> size_dict
    {'a': 4, 'b': 2, 'c': 4, 'd': 2, 'e': 2, 'f': 4, 'g': 3, 'h': 4, 'i': 4, 'j': 3, 'k': 4, 'l': 2, 'm': 3, 'n': 3, 'o': 2, 'p': 2, 'q': 2, 'r': 4, 's': 3, 't': 4, 'u': 4, 'v': 2}
    """

    # handle inputs
    assert (
        number_of_tensors >= 0
    ), f"number_of_tensors {number_of_tensors} has to be non-negative."
    assert regularity >= 0, f"regularity {regularity} has to be non-negative."
    assert (
        number_of_output_indices >= 0
    ), f"number_of_output_indices {number_of_output_indices} has to be non-negative."
    assert min_axis_size >= 0, f"min_axis_size {min_axis_size} has to be non-negative."
    assert max_axis_size >= 0, f"max_axis_size {max_axis_size} has to be non-negative."

    # create rng
    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(seed)

    # total number of indices
    assert (
        number_of_output_indices > 0
    ) or not global_dim, f"If a global dimension is to be used, the number of output indices has to be at least 1."

    number_of_output_indices -= (
        1 * global_dim
    )  # reserve one output index for global dimension

    number_of_indices = (
        int(number_of_tensors * regularity) // 2 + number_of_output_indices
    )  # NOTE: output indices are not counted for degree.
    tensors = []
    output = []

    size_dict = {
        get_symbol(i): rng.integers(min_axis_size, max_axis_size + 1)
        for i in range(number_of_indices)
    }

    # generate TN as einsum string
    for index_number, index in enumerate(size_dict):
        # generate first two tensors connected by an edge to start with
        if index_number == 0:
            tensors.append(index)
            tensors.append(index)
            continue

        # generate a bound/edge
        if index_number < number_of_indices - number_of_output_indices:

            # add tensors and connect to existing tensors, until number of tensors is reached
            if len(tensors) < number_of_tensors:
                connect_to_tensor = rng.integers(0, len(tensors))
                tensors[connect_to_tensor] += index
                tensors.append(index)
            # add edges between existing tensors
            else:
                tensor_1 = rng.integers(0, len(tensors))
                tensor_2 = rng.integers(0, len(tensors))
                while tensor_2 == tensor_1:
                    tensor_2 = rng.integers(0, len(tensors))
                tensors[tensor_1] += index
                tensors[tensor_2] += index

        # generate an output index
        else:
            tensor = rng.integers(0, len(tensors))
            tensors[tensor] += index
            output += index

    # check specs
    assert (
        len(tensors) == number_of_tensors
    ), f"number generated tensors/tensors = {len(tensors)} does not match number_of_tensors = {number_of_tensors}."
    assert (
        len(output) == number_of_output_indices
    ), f"number of generated output indices = {len(output)} does not match number_of_output_indices = {number_of_output_indices}."
    # assert len(_find_disconnected_subgraphs([set(input) for input in tensors], set(output))) == 1, "the generated graph is not connected." # check if graph is connected

    # possibly add the same global dim to every arg
    if global_dim:
        gdim = get_symbol(number_of_indices)
        size_dict[gdim] = rng.integers(min_axis_size, max_axis_size + 1)
        for i in range(number_of_tensors):
            tensors[i] += gdim
        output += gdim

    # randomly transpose the output indices and form equation
    output = "".join(rng.permutation(output))
    tensors = ["".join(rng.permutation(list(tensor))) for tensor in tensors]
    eq = "{}->{}".format(",".join(tensors), output)

    # make the shapes
    shapes = [tuple(size_dict[ix] for ix in op) for op in tensors]

    ret = (eq, shapes)

    if return_size_dict:
        ret += (size_dict,)

    return ret
