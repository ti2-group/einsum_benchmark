import math
from collections import Counter
import numpy as np


class Meta_Info_Instance:
    """Represents an instance of meta information for a computation.

    Args:
        tensors (int): The number of tensors involved in the computation.
        different_indices (int): The number of different indices in the computation.
        hadamard_products (int): The number of Hadamard products in the computation.
        edges (int): The number of contraction edges in the computation.
        hyperedges (int): The number of contraction hyperedges in the computation.
        tensors_in_largest_hyperedge (int): The number of tensors in the largest hyperedge in the computation.
        tensors_with_traces_or_diagonals (int): The number of tensors with traces or diagonals in the computation.
        independent_components (int): The number of independent components in the computation.
        tensors_in_largest_component (int): The number of tensors in the largest component in the computation.
        smallest_dimension_size (int): The size of the smallest dimension in the computation.
        largest_dimension_size (int): The size of the largest dimension in the computation.
        log2_output_size (float): The logarithm base 2 of the output size of the computation.

    Attributes:
        tensors (int): The number of tensors involved in the computation.
        different_indices (int): The number of different indices in the computation.
        hadamard_products (int): The number of Hadamard products in the computation.
        edges (int): The number of contraction edges in the computation.
        hyperedges (int): The number of contraction hyperedges in the computation.
        tensors_in_largest_hyperedge (int): The number of tensors in the largest hyperedge in the computation.
        tensors_with_traces_or_diagonals (int): The number of tensors with traces or diagonals in the computation.
        independent_components (int): The number of independent components in the computation.
        tensors_in_largest_component (int): The number of tensors in the largest component in the computation.
        smallest_dimension_size (int): The size of the smallest dimension in the computation.
        largest_dimension_size (int): The size of the largest dimension in the computation.
        log2_output_size (float): The logarithm base 2 of the output size of the computation.
    """

    def __init__(
        self,
        tensors,
        different_indices,
        hadamard_products,
        edges,
        hyperedges,
        tensors_in_largest_hyperedge,
        tensors_with_traces_or_diagonals,
        independent_components,
        tensors_in_largest_component,
        smallest_dimension_size,
        largest_dimension_size,
        log2_output_size,
    ):
        self.tensors = tensors
        self.different_indices = different_indices
        self.hadamard_products = hadamard_products
        self.edges = edges
        self.hyperedges = hyperedges
        self.tensors_in_largest_hyperedge = tensors_in_largest_hyperedge
        self.tensors_with_traces_or_diagonals = tensors_with_traces_or_diagonals
        self.independent_components = independent_components
        self.tensors_in_largest_component = tensors_in_largest_component
        self.smallest_dimension_size = smallest_dimension_size
        self.largest_dimension_size = largest_dimension_size
        self.log2_output_size = log2_output_size

    def __str__(self):
        info_str = (
            f"tensors: {self.tensors}\n"
            f"different_indices: {self.different_indices}\n"
            f"hadamard_products: {self.hadamard_products}\n"
            f"contraction_edges: {self.edges}\n"
            f"contraction_hyperedges: {self.hyperedges}\n"
            f"tensors_in_largest_hyperedge: {self.tensors_in_largest_hyperedge}\n"
            f"tensors_with_traces_or_diagonals: {self.tensors_with_traces_or_diagonals}\n"
            f"independent_components: {self.independent_components}\n"
            f"tensors_in_largest_component: {self.tensors_in_largest_component}\n"
            f"smallest_dimension_size: {self.smallest_dimension_size}\n"
            f"largest_dimension_size: {self.largest_dimension_size}\n"
            f"log2(output_size): {format(self.log2_output_size, '.2f')}"
        )
        return info_str


def compute_meta_info_of_einsum_instance(format_string, tensors):
    """Compute meta information for an einsum instance.

    Args:
        format_string (str): The einsum format string.
        tensors (list): A list of input tensors.

    Returns:
        Meta_Info_Instance: An instance of the Meta_Info_Instance class containing the computed meta information.
            It has the following Attributes:
                - tensors (int): The number of tensors involved in the computation.
                - different_indices (int): The number of different indices in the computation.
                - hadamard_products (int): The number of Hadamard products in the computation.
                - edges (int): The number of contraction edges in the computation.
                - hyperedges (int): The number of contraction hyperedges in the computation.
                - tensors_in_largest_hyperedge (int): The number of tensors in the largest hyperedge in the computation.
                - tensors_with_traces_or_diagonals (int): The number of tensors with traces or diagonals in the computation.
                - independent_components (int): The number of independent components in the computation.
                - tensors_in_largest_component (int): The number of tensors in the largest component in the computation.
                - smallest_dimension_size (int): The size of the smallest dimension in the computation.
                - largest_dimension_size (int): The size of the largest dimension in the computation.
                - log2_output_size (float): The logarithm base 2 of the output size of the computation.


    """
    number_of_tensors = len(tensors)
    format_string = format_string.replace(" ", "")
    str_in, str_out = format_string.split("->")
    inputs_str = str_in.split(",")
    inputs = []
    number_of_tensors_with_traces_or_diagonals = 0
    for s in inputs_str:
        processed = set(s)
        inputs.append(frozenset(processed))
        if len(processed) < len(s):
            number_of_tensors_with_traces_or_diagonals += 1
    output = frozenset(set(str_out))
    unique = set()
    for t in inputs:
        unique.add(t)
    unique = list(unique)
    histogram = Counter()
    for t in unique:
        histogram.update(t - output)
    number_of_hadamard_products = number_of_tensors - len(unique)
    number_of_different_indices = len(histogram)
    number_of_different_indices += len(output)

    number_of_edges = sum(value > 1 for value in histogram.values())
    number_of_hyperedges = sum(value > 2 for value in histogram.values())
    edges = {key: [] for key in histogram.keys()}
    number_of_tensors_in_largest_hyperedge = 0
    if number_of_hyperedges > 0:
        number_of_tensors_in_largest_hyperedge = max(
            value for value in histogram.values()
        )
    subgraphs = []
    for key, t in enumerate(unique):
        if len(t) == 0:
            subgraphs.append([key])
        else:
            no_output = t - output
            if len(no_output) == 0:
                subgraphs.append([key])
            else:
                for c in no_output:
                    edges[c].append(key)
    for key in edges.keys():
        if len(edges[key]) == 0:
            continue
        subgraph = []
        inserted = set()
        q = []
        q.extend(edges[key])
        i = 0
        edges[key] = []
        while i < len(q):
            if q[i] not in inserted:
                inserted.add(q[i])
                subgraph.append(q[i])
                for c in unique[q[i]] - output:
                    q.extend(edges[c])
                    edges[c] = []
            i += 1
        subgraphs.append(subgraph)

    number_of_independent_components = len(subgraphs)

    number_of_tensors_in_largest_component = max(len(graph) for graph in subgraphs)

    def _find_smallest_largest_dimension_sizes(arrays):
        smallest_dim = float("inf")
        largest_dim = 0
        for item in arrays:
            dims = []
            if isinstance(item, np.ndarray):
                dims = item.shape
            elif isinstance(item, list):
                dims = item
            elif isinstance(item, tuple):
                dims = item
            elif np.isscalar(item):
                dims = []
                shapes.append([])
            else:
                try:
                    dims = item.shape
                except Exception as e:
                    print(f"An error occurred: {e}")
            if dims:
                smallest_dim = min(smallest_dim, *dims)
                largest_dim = max(largest_dim, *dims)
            else:
                smallest_dim = min(smallest_dim, 1)
                largest_dim = max(largest_dim, 1)

        smallest_dim = smallest_dim if smallest_dim != float("inf") else 0
        return smallest_dim, largest_dim

    smallest_dimension_size, largest_dimension_size = (
        _find_smallest_largest_dimension_sizes(tensors)
    )

    output_size = 1

    if len(output) > 0:

        def _get_sizes(einsum_notation, shapes):
            index_sizes = {}
            for einsum_index, shape in zip(
                einsum_notation.split("->")[0].split(","), shapes
            ):
                if not hasattr(shape, "__iter__"):
                    shape = list(shape)
                for index, dimension in zip(einsum_index, shape):
                    if not index in index_sizes:
                        index_sizes[index] = dimension
                    else:
                        if index_sizes[index] != dimension:
                            raise Exception(f"Dimension error for index '{index}'.")
            return index_sizes

        shapes = []
        for arg in tensors:
            if isinstance(arg, np.ndarray):
                shapes.append(arg.shape)
            elif isinstance(arg, list):
                shapes.append(arg)
            elif isinstance(arg, tuple):
                shapes.append(arg)
            elif np.isscalar(arg):
                shapes.append([])
            else:
                try:
                    shapes.append(arg.shape)
                except Exception as e:
                    print(f"An error occurred: {e}")
        sizes = _get_sizes(format_string, shapes)

        for c in output:
            output_size *= sizes[c]

    meta_info_for_instance = Meta_Info_Instance(
        number_of_tensors,
        number_of_different_indices,
        number_of_hadamard_products,
        number_of_edges,
        number_of_hyperedges,
        number_of_tensors_in_largest_hyperedge,
        number_of_tensors_with_traces_or_diagonals,
        number_of_independent_components,
        number_of_tensors_in_largest_component,
        smallest_dimension_size,
        largest_dimension_size,
        math.log2(output_size),
    )
    return meta_info_for_instance
