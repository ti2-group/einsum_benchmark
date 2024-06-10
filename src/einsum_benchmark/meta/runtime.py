import importlib
import opt_einsum as oe
import numpy as np
from collections import Counter
import math
import time


def to_ssa_path(linear_path):
    """Convert a linear contraction path to the Single Static Assignment (SSA) form.

    Args:
        linear_path (list): A list of tuples representing the linear contraction path.

    Returns:
        list: A list of tuples representing the SSA form of the contraction path.

    Raises:
        RuntimeError: If the path contains negative indices.
        RuntimeError: If the path cannot be converted to SSA form.
        RuntimeError: If the path contains repeating indices within a contraction pair.
    """

    class IdVector:
        def __init__(self, num_ids, elements):
            self.buckets = []
            i = 0
            while i < num_ids - elements + 1:
                v = [j for j in range(i, i + elements)]
                self.buckets.append(v)
                i += elements
            if i < num_ids:
                v = [j for j in range(i, num_ids)]
                self.buckets.append(v)

        def get_set(self, linear_id):
            bucket = 0
            sizes = len(self.buckets[0])
            while sizes <= linear_id:
                bucket += 1
                sizes += len(self.buckets[bucket])

            idx = linear_id - (sizes - len(self.buckets[bucket]))
            ssa_idx = self.buckets[bucket][idx]
            self.buckets[bucket].pop(idx)
            if len(self.buckets[bucket]) == 0:
                self.buckets.pop(bucket)
            return ssa_idx

    num_ids = len(linear_path) * 2
    elements = 16384
    tensors = IdVector(num_ids, elements)

    ssa_path = []
    c = len(linear_path) + 1  # number of input tensors
    repeating = False

    for first, second in linear_path:
        if first < 0 or second < 0:
            raise RuntimeError(
                "ERROR: Path is incorrect, negative indices are not allowed."
            )
        if first >= c or second >= c:
            raise RuntimeError(
                "ERROR: Path is incorrect, it cannot be converted from linear to SSA form."
            )
        if first > second:
            t1 = tensors.get_set(first)
            t2 = tensors.get_set(second)
        else:
            t2 = tensors.get_set(second)
            t1 = tensors.get_set(first)
        c -= 1
        repeating |= first == second
        ssa_path.append((t1, t2))

    if repeating:
        raise RuntimeError(
            "ERROR: Repeating indices are not allowed within a contraction path pair."
        )

    return ssa_path


def get_ops_and_max_size(format_string, annotated_ssa_path, *tensors, size_dict=None):
    """Calculates the total number of operations and the maximum size of intermediate tensors.

       Calculates the total number of operations and the maximum size of intermediate tensors
       for a given format string, annotated SSA path, and input tensors.

    Args:
        format_string (str): The format string specifying the input and output tensors.
        annotated_ssa_path (list): A list of tuples representing the annotated SSA path.
        tensors (tuple): Input tensors.
        size_dict (dict, optional): A dictionary mapping characters in the format string to their sizes.

    Returns:
        tuple: A tuple containing the logarithm base 10 of the total number of operations and
               the logarithm base 2 of the maximum size of intermediate tensors.
    """
    inputs, output = format_string.split("->")
    inputs = inputs.split(",")
    if size_dict is None:
        shapes = [tensor.shape for tensor in tensors]
        size_dict = {}
        for input, shape in zip(inputs, shapes):
            for char, size in zip(input, shape):
                if char in size_dict:
                    assert size_dict[char] == size
                size_dict[char] = size
    else:
        shapes = [tuple([size_dict[char] for char in input]) for input in inputs]

    total_ops = 0
    max_size = 0

    for input in inputs:
        size = np.prod([size_dict[char] for char in input])
        # max_size = max(max_size, size)

    for first, second, expression in annotated_ssa_path:
        t1, t2 = shapes[first], shapes[second]
        _, path_info = oe.contract_path(expression, t1, t2, shapes=True)
        total_ops += path_info.opt_cost
        max_size = max(max_size, int(path_info.largest_intermediate))
        # if path_info.largest_intermediate > 17:
        # print(expression, t1, t2)
        output = expression.split("->")[1]
        output_shape = tuple([size_dict[char] for char in output])
        shapes.append(output_shape)

    return math.log10(total_ops), math.log2(max_size)


def to_annotated_ssa_path(format_string, ssa_path, is_ascii=False):
    """Annote an SSA path with their pairwise einsum format string.

    Args:
        format_string (str): The format string representing the einsum expression.
        ssa_path (list): A list of tuples representing the SSA path indices.
        is_ascii (bool, optional): Flag indicating whether to convert the annotated SSA path
            to ASCII characters. Defaults to False.

    Returns:
        list: Annotated SSA path, where each element is a tuple containing the indices and
            pairwise format_string for each step in the SSA path.
    """
    inputs, output = format_string.split("->")
    inputs = inputs.split(",")
    assert (
        len(inputs) >= 2
    ), "Einsum expressions involving just one Tensor are not supported."
    format_string = format_string.replace(" ", "")
    histogram = Counter(format_string)

    annotated_ssa_path = []
    index = 0
    for first, second in ssa_path:
        index += 1
        t1 = inputs[first]
        t2 = inputs[second]
        visited = set()
        unique_indices = []

        for char in t1 + t2:
            if char not in visited:
                unique_indices.append(char)
                visited.add(char)
            histogram[char] -= 1

        if index == len(ssa_path):
            t3 = output
        else:
            t3 = "".join(char for char in unique_indices if histogram[char] > 0)
            for char in t3:
                histogram[char] += 1

        pairwise_expression = f"{t1},{t2}->{t3}"

        if is_ascii:
            ascii_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ascii_index = 0
            char_mapping = {}
            ascii_expression_parts = []
            for char in pairwise_expression:
                if char in ",->":
                    ascii_expression_parts.append(char)
                else:
                    if char not in char_mapping:
                        if ascii_index == len(ascii_chars):
                            raise RuntimeError(
                                f"ERROR: {pairwise_expression} cannot be converted to ASCII, it is too large."
                            )
                        char_mapping[char] = ascii_chars[ascii_index]
                        ascii_index += 1
                    ascii_expression_parts.append(char_mapping[char])
            pairwise_expression = "".join(ascii_expression_parts)

        annotated_ssa_path.append((first, second, pairwise_expression))
        inputs.append(t3)
    return annotated_ssa_path


def jensum(annotated_ssa_path, *arguments, debug=False):
    """Perform a series of tensor contractions based on the annotated_ssa_path.

    Args:
        annotated_ssa_path (list): A list of tuples representing the annotated SSA path.
            Each tuple contains three elements: the indices of the tensors to contract,
            and the contraction expression.
        *arguments: Variable number of tensor arguments.
        debug (bool, optional): If True, print debug information during the contractions.

    Returns:
        The final tensor resulting from the series of contractions.

    Raises:
        AssertionError: If the number of contractions is less than 1.
        AssertionError: If the number of tensors in arguments is less than 2.
        RuntimeError: If the number of tensors in arguments contradicts the number of entries in annotated_ssa_path.
    """
    l = list(arguments)
    num_contractions = len(annotated_ssa_path)
    assert num_contractions >= 1
    assert (
        len(l) >= 2
    ), "Einsum expressions involving just one Tensor are not supported."
    if len(l) - 1 != num_contractions:
        RuntimeError(
            "ERROR: Number of tensors in arguments contradicts the number of entries in annotated_ssa_path."
        )

    i = 1

    for first, second, expression in annotated_ssa_path:
        t1, t2 = l[first], l[second]
        if debug:
            print(i, "of", num_contractions, expression)
        t3 = oe.contract(expression, t1, t2)
        l.append(t3)
        l[first] = None
        l[second] = None
        i += 1

    return l[-1]


def jensum_meta(annotated_ssa_path, *arguments, debug=False):
    """Compute the meta information and perform tensor contractions using the jensum algorithm.

    Args:
        annotated_ssa_path (list): A list of tuples representing the annotated SSA path for each contraction.
            Each tuple contains three elements: the indices of the tensors to contract,
            and the contraction expression.
        arguments (tuple): The input tensors to be contracted.
        debug (bool, optional): If True, debug information will be printed. Defaults to False.

    Returns:
        tuple: A tuple containing the final contracted tensor and a list of meta information for each contraction.
        The list of meta information contains for each step, where A and B are input tensors and C is the output tensor:
        - The number of operations (flops).
        - The sizes of the input tensors (size_A, size_B, size_C).
        - The densities of the input and output tensors (density_A, density_B, density_C).

    Raises:
        AssertionError: If the number of contractions is less than 1 or the number of tensors in arguments is less than 2.
        RuntimeError: If the number of tensors in arguments contradicts the number of entries in annotated_ssa_path.
        RuntimeError: If the density of a tensor cannot be computed due to missing size or numel() attribute.

    """
    l = list(arguments)
    num_contractions = len(annotated_ssa_path)
    assert num_contractions >= 1
    assert (
        len(l) >= 2
    ), "Einsum expressions involving just one Tensor are not supported."
    if len(l) - 1 != num_contractions:
        RuntimeError(
            "ERROR: Number of tensors in arguments contradicts the number of entries in annotated_ssa_path."
        )

    i = 1

    def compute_tensor_density(tensor):
        # if not isinstance(tensor, np.ndarray):
        #     tensor = np.asarray(tensor)
        total_elements = 0
        if hasattr(tensor, "numel"):
            total_elements = tensor.numel()
        elif hasattr(tensor, "size"):
            total_elements = tensor.size
        else:
            raise RuntimeError(
                "ERROR: Cannot compute density of tensor, neither size nor numel() are defined"
            )
        if hasattr(tensor, "count_nonzero"):
            non_zero_count = tensor.count_nonzero()
        else:
            non_zero_count = np.count_nonzero(tensor)

        if total_elements == 0:
            return 0

        if hasattr(non_zero_count, "item"):
            non_zero_count = non_zero_count.item()
        if hasattr(total_elements, "item"):
            total_elements = total_elements.item()

        density = non_zero_count / total_elements
        return density, total_elements

    def convert_density_for_print(density):
        if density < 0.00001:
            return np.format_float_scientific(density, precision=2)
        return np.round(density, 5)

    jmeta = []

    original_dtype = l[0].dtype

    for first, second, expression in annotated_ssa_path:
        t1, t2 = l[first], l[second]
        d1, size_A = compute_tensor_density(t1)
        d2, size_B = compute_tensor_density(t2)
        path, path_info = oe.contract_path(expression, t1, t2)
        flops = path_info.opt_cost
        size_C = int(path_info.largest_intermediate)
        flops_log10, size_C_log2 = math.log10(flops), math.log2(size_C)

        size_A_log2 = math.log2(size_A)
        size_B_log2 = math.log2(size_B)

        if debug:
            print(
                f"{i} of {num_contractions}: {expression} log10[OPs]: {round(flops_log10, 2)} log2[SIZE]: ({round(size_A_log2, 2)}, {round(size_B_log2, 2)}, {round(size_C_log2, 2)}) density[A, B]: ({convert_density_for_print(d1)}, {convert_density_for_print(d2)}), dtypes: {t1.dtype}, {t2.dtype}\n"
            )

        start_time = time.time()
        t3 = oe.contract(expression, t1, t2)
        end_time = time.time()
        execution_time = end_time - start_time
        if debug:
            print(f"Single contraction Execution time: {execution_time} seconds")

        d3, size_C = compute_tensor_density(t3)
        # if debug:
        #     print(
        #         f"Resulting tensor density: {convert_density_for_print(d3)}, result dtype: {t3.dtype}, max: {t3.max()}\n"
        #     )

        if t3.dtype != original_dtype:
            # cast to original dtype
            t3 = t3.type(original_dtype)
        l.append(t3)

        l[first] = None
        l[second] = None
        i += 1

        jmeta.append((int(flops), (size_A, size_B, size_C), (d1, d2, d3)))

    return l[-1], jmeta


def min_avg_density(jmeta):
    """Calculate the minimum density and average density of contractions.

    The average density is the number of non-zero entries divided by the total number of entries.

    Args:
        jmeta (list): A list of contractions, where each contraction is represented as a tuple
                      containing information about flops, sizes, and densities.

    Returns:
        tuple: A tuple containing the minimum density and average density of contractions.

    """
    min_density = math.inf
    nnz_entries = 0
    all_entries = 0
    for contraction in jmeta:
        min_density = min(min_density, contraction[2][2])
        nnz_entries += contraction[2][2] * contraction[1][2]
        all_entries += contraction[1][2]

    avg_density = nnz_entries / all_entries
    # Convert torch tensors if necessary
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.is_tensor(min_density):
            min_density = min_density.item()
        if torch.is_tensor(avg_density):
            avg_density = avg_density.item()
    return min_density, avg_density


def linear_path_runtime_meta(format_string, linear_path, *arguments, debug=False):
    """Compute the runtime metadata for a linear path.

    This functions returns all metadata as given in the metadata.csv

    Args:
        format_string (str): The format string specifying the einsum operation.
        linear_path (str): The linear path representing the einsum operation.
        *arguments: Variable length arguments for the einsum operation.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        tuple: A tuple containing the result of the einsum expression,
            the execution time, the minimum density, and the average density.

    """
    ssa_path = to_ssa_path(linear_path)
    annotated_ssa_path = to_annotated_ssa_path(
        format_string, ssa_path=ssa_path, is_ascii=True
    )
    start_time = time.time()
    result_meta, jmeta = jensum_meta(annotated_ssa_path, *arguments, debug=debug)
    end_time = time.time()
    execution_time = end_time - start_time
    min_density, avg_density = min_avg_density(jmeta)
    return result_meta, execution_time, min_density, avg_density
