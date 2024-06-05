import importlib
import opt_einsum as oe
import numpy as np
from collections import Counter
import math
import time
from opt_einsum.paths import linear_to_ssa


def get_ops_and_max_size(format_string, annotated_ssa_path, *tensors, size_dict=None):
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
    # information for each contraction: (flops, (size_A, size_B, size_C), (density_A, density_B, density_C))
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
    ssa_path = linear_to_ssa(linear_path)
    annotated_ssa_path = to_annotated_ssa_path(
        format_string, ssa_path=ssa_path, is_ascii=True
    )
    start_time = time.time()
    result_meta, jmeta = jensum_meta(annotated_ssa_path, *arguments, debug=debug)
    end_time = time.time()
    execution_time = end_time - start_time
    min_density, avg_density = min_avg_density(jmeta)
    return result_meta, execution_time, min_density, avg_density
