import importlib.util
import subprocess
from typing import Optional, Literal, List, Hashable, Dict, Tuple
from .runtime import (
    jensum,
    jensum_meta,
    to_annotated_ssa_path,
    linear_path_runtime_meta,
    get_ops_and_max_size,
)
from .info import compute_meta_info_of_einsum_instance

__all__ = [
    "find_path",
    "jensum",
    "jensum_meta",
    "to_annotated_ssa_path",
    "linear_path_runtime_meta",
    "get_ops_and_max_size",
    "compute_meta_info_of_einsum_instance",
]

Inputs = List[List[Hashable]]
Output = List[Hashable]
SizeDict = Dict[Hashable, int]
Path = List[Tuple[int, ...]]


def find_path(
    format_string: str,
    *tensors,
    minimize: Literal["flops", "size"],
    n_trials: int = 128,
    n_jobs: int = 10,
    show_progress_bar: bool = True,
    timeout: Optional[int] = None,
):
    if importlib.util.find_spec("kahypar") is None:
        print("kahypar is not installed. Please install kahypar and cgreedy")
        return None
    if importlib.util.find_spec("cgreedy") is None:
        print("cgreedy is not installed")
        return None

    try:
        subprocess.run(
            ["g++", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["gcc", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print(
            "g++ or gcc is not installed. Please install g++ and gcc to compile the extension."
        )
        return None

    from . import path_finder

    inputs, output = format_string.split("->")
    inputs = inputs.split(",")

    shapes = [tensor.shape for tensor in tensors]
    size_dict = {}
    for input, shape in zip(inputs, shapes):
        for char, size in zip(input, shape):
            if char in size_dict:
                assert size_dict[char] == size
            size_dict[char] = size

    return path_finder.hyper_optimized_hhg(
        inputs,
        output,
        size_dict,
        minimize,
        n_trials,
        n_jobs,
        show_progress_bar,
        timeout,
    )
