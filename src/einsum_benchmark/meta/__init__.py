import importlib.util
import subprocess
from typing import Optional, Literal, List, Hashable, Dict, Tuple
from .runtime import (
    to_ssa_path,
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
    "to_ssa_path",
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
    """Optimize a path for evaluating an einsum expression.

    Args:
        format_string (str): The Einstein summation notation expression.
        *tensors: The input tensors.
        minimize (Literal["flops", "size"]): The objective to minimize, either "flops" or "size".
        n_trials (int, optional): The number of trials for the optimization process. Defaults to 128.
        n_jobs (int, optional): The number of parallel jobs to run. Defaults to 10.
        show_progress_bar (bool, optional): Whether to show a progress bar during optimization. Defaults to True.
        timeout (int, optional): The maximum time in seconds for the optimization process. Defaults to None.

    Returns:
        str: An ssa path for evaluating the einsum expression.
    """
    if (
        importlib.util.find_spec("kahypar") is None
        or importlib.util.find_spec("cgreedy") is None
        or importlib.util.find_spec("optuna") is None
    ):
        raise ImportError(
            """You need to install the optional dependencies for path to use this function
            
            You can do this with pip install "einsum_benchmark[path]"
            """
        )

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
