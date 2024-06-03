import importlib.util
import subprocess
from typing import Optional, Literal, List, Hashable, Dict, Tuple

Inputs = List[List[Hashable]]
Output = List[Hashable]
SizeDict = Dict[Hashable, int]
Path = List[Tuple[int, ...]]


def find(
    inputs: Inputs,
    output: Output,
    size_dict: SizeDict,
    minimize: Literal["flops", "size"],
    n_trials: int = 128,
    n_jobs: int = 10,
    show_progress_bar: bool = True,
    timeout: Optional[int] = None,
):
    if importlib.util.find_spec("kahypar") is None:
        print(
            "kahypar is not installed. Please install kahypar, cython and g++ to use the path finder"
        )
        return None
    if importlib.util.find_spec("cython") is None:
        print(
            "Cython is not installed. Please install Cython, and g++ to compile the extension."
        )
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
