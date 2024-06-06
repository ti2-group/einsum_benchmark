import pooch
import opt_einsum as oe
import numpy as np
import math


def get_file_paths():
    POOCH = pooch.create(
        path=pooch.os_cache("einsum_benchmark"),
        base_url="doi:10.5281/zenodo.11477304",
        registry=None,
    )

    # Automatically populate the registry
    POOCH.load_registry_from_doi()

    # Fetch one of the files in the repository
    fname = POOCH.fetch(
        "instances.zip",
        progressbar=True,
        processor=pooch.Unzip(extract_dir="instances"),
    )

    # print(fname)
    sorted_fname = sorted(fname)
    return sorted_fname


def compute_oe_path_from_arrays(format_string, arrays):
    # compute example path
    path, path_info = oe.contract_path(format_string, *arrays)
    return path, path_info


def compute_oe_path_from_shapes(format_string, shapes):
    # generate arrays based on shapes
    arrays = [np.random.rand(*shape) for shape in shapes]
    return compute_oe_path_from_arrays(format_string, arrays)


def print_oe_path_metrics(path_info):
    flops_log10 = round(math.log10(path_info.opt_cost), 2)
    size_log2 = round(math.log2(path_info.largest_intermediate), 2)
    print("log10[FLOPs]:", flops_log10)
    print("log2[SIZE]:", size_log2)
