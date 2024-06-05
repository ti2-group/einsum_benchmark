import os
import numpy as np
import pickle
from einsum_benchmark.meta import compute_meta_info_of_einsum_instance
from opt_einsum.paths import ssa_to_linear

import pandas as pd

directory = "./instances"
files = []
for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        files.append(filename)

print(len(files))

merged_metadata = []

for filename in files:
    print(filename)
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, "rb") as file:
            instance = pickle.load(file)

        (
            eq,
            tensors,
            path_metas,
            result_sum,
        ) = instance

        (size_path_meta, flops_path_meta) = path_metas
        (
            size_linear_path,
            size_path_size,
            size_path_flops,
            size_min_density,
            size_avg_density,
        ) = size_path_meta
        (
            flops_linear_path,
            flops_path_size,
            flops_path_flops,
            flops_min_density,
            flops_avg_density,
        ) = flops_path_meta

        meta_info = compute_meta_info_of_einsum_instance(eq, tensors)

        data = {}
        data["filename"] = filename[:-4]
        data.update(meta_info.__dict__)

        data["file_size_in_mb"] = os.path.getsize(file_path) / 1024 / 1024
        data["dtype"] = tensors[0].dtype
        data["opt_flops_path_size_log2"] = flops_path_size
        data["opt_flops_path_flops_log10"] = flops_path_flops
        data["opt_flops_min_density"] = flops_min_density
        data["opt_flops_avg_density"] = flops_avg_density
        data["opt_size_path_size_log2"] = size_path_size
        data["opt_size_path_flops_log10"] = size_path_flops
        data["opt_size_min_density"] = size_min_density
        data["opt_size_avg_density"] = size_avg_density
        data["sum_output"] = result_sum
        merged_metadata.append(data)
    except Exception as e:
        print(f"Error processing file {filename}")
        print(e)

df = pd.DataFrame(merged_metadata)
df.to_csv("metadata.csv", index=False)
df.to_excel("metadata.xlsx", index=False)
