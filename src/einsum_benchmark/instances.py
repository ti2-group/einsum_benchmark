import os
from .util import get_file_paths
import pickle
from typing import Any, NamedTuple, List, Tuple

PathMeta = NamedTuple(
    "PathMeta",
    [
        ("path", List[Tuple[int, int]]),
        ("size", float),
        ("flops", float),
        ("min_density", float),
        ("avg_density", float),
    ],
)

PathMetas = NamedTuple(
    "PathMetas",
    [
        ("opt_size", PathMeta),
        ("opt_flops", PathMeta),
    ],
)

BenchMarkInstance = NamedTuple(
    "BenchMarkInstance",
    [
        ("format_string", str),
        ("tensors", List[Any]),
        ("paths", PathMetas),
        ("result_sum", Any),
        ("name", str),
    ],
)


class InstanceFiles:
    def __init__(self):
        self._file_paths = None
        self._path_by_file_name = None

    def _get_file_paths(self):
        if self._file_paths is None:
            self._file_paths = get_file_paths()
            self._path_by_file_name = {
                os.path.splitext(os.path.basename(p))[0]: p for p in self._file_paths
            }
        return self._file_paths

    def _get_file_path_by_file_name(self, file_name):
        if self._path_by_file_name is None:
            self._get_file_paths()
        return self._path_by_file_name[file_name]

    def _load_file(self, file_path):
        with open(file_path, "rb") as f:
            instance = pickle.load(f)

        (
            format_string,
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

        size_path = PathMeta(
            path=size_linear_path,
            size=size_path_size,
            flops=size_path_flops,
            min_density=size_min_density,
            avg_density=size_avg_density,
        )
        flops_path = PathMeta(
            path=flops_linear_path,
            size=flops_path_size,
            flops=flops_path_flops,
            min_density=flops_min_density,
            avg_density=flops_avg_density,
        )

        path_metas = PathMetas(
            opt_size=size_path,
            opt_flops=flops_path,
        )

        named_instance = BenchMarkInstance(
            format_string=format_string,
            tensors=tensors,
            paths=path_metas,
            result_sum=result_sum,
            name=os.path.splitext(os.path.basename(file_path))[0],
        )

        return named_instance

    def __getitem__(self, file_name):
        file_path = self._get_file_path_by_file_name(file_name)
        return self._load_file(file_path)

    def __iter__(self):
        for file_path in self._get_file_paths():
            yield self._load_file(file_path)

    def values(self):
        return self.__iter__()

    def items(self):
        for file_path in self._get_file_paths():
            yield file_path, self._load_file(file_path)

    def keys(self):
        self._get_file_paths()
        return sorted(self._path_by_file_name.keys())

    def __len__(self):
        return len(self._get_file_paths())

    def __contains__(self, file_name):
        return file_name in self.keys()
