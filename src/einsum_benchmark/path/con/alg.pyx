# distutils: language = c++
# distutils: extra_compile_args = -ffast-math -Ofast -std=c++17 -march=native -fopenmp -static-libgcc -static-libstdc++ -Wno-unused-function -Wno-address
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: embedsignature = True
# distutils: extra_link_args = -fopenmp -w

from libc.stdint cimport uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool

import opt_einsum as oe
import time

cdef extern from "cpath.h":
    cppclass Expression[T]:
        Expression()

        Expression(unordered_map[T, uint64_t] index_to_size_map)

        void add(vector[T] & input_vector)

        void simplify()

    cppclass Path_Pair:
        uint64_t a
        uint64_t b
        Path_Pair()

    cppclass Path:
        vector[Path_Pair] ssa
        vector[Path_Pair] linear

    cppclass Metrics:
        double log10_flops
        double max_log2_size

    cppclass GreedyResult:
        Path path
        Metrics metrics

    cdef enum Minimize:
        FLOPS,
        INTERMEDIATE_SIZE

    GreedyResult greedy_exec[T](const Expression[T] & expression, uint64_t seed, uint64_t max_repeats, double max_time, bool progbar, Minimize minimize, bool is_outer_optimal, uint32_t threshold_optimal, const unsigned int num_threads, const bool generate_linear)

class CGreedy(oe.paths.PathOptimizer):
    def __init__(self, seed = 0, minimize='size', max_repeats = 16, max_time = 0.0, progbar = False, is_outer_optimal = False, uint32_t threshold_optimal = 12, uint32_t threads = 0, bool is_linear = True):
        """
         Initialize the CGreedy optimizer.

         Parameters:
         ----------
         seed : int, optional
             Random seed for reproducibility. Default is 0.

         minimize : str, optional
             Criterion to minimize. Either 'size' or 'flops'. Default is 'size'.

         max_repeats : int, optional
             Maximum number of times the optimization can be repeated. Default is 16.

         max_time : float, optional
             Maximum time (in seconds) the optimizer is allowed to run. If set to 0.0 or less,
             there's no time limit. Default is 0.0.

         progbar : bool, optional
             Whether to display a progress bar during optimization. Default is False.

         is_outer_optimal: bool, optional
             Whether to consider outer products in the optimal search. Default is False.

         threshold_optimal: uint, optional
             Maximum number of input tensors to perform an expensive optimal search instead
             of a greedy search. Default is 12.
             
         threads: uint, optional
             Number of threads to be used for the greedy algorithm. Default is 0. 
             (Setting the value to 0 uses all available threads.)

         Raises:
         ------
         Exception:
             If the 'minimize' parameter is not either 'size' or 'flops'.

         Attributes:
         -----------
         flops_log10 : float
             Log base 10 of the number of flops (floating-point operations). Initialized to negative infinity.
             This value is updated after the algorithm is executed.

         size_log2 : float
             Log base 2 of the biggest intermediate tensor size. Initialized to negative infinity.
             This value is updated after the algorithm is executed.

         path_time : float
             Time (in seconds) used internally to compute the contraction path. Initialized to negative infinity.
             This value is updated after the algorithm is executed.

         """
        self.seed = seed
        self.max_repeats = max_repeats
        self.max_time = max_time
        self.progbar = progbar
        if minimize in {"size", "flops"}:
            self.minimize = minimize
        else:
            raise Exception("ERROR: minimize parameter can only be 'size' or 'flops'.")
        if threshold_optimal < 4 or threshold_optimal > 64:
            raise Exception("ERROR: valid input for 'threshold_optimal' is a number between 3 and 64.")
        self.threshold_optimal = threshold_optimal
        self.is_outer_optimal = is_outer_optimal
        self.minimize = minimize
        self.threads = threads
        self.is_linear = is_linear
        self.flops_log10 = float("-inf")
        self.size_log2 = float("-inf")
        self.path_time = float("-inf")

    def __call__(self, inputs, output, sizes, memory_limit=None):
        cdef:
            unordered_map[uint32_t, uint64_t] index_to_size_map
            uint64_t i
            uint64_t value
            vector[uint32_t] v
            Expression[uint32_t] expression
            set pset
            str c
            GreedyResult result
            list linear_path = []
            Path_Pair pp
            Minimize minimize

        if len(inputs) == 1:
            return 0,
        elif len(inputs) == 2:
            return 0, 1

        cdef bool is_print_cpp = False # set to true to print the tensor network as C++ code
        if is_print_cpp: print()
        if is_print_cpp: print()
        if is_print_cpp: print("std::unordered_map<uint32_t, uint64_t> index_to_size{", end = "")
        for c, value in sizes.items():
            if is_print_cpp: print("{", ord(c), ",", value, "},", end = "")
            index_to_size_map[ord(c)] = value
        if is_print_cpp: print("};")
        if is_print_cpp: print("Expression<uint32_t> expression(index_to_size);")

        expression = Expression[uint32_t](index_to_size_map)

        for pset in inputs:
            v.clear()
            for c in sorted(pset):
                v.push_back(ord(c))
            expression.add(v)
            if is_print_cpp: print(
            "expression.add(std::vector<uint32_t>{" + ",".join([str(ord(c)) for c in sorted(pset)]) + "});", end = "")
        v.clear()
        for c in sorted(output):
            v.push_back(ord(c))
        if is_print_cpp: print(
                    "expression.add(std::vector<uint32_t>{" + ",".join([str(ord(c)) for c in sorted(output)]) + "});")
        expression.add(v)

        if is_print_cpp: quit()

        expression.simplify()

        if self.minimize == "flops":
            minimize = Minimize.FLOPS
        elif self.minimize == "size":
            minimize = Minimize.INTERMEDIATE_SIZE
        else:
            raise Exception("ERROR: minimize parameter can only be 'size' or 'flops'.")

        tic = time.time()
        result = greedy_exec[uint32_t](expression, self.seed, self.max_repeats, self.max_time, self.progbar, minimize, self.is_outer_optimal, self.threshold_optimal, self.threads, self.is_linear)
        toc = time.time()

        if self.is_linear:
            for i in range(result.path.linear.size()):
                linear_path.append((result.path.linear[i].a, result.path.linear[i].b))
        else:
            for i in range(result.path.ssa.size()):
                linear_path.append((result.path.ssa[i].a, result.path.ssa[i].b))



        self.path_time = toc - tic
        self.size_log2 = result.metrics.max_log2_size
        self.flops_log10 = result.metrics.log10_flops
        return linear_path
