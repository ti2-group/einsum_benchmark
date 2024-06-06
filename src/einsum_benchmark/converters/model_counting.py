import sys
import numpy as np
import opt_einsum


class WmcDimacsParser:
    """
    This classes provides a method for parsing a file in dimacs weighted cnf
    format into a benchmark datapoint representing the tensor network
    corresponding to the weighted cnf.
    """

    def __init__(self, max_dim=3):
        """
        file_format: concrete type of dimacs format, see enum DimacsFormat
        max_dim: highest allowed number of indices in one clause, memory usage per clause is O(2^d) with d number of indices in clause
        """
        self.index = 0
        self.dimacs = None
        self.inputs = []
        self.tensors = []
        self.max_dim = max_dim
        self.largest_var_idx_used = 0
        self.found_vars = set()
        self.added_weights = set()
        self.total_tensor_size = 0

    def parse(self, filepath: str):
        with open(filepath, "r") as file:
            text = file.read()
            text = self.__remove_comments(text)
            self.dimacs = text.split()  # split by whitespace
        self.index = 0
        self.inputs = []
        self.tensors = []
        self.found_vars = set()
        self.total_tensor_size = 0

        num_vars, num_clauses = self.__read_metadata()
        if num_vars + 141 > 1114111:
            raise Exception("Number of variables too large")
        self.largest_var_idx_used = num_vars + 1

        while self.index < len(self.dimacs):
            if self.__is_weight_line():
                self.__parse_weight()  # current line describes a weight, not a clause
            else:
                self.__parse_clause()
            if self.total_tensor_size > 10 * 1024 * 1024 * 1024:
                raise Exception(
                    f"Overall size of tensors in {filepath} is greater than 10GB"
                )
        size_dict = {i: 2 for t in self.inputs for i in t}
        return (
            num_vars,
            num_clauses,
            self.found_vars,
            self.inputs,
            size_dict,
            self.tensors,
        )

    def __remove_comments(self, text):
        lines = text.splitlines()
        lines = [line for line in lines if not self.__is_comment(line)]
        return "\n".join(lines)

    def __is_comment(self, line):
        return line.startswith("c") and not line.startswith("c p")

    def __read_metadata(self) -> (int, int):
        while self.dimacs[self.index] != "p":
            self.index += 1  # Skip any extra lines to get to the metadata
        num_vars = int(self.dimacs[self.index + 2])
        num_clauses = int(self.dimacs[self.index + 3])
        self.index += 4
        return num_vars, num_clauses

    def __is_weight_line(self) -> bool:
        is_weight_line = (
            self.dimacs[self.index] == "c"
            and self.dimacs[self.index + 1] == "p"
            and self.dimacs[self.index + 2] == "weight"
        )
        if is_weight_line:
            self.index += 3
        return is_weight_line

    def __parse_weight(self) -> None:
        variable_index = int(
            self.dimacs[self.index]
        )  # index of the variable being weighted
        self.found_vars.add(abs(variable_index))
        weight = float(
            self.dimacs[self.index + 1]
        )  # weight being attributed to the variable
        if variable_index > 0:  # non-negated weight -> add weight tensor
            self.inputs.append(opt_einsum.get_symbol(variable_index))
            self.tensors.append(np.array([1 - weight, weight], dtype=np.float64))
            self.total_tensor_size += sys.getsizeof(self.tensors[-1])
            self.added_weights.add(abs(variable_index))
        else:
            if abs(variable_index) not in self.added_weights:
                assert (
                    False
                ), "Negative weight came first, needs to be implemented? Positive might be missing"
            pass  # ignore negated weights, since we add the negated weight along with the non-negated one

        self.index += 3  # these formats have an extra 0 at the end of the line

    def __parse_clause(self) -> None:
        clause = ()  # start a new clause
        index_tuple = ()  # index tuple for clause
        index_clause = []
        while self.dimacs[self.index] != "0":
            variable_index = int(self.dimacs[self.index])
            self.found_vars.add(abs(variable_index))
            clause += (opt_einsum.get_symbol(abs(variable_index)),)
            index_tuple += (1 if variable_index < 0 else 0,)
            index_clause.append(variable_index)
            self.index += 1
        if len(index_tuple) <= self.max_dim:
            self.inputs.append(clause)
            self.tensors.append(self.__create_clause_tensor(index_tuple))
            self.total_tensor_size += sys.getsizeof(self.tensors[-1])
        else:
            inputs, _, tensors = self.__map_clause_to_small_tensors(
                index_clause, self.__free_variable()
            )
            self.inputs += inputs
            self.tensors += tensors
            self.total_tensor_size += sum(sys.getsizeof(tensor) for tensor in tensors)
        self.index += 1

    def __create_clause_tensor(self, index_tuple: tuple[int]) -> np.ndarray:
        if len(index_tuple) > self.max_dim:
            raise Exception(
                f"Clause with {len(index_tuple)} indices encountered, maximum is {self.max_dim}."
            )
        tensor = np.ones([2] * len(index_tuple), dtype=np.float64)
        tensor[index_tuple] = 0
        return tensor

    def __map_clause_to_small_tensors(
        self, clause: list[int], merge_variable: int
    ) -> tuple[list[int], str, list[np.ndarray]]:
        tensors = []
        indices = []
        output = ""
        merge_index = opt_einsum.get_symbol(merge_variable)
        tensors.append(np.asarray([-1, 1], dtype=np.float64))
        indices.append(merge_index)
        for var in clause:
            index = opt_einsum.get_symbol(abs(var))
            output += index
            tensor = np.ones((2, 2), dtype=np.float64)
            if var >= 0:
                tensor[1, 0] = 0
            else:
                tensor[0, 0] = 0
            tensors.append(tensor)
            indices.append(index + merge_index)
        return indices, output, tensors

    def __free_variable(self) -> int:
        largest_used_so_far = self.largest_var_idx_used
        self.largest_var_idx_used += 1
        return largest_used_so_far


def dimacs_to_einsum(
    filepath: str, clause_split_threshold=3
) -> tuple[int, int, set, str, dict, list[np.ndarray]]:
    """
    Converts a DIMACS file to a tensor network representation.

    Args:
        filepath (str): The path to the DIMACS file.
        clause_split_threshold (int, optional): The maximum dimension for splitting clauses. Defaults to 3.

    Returns:
        tuple[int, int, str, dict, list[np.ndarray]]: The number of variables, the number of clauses, the einsum string, the size dictionary, and the tensors.
    """
    parser = WmcDimacsParser(max_dim=clause_split_threshold)
    num_vars, num_clauses, found_vars, inputs, size_dict, tensors = parser.parse(
        filepath
    )

    eq = ",".join(["".join(input) for input in inputs]) + "->"
    return num_vars, num_clauses, found_vars, eq, size_dict, tensors
