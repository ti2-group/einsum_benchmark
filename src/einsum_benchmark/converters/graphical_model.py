import numpy as np
import opt_einsum as oe


def parse_evidence_file(file_path):
    with open(file_path + ".evid", "r") as file:
        line = file.readline()

    parts = list(map(int, line.split()))
    num_observed_variables = parts.pop(0)
    observed_variables = {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}

    assert num_observed_variables == len(
        observed_variables
    ), "The number of observed variables and the number of provided observations do not match."

    return observed_variables


def parse_uai_file(file_path):
    with open(file_path, "r") as file:
        lines = file.read().splitlines()

    # Parse preamble
    graph_type = lines[0]
    num_variables = int(lines[1])
    variable_cardinalities = list(map(int, lines[2].split()))
    assert num_variables == len(
        variable_cardinalities
    ), "The number of variables and the number of provided cardinalities do not match."
    num_functions = int(lines[3])
    function_scopes = [
        list(map(int, line.split()[1:])) for line in lines[4 : 4 + num_functions]
    ]

    # Parse function tables
    function_tables = []

    line_index = 4 + num_functions
    # Skip blank line after preamble (sometimes its missing)
    if line_index < len(lines) and not lines[line_index]:
        line_index += 1

    while line_index < len(lines):
        table_values = []
        if len(lines[line_index].split()) == 1:
            num_table_values = int(lines[line_index])
            line_index += 1
        else:
            splitted = lines[line_index].split()
            num_table_values = int(splitted[0])
            table_values.extend(map(float, splitted[1:]))
            line_index += 1

        while len(table_values) < num_table_values:
            table_values.extend(map(float, lines[line_index].split()))
            line_index += 1

        shape = [
            variable_cardinalities[i] for i in function_scopes[len(function_tables)]
        ]
        # get product of all cardinalities in the scope
        assert num_table_values == np.prod(
            shape
        ), f"Table values do not match the scope. {line_index} {num_table_values} {len(function_tables)} {function_scopes[len(function_tables)]} {np.prod(shape)} \n {lines[line_index]} "
        # Reshape the table values according to the function scope
        function_tables.append(np.array(table_values).reshape(shape))

        if line_index < len(lines) and not lines[line_index]:
            line_index += 1

    tensor_indices = [
        [oe.get_symbol(var) for var in scope] for scope in function_scopes
    ]

    # parse evidence file
    observed_variables = parse_evidence_file(file_path)
    return {
        "graph_type": graph_type,
        "num_variables": num_variables,
        "variable_cardinalities": variable_cardinalities,
        "num_functions": num_functions,
        "function_scopes": function_scopes,
        "tensors": function_tables,
        "tensor_indices": tensor_indices,
        "format_string": ",".join(["".join(indices) for indices in tensor_indices]),
        "observed_variables": observed_variables,
    }
