from .model_counting import dimacs_to_einsum
from .graphical_model import parse_uai_file


def tn_from_uai_file(file_path: str) -> tuple[str, list]:
    """
    Parses a UAI file and returns the einsum format string and tensors representing the graphical model.

    Args:
        file_path (str): The path to the UAI file.

    Returns:
        tuple[str, list]: A tuple containing the format string and a list of tensors.
    """
    parsed_file = parse_uai_file(file_path)
    return parsed_file["format_string"], parsed_file["tensors"]


def tn_from_dimacs_file(file_path: str, clause_split_threshold=3) -> tuple[str, list]:
    """
    Converts a DIMACS (weighted) model counting file to a tensor network.

    Args:
        file_path (str): The path to the DIMACS file.
        clause_split_threshold (int, optional): The threshold for splitting clauses. Defaults to 3.

    Returns:
        tuple[str, list]: A tuple containing the equation and the list of tensors.
    """
    num_vars, num_clauses, found_vars, eq, size_dict, tensors = dimacs_to_einsum(
        filepath=file_path, clause_split_threshold=clause_split_threshold
    )
    return eq, tensors
