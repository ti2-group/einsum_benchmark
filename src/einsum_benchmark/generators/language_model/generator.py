from collections import Counter

from .architecture import build_mera_structure, assign_axis_sizes_per_layer
from .symbols import SymbolGenerator, remaining_symbols


def gen_structure_and_shapes(
    mera_depth: int, hidden_axis_size: int, observable_axis_size: int
) -> tuple[list[str], list[tuple[int, ...]]]:
    """Builds the structure of a MERA tensor network and the corresponding shapes of the included tensors.

    Parameters
    ----------
    mera_depth : int
        Number of layers with 4th-order tensors.
    hidden_axis_size : int
        Domain size of hidden variables.
    observable_axis_size : int
        Domain size of observable variables.

    Returns
    -------
    tuple[list[str], list[tuple[int, ...]]]
        Index strings and shapes for the corresponding tensors in the same order as the index strings.
    """

    layer_sizes = [hidden_axis_size] * (2 * mera_depth) + [observable_axis_size]
    index_string_layers = build_mera_structure(mera_depth)
    size_dict = assign_axis_sizes_per_layer(index_string_layers, layer_sizes)
    flat_index_strings = [
        index_string
        for index_string_layer in index_string_layers
        for index_string in index_string_layer
    ]
    shapes = [
        tuple(size_dict[symbol] for symbol in index_string)
        for index_string in flat_index_strings
    ]
    return flat_index_strings, shapes


def build_p_first_and_last(
    index_strings: list[str], shapes: list[tuple[int, ...]]
) -> tuple[str, list[tuple[int, ...]]]:
    """Build an expression to get the distribution of the first and last token of the context in the model.

    Parameters
    ----------
    index_strings : list[str]
        The index strings that define the network structure.
    shapes : list[tuple[int, ...]]
        The shapes of the tensors in the network. Should have the same order as the index strings.

    Returns
    -------
    tuple[str, list[tuple[int, ...]]]
        The einsum format string, which is needed to compute the distribution, and the shapes of its arguments.
    """

    # get all observables (in the order they appear in the context)
    symbol_counts = Counter(
        [symbol for index_string in index_strings for symbol in index_string]
    )
    observables = sorted(
        [symbol for symbol, count in symbol_counts.items() if count == 1]
    )
    # choose the first and last observable as the output
    output_string = f"{observables[0]}{observables[-1]}"
    # for all hidden variables, we need 2 symbols each, because we are basically squaring the quantum state: if x_a = sum_bc y_abc, then x_a^2 = (sum_bc y_abc)(sum_bc y_abc) = sum_bc sum_de y_abc y_ade -> we needed to introduce new indices d and e
    symbol_generator = SymbolGenerator(remaining_symbols(symbol_counts.keys()))
    hidden_variables = [symbol for symbol, count in symbol_counts.items() if count > 1]
    replacements = {symbol: symbol_generator.generate() for symbol in hidden_variables}
    # replace hidden symbols with the same symbols - for convenience
    replacements |= {observable: observable for observable in observables}
    second_index_strings = [
        "".join([replacements[symbol] for symbol in index_string])
        for index_string in index_strings
    ]
    # build an expression to get the distribution the model represents
    input_strings = index_strings + second_index_strings
    format_string = f"{','.join(input_strings)}->{output_string}"
    argument_shapes = shapes + shapes
    return format_string, argument_shapes


def build_sliding_likelihood(
    index_strings: list[str], shapes: list[tuple[int, ...]], batch_size: int
) -> tuple[str, list[tuple[int, ...]]]:
    """Build an expression to get the likelihood of the model on an imaginary batch of training data.

    Parameters
    ----------
    index_strings : list[str]
        The index strings that define the network structure.
    shapes : list[tuple[int, ...]]
        The shapes of the tensors in the network. Should have the same order as the index strings.
    batch_size : int
        Number contexts windows to compute the likelihood for.

    Returns
    -------
    tuple[str, list[tuple[int, ...]]]
        The einsum format string, which is needed to compute the batch likelihood, and the shapes of its arguments.
    """

    # get all observables (in the order they appear in the context)
    symbol_counts = Counter(
        [symbol for index_string in index_strings for symbol in index_string]
    )
    observables = sorted(
        [symbol for symbol, count in symbol_counts.items() if count == 1]
    )
    # for all hidden variables, we need 2 symbols each, because we are basically squaring the quantum state: if x_a = sum_bc y_abc, then x_a^2 = (sum_bc y_abc)(sum_bc y_abc) = sum_bc sum_de y_abc y_ade -> we needed to introduce new indices d and e
    symbol_generator = SymbolGenerator(remaining_symbols(symbol_counts.keys()))
    hidden_variables = [symbol for symbol, count in symbol_counts.items() if count > 1]
    replacements = {symbol: symbol_generator.generate() for symbol in hidden_variables}
    # replace hidden symbols with the same symbols - for convenience
    replacements |= {observable: observable for observable in observables}
    second_index_strings = [
        "".join([replacements[symbol] for symbol in index_string])
        for index_string in index_strings
    ]
    # build an expression to get the batch likelihood of the model on the training data
    batch_symbol = symbol_generator.generate()
    token_batch_index_strings = [
        f"{batch_symbol}{observable}" for observable in observables
    ]
    format_string = f"{','.join(index_strings)},{','.join(second_index_strings)},{','.join(token_batch_index_strings)},{','.join(token_batch_index_strings)}->{batch_symbol}"
    # now we need to build the shapes of the token batches we don't have the size dict information here, so we build it from the shapes
    size_dict = {
        symbol: size
        for index_string, shape in zip(index_strings, shapes)
        for symbol, size in zip(index_string, shape)
    }
    token_batch_shapes = [
        (batch_size, size_dict[observable]) for observable in observables
    ]
    argument_shapes = shapes + shapes + token_batch_shapes + token_batch_shapes
    return format_string, argument_shapes


def generate_benchmark_p_first_and_last(
    mera_depth: int, axis_size_hidden: int, axis_size_observable: int
) -> tuple[str, list[tuple[int, ...]]]:
    """Generates an einsum query and shape arguments for computing the distribution of the first and last observable in a model with the given parameters.

    Parameters
    ----------
    mera_depth : int
        Number of layers in a MERA network with 4th-order tensors.
    axis_size_hidden : int
        Domain size of hidden variables.
    axis_size_observable : int
        Domain size of observable variables.

    Returns
    -------
    tuple[str, list[tuple[int, ...]]]
        The einsum format string, which is needed to compute the distribution, and the shapes of its arguments.
    """

    index_strings, shapes = gen_structure_and_shapes(
        mera_depth, axis_size_hidden, axis_size_observable
    )
    return build_p_first_and_last(index_strings, shapes)


def generate_benchmark_sliding_likelihood(
    mera_depth: int, axis_size_hidden: int, axis_size_observable: int, batch_size: int
) -> tuple[str, list[tuple[int, ...]]]:
    """Generates an einsum query and shape arguments for computing the likelihood of the model on an imaginary batch of training data.

    Parameters
    ----------
    mera_depth : int
        Number of layers in a MERA network with 4th-order tensors.
    axis_size_hidden : int
        Domain size of hidden variables.
    axis_size_observable : int
        Domain size of observable variables.
    batch_size : int
        Number of context windows to compute the likelihood for.

    Returns
    -------
    tuple[str, list[tuple[int, ...]]]
        The einsum format string, which is needed to compute the batch likelihood, and the shapes of its arguments.
    """

    index_strings, shapes = gen_structure_and_shapes(
        mera_depth, axis_size_hidden, axis_size_observable
    )
    return build_sliding_likelihood(index_strings, shapes, batch_size)


def main():
    # example usage
    format_string, argument_shapes = generate_benchmark_p_first_and_last(
        mera_depth=1, axis_size_hidden=3, axis_size_observable=11
    )
    print(format_string, argument_shapes)
    format_string, argument_shapes = generate_benchmark_sliding_likelihood(
        mera_depth=1, axis_size_hidden=3, axis_size_observable=11, batch_size=100
    )
    print(format_string, argument_shapes)


if __name__ == "__main__":
    main()
