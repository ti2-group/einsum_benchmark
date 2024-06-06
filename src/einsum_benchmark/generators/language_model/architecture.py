from collections import Counter

from .symbols import SymbolGenerator


# TODO: name the observables in a way that they are sorted in the order of the context


def build_mera_structure(mera_depth: int) -> list[list[str]]:
    """Builds the structure (the index string layers) for a MERA with the given depth.

    Parameters
    ----------
    mera_depth : int
        The number of layers with 4th-order tensors.

    Returns
    -------
    list[list[str]]
        Layers of index strings which defines the tensor network structure.
    """

    assert mera_depth >= 0, "MERA depth must be non-negative."
    symbol_generator = SymbolGenerator()
    layers: list[list[str]] = []
    current_lowest_symbols = ""
    # layer 0 only has 1 matrix
    first_layer = [symbol_generator.generate(2)]
    layers.append(first_layer)
    current_lowest_symbols = "".join(first_layer)
    # for every layer, create 3rd-order tensors and 4th-order tensors
    for _ in range(mera_depth):
        # create 3rd-order tensors at every axis hanging off the bottom
        third_order_output_symbols = [
            symbol_generator.generate(2) for _ in current_lowest_symbols
        ]
        third_order_layer = [
            input_symbol + output_symbols
            for input_symbol, output_symbols in zip(
                current_lowest_symbols, third_order_output_symbols
            )
        ]
        layers.append(third_order_layer)
        current_lowest_symbols = "".join(third_order_output_symbols)
        # create 4th-order tensors between the 3rd-order tensors - note that the first and last symbols stay the same
        fourth_order_input_symbols = [
            current_lowest_symbols[2 * i - 1 : 2 * i + 1]
            for i in range(1, len(current_lowest_symbols) // 2)
        ]
        fourth_order_output_symbols = [
            symbol_generator.generate(2) for _ in fourth_order_input_symbols
        ]
        fourth_order_layer = [
            input_symbols + output_symbols
            for input_symbols, output_symbols in zip(
                fourth_order_input_symbols, fourth_order_output_symbols
            )
        ]
        layers.append(fourth_order_layer)
        current_lowest_symbols = (
            current_lowest_symbols[0]
            + "".join(fourth_order_output_symbols)
            + current_lowest_symbols[-1]
        )
    # now rename the lowest symbols such that, when sorted by symbol, they are in the order of the context
    sorted_observables = sorted(current_lowest_symbols)
    replacements = {
        symbol: new_symbol
        for symbol, new_symbol in zip(current_lowest_symbols, sorted_observables)
    }
    for layer in layers[-2:]:
        for i, index_string in enumerate(layer):
            layer[i] = "".join(
                [
                    replacements[symbol] if symbol in replacements else symbol
                    for symbol in index_string
                ]
            )
    return layers


def assign_axis_sizes_per_layer(
    index_string_layers: list[list[str]], layer_axis_sizes: list[int]
) -> dict[str, int]:
    """For each axis layer, assign one axis size to all axes in that layer.

    Parameters
    ----------
    index_string_layers : list[list[str]]
        Layers of index strings which defines the tensor network structure.
    layer_axis_sizes : list[int]
        List of one axis size for each layer of axes.

    Returns
    -------
    dict[str, int]
        Size dict, maps each symbol to the size of the corresponding axis.
    """

    assert len(index_string_layers) == len(
        layer_axis_sizes
    ), f"The number of layers ({len(index_string_layers)}) must match the number of axis sizes ({len(layer_axis_sizes)})."
    size_dict = {}
    already_seen_symbols: set[str] = set()
    # from the top downwards, collect all symbols we have never seen before, and assign the axis size of the layer it is found in.
    # then, add them to the already seen symbols
    for layer, size in zip(index_string_layers, layer_axis_sizes):
        new_symbols = [
            symbol
            for index_string in layer
            for symbol in index_string
            if symbol not in already_seen_symbols
        ]
        for symbol in new_symbols:
            size_dict[symbol] = size
        already_seen_symbols |= set(new_symbols)
    # special exception: for the observables, assign the last axis size - there might be observables which are not in the last layer, now we overwrite their axis sizes
    # observables are symbols that occur only once in the whole network
    all_symbols = "".join(
        [
            symbol
            for layer in index_string_layers
            for index_string in layer
            for symbol in index_string
        ]
    )
    symbol_occurences = Counter(all_symbols)
    observables = [symbol for symbol, count in symbol_occurences.items() if count == 1]
    for observable in observables:
        size_dict[observable] = layer_axis_sizes[-1]
    return size_dict
