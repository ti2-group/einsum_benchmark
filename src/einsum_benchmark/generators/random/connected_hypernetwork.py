from typing import Union, Optional, Tuple, Dict, Collection
from numpy.random import default_rng
#from opt_einsum.paths import _find_disconnected_subgraphs #NOTE: only for testing

from generators.util import get_symbol

PathType = Collection[Tuple[int, ...]]

def random_tensor_hyper_network(
    
    number_of_tensors: int,
    regularity: float,
    max_tensor_order: int = None,
    max_edge_order: int = 2,
    diagonals_in_hyper_edges: bool = False,
    number_of_output_indices: int = 0,
    max_output_index_order: int = 1,
    diagonals_in_output_indices: bool = False,
    number_of_self_edges: int = 0,
    max_self_edge_order: int = 2,
    number_of_single_summation_indices: int = 0,
    global_dim: bool = False,
    min_axis_size: int = 2,
    max_axis_size: int = 10,
    seed: Optional[int] = None,
    return_size_dict: bool = False,
    ) -> Union[Tuple[str, PathType, Dict[str, int]], Tuple[str, PathType]]:

    """Generate a random contraction and shapes.

    Parameters
    ----------
    number_of_tensors : int
        Number of tensors/arrays in the TN.
    regularity : float
        'Regularity' of the TN. This determines how
        many indices/axes each tensor shares with others on average (not counting output indices, global dimensions, self edges and single summation indices).
    max_tensor_order: int = None, optional
        The maximum order (number of axes/dimensions) of the tensors. If ``None``, use an upper bound calculated from other parameters.
    max_edge_order: int, optional
        The maximum order of hyperedges.
    diagonals_in_hyper_edges: bool = False,
        Whether diagonals can appear in hyper edges, e.g. in "aab,ac,ad -> bcd" a is a hyper edge with a diagonal in the first tensor.
    number_of_output_indices : int, optional
        Number of output indices/axes (i.e. the number of non-contracted indices) including global dimensions.
        Defaults to 0 if global_dim = False, i.e., a contraction resulting in a scalar, and to 1 if global_dim = True.
    max_output_index_order: int = 1, optional
        Restricts the number of times the same output index can occur.
    diagonals_in_output_indices: bool = False,
        Whether diagonals can appear in output indices, e.g. in "aab,ac -> abc" a is an output index with a diagonal in the first tensor.
    number_of_self_edges: int = 0, optional
        The number of self edges/traces (e.g. in "ab,bcdd->ac" d represents a self edge).
    max_self_edge_order: int = 2, optional
        The maximum order of a self edge e.g. in "ab,bcddd->ac" the self edge represented by d has order 3.
    number_of_single_summation_indices: int = 0, optional
        The number of indices that are not connected to any other tensors and do not show up in the ouput (e.g. in "ab,bc->c" a is a single summation index).
    min_axis_size : int, optional
        Minimum size of an axis/index (dimension) of the tensors.
    max_axis_size : int, optional
        Maximum size of an axis/index (dimension) of the tensors.
    seed: int, optional
        If not None, seed numpy's random generator with this.
    global_dim : bool, optional
        Add a global, 'broadcast', dimension to every operand.
    return_size_dict : bool, optional
        Return the mapping of indices to sizes.

    Returns
    -------
    eq : str
        The einsum expression string.
    shapes : list[tuple[int]]
        The shapes of the tensors/arrays.
    size_dict : dict[str, int]
        The dict of index sizes, only returned if ``return_size_dict=True``.

    Examples
    --------
    'usual' Tensor Hyper Networks
    >>> eq, shapes, size_dict = random_tensor_hyper_network(
        number_of_tensors = 10
        regularity = 2.5
        max_tensor_order = 10
        max_edge_order = 5
        number_of_output_indices = 5
        min_axis_size = 2
        max_axis_size = 4
        return_size_dict = True,
        seed = 12345
    )
    >>> eq
    'bdca,abhcdg,cbmd,cfd,ed,e,figj,gl,h,nik->jnmkl'

    >>> shapes
    [(2, 2, 2, 2), 
    (2, 2, 4, 2, 2, 3), 
    (2, 2, 4, 2), 
    (2, 2, 2), 
    (2, 2), 
    (2,), 
    (2, 4, 3, 3), 
    (3, 2), 
    (4,), 
    (3, 4, 3)]

    >>> size_dict
    {'a': 2, 'b': 2, 'c': 2, 'd': 2, 'e': 2, 'f': 2, 'g': 3, 'h': 4, 'i': 4, 'j': 3, 'k': 3, 'l': 2, 'm': 4, 'n': 3}

    Tensor Hyper Networks with self edges (of higher order), single summation indices, output indices of higher order and a global dimension
    >>> eq, shapes = random_tensor_hyper_network(
        number_of_tensors = 10, 
        regularity = 2.5, 
        max_tensor_order = 5, 
        max_edge_order = 6,
        number_of_output_indices = 5, 
        max_output_index_order = 3,
        number_of_self_edges = 4, 
        max_self_edge_order = 3, 
        number_of_single_summation_indices = 3, 
        global_dim = True,
        min_axis_size = 2, 
        max_axis_size = 4,
        seed = 12345
    )
    >>> eq
    'caxpp,afxeb,nbkxn,jdkxc,tdqxv,hxgre,jlxfi,xsgmm,howxo,xuijl->utvwx'

    >>> shapes
    [(2, 4, 4, 3, 3), 
    (4, 3, 4, 3, 3), 
    (3, 3, 2, 4, 3), 
    (2, 2, 2, 4, 2), 
    (2, 2, 2, 4, 3), 
    (4, 4, 2, 3, 3), 
    (2, 2, 4, 3, 3), 
    (4, 3, 2, 3, 3), 
    (4, 2, 2, 4, 2), 
    (4, 2, 3, 2, 2)]

    Tensor Hyper Networks as above but with diagonals in hyper edges and output indices
    >>> eq, shapes = random_tensor_hyper_network(
        number_of_tensors = 10, 
        regularity = 3.0, 
        max_tensor_order = 10, 
        max_edge_order = 3,
        diagonals_in_hyper_edges = True,
        number_of_output_indices = 5, 
        max_output_index_order = 3,
        diagonals_in_output_indices = True, 
        number_of_self_edges = 4, 
        max_self_edge_order = 3, 
        number_of_single_summation_indices = 3, 
        global_dim = True,
        min_axis_size = 2, 
        max_axis_size = 4, 
        seed = 12345
    )
    >>> eq
    'cabxk,gkegax,wldxbrb,ctoxdfo,xvdlv,weehx,nfnkx,spgpixqu,xjimhm,ijx->uvwtx'

    >>> shapes
    [(3, 2, 4, 3, 2), 
    (2, 2, 3, 2, 2, 3), 
    (4, 4, 3, 3, 4, 3, 4), 
    (3, 4, 3, 3, 3, 3, 3), 
    (3, 3, 3, 4, 3), 
    (4, 3, 3, 2, 3), 
    (4, 3, 4, 2, 3), 
    (3, 3, 2, 3, 2, 3, 2, 2), 
    (3, 4, 2, 2, 2, 2), 
    (2, 4, 3)]
    """

    # handle inputs
    assert number_of_tensors >= 0, f"number_of_tensors {number_of_tensors} has to be non-negative."
    assert regularity >= 0, f"regularity {regularity} has to be non-negative."
    assert max_tensor_order >= 0, f"max_tensor_order {max_tensor_order} has to be non-negative."
    assert max_edge_order >= 0, f"max_edge_order {max_edge_order} has to be non-negative."
    assert number_of_output_indices >= 0, f"number_of_output_indices {number_of_output_indices} has to be non-negative."
    assert min_axis_size >= 0, f"min_axis_size {min_axis_size} has to be non-negative."
    assert max_axis_size >= 0, f"max_axis_size {max_axis_size} has to be non-negative."

    # handle 'None' in tensors
    if max_tensor_order == None:
        max_tensor_order = int((number_of_tensors - 1) * regularity + number_of_self_edges * max_self_edge_order + number_of_output_indices * max_output_index_order + number_of_single_summation_indices) # in the worst case, everything gets attached to one tensor

    # check if tensors make sense
    assert regularity <= max_tensor_order, 'regularity cannot be higher than chosen max_tensor_order.'

    # handle global dim
    assert (number_of_output_indices > 0) or not global_dim, f"If a global dimension is to be used, the number of output indices has to be at least 1."
    number_of_output_indices -= 1 * global_dim # reserve one output index for global dimension
    max_tensor_order -= 1 * global_dim # reserve one spot for global dim in every tensor 

    # check if max_tensor_order suffices to fit all connecting edge, output indices, self edges and single summation indices
    assert max_tensor_order * number_of_tensors >= int(regularity * number_of_tensors) + number_of_output_indices + number_of_self_edges * 2 + number_of_single_summation_indices, f"the (max_tensor_order - 1 * global_dim) * number_of_tensors =  {max_tensor_order * number_of_tensors} is not high enough to fit all {int(regularity*number_of_tensors)} connecting indices, {number_of_output_indices} output_indices, {2 * number_of_self_edges} indices of self_edges and {number_of_single_summation_indices} single summation indices." 

    # create rng
    if seed is None:
        rng = default_rng()
    else: 
        rng = default_rng(seed)
    
    number_of_connecting_indices = int(number_of_tensors * regularity) # how many indices make up the underlying hypergraph. To this hyperedges contribute += order, These do not contribute: self edges, summation/single contr. and out edges
    number_of_spaces = number_of_tensors * max_tensor_order # number of spaces (total number of indices that can be placed in tensors such that the max order is satisfied) that are not filled and not reserved
    number_of_reserved_spaces = number_of_connecting_indices + 2 * number_of_self_edges + number_of_single_summation_indices + number_of_output_indices # how many spaces are at least neccessary to fulfil the given specifications 
    non_reserved_spaces = number_of_spaces - number_of_reserved_spaces
    
    number_of_connecting_indices_to_do = number_of_connecting_indices # keep track of how may connections are left to do
    tensors = []
    output = ""
    not_max_order_tensors = [] # keeps track of existing tensors to which indices can be added to
    free_spaces_in_not_max_order_tensors = 0 # tracks how many spaces are free in not_max_order_tensors

    # ADD ALL TENSORS such that they are connected to the graph with (hyper-)edges

    # create start tensors
    tensors.append(get_symbol(0))
    tensors.append(get_symbol(0))
    number_of_connecting_indices_to_do -= 2 # placed two indices
    not_max_order_tensors.append(0)
    not_max_order_tensors.append(1)
    number_of_reserved_spaces -= 2 # placed two indices
    free_spaces_in_not_max_order_tensors += 2 * (max_tensor_order - 1) # one index in both tensors

    for tensor_number in range(2, number_of_tensors):
        index = get_symbol(tensor_number - 1)

        # determine order of hyperedge
        number_of_tensors_to_do = number_of_tensors - tensor_number
        
        # determine max order
        if diagonals_in_hyper_edges:
            max_order = min(free_spaces_in_not_max_order_tensors + max_tensor_order, max_edge_order, number_of_connecting_indices_to_do - 2 * (number_of_tensors_to_do - 1), 2 + non_reserved_spaces) # can only connect as many times to not_max_order_tensors, as there are free spaces, need to respect the max edge order, how many connections we can still do and how many spaces are not reserved
        else:
            max_order = min(len(not_max_order_tensors) + 1, max_edge_order, number_of_connecting_indices_to_do - 2 * (number_of_tensors_to_do - 1), 2 + non_reserved_spaces) # we can only connect to existing tensors which do not have the max order, respect the max edge order, need to make sure we have enough indices left for the other tensors and need to respect the number of not reserved spaces

        order = rng.integers(2, max_order + 1)

        # determine tensors to connect to
        if diagonals_in_hyper_edges:
            for index_number in range(order):
                # fist connect to already existing tensors
                if index_number == 1:
                    tensors.append(index)
                    not_max_order_tensors.append(len(tensors) - 1)
                    continue
                tensor = rng.choice(not_max_order_tensors) #NOTE: we can get diagonals over one tensor in a hyperedge
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
        else:
            # connect to other tensors
            connect_to_tensors = rng.choice(not_max_order_tensors, size = order - 1, replace = False, shuffle = False)
            for tensor in connect_to_tensors:
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
            # connect to new tensor
            tensors.append(index)
            not_max_order_tensors.append(len(tensors) - 1)
        
        # update tracking
        number_of_connecting_indices_to_do -= order
        #number_of_reserved_spaces -= order # took care of one tensor
        #number_of_spaces -= order
        non_reserved_spaces -= (order - 2) # those spaces were taken in addition to neccessary/reserved spaces
        free_spaces_in_not_max_order_tensors += max_tensor_order - order # added one new tensor but filled order spaces

    assert len(tensors) == number_of_tensors, f"The number of created tensors/tensors = {len(tensors)} does not match number_of_tensors = {number_of_tensors}."

    # REMAINING CONNECTIONS between tensors
    number_of_used_indices = number_of_tensors - 1

    while number_of_connecting_indices_to_do > 0:
        index = get_symbol(number_of_used_indices)

        # determine order of hyperedge:
        if diagonals_in_hyper_edges:
            max_order = min(free_spaces_in_not_max_order_tensors, max_edge_order, number_of_connecting_indices_to_do) # can only fill free spaces in tensors that do not have max order, need to respect the max edge order, how many connections we can still do and how many spaces are not reserved
        else:
            max_order = min(len(not_max_order_tensors), max_edge_order, number_of_connecting_indices_to_do) # can only connect to tensors that do not have max order, need to respect the max edge order, how many connections we can still do and how many spaces are not reserved

        order = rng.integers(2, max_order + 1)
        # make sure that number_of_connecting indices to do is not left at 1
        while number_of_connecting_indices_to_do - order == 1:
            order = rng.integers(2, max_order + 1)

        # determine tensors to connect to
        if order == 2: # no pure self edges here
            connect_to_tensors = rng.choice(not_max_order_tensors, size = 2, replace = False, shuffle = False)
            for tensor in connect_to_tensors:
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
        elif diagonals_in_hyper_edges:
            for _ in range(order):
                tensor = rng.choice(not_max_order_tensors) #NOTE: we can get diagonals over one tensor in a hyperedge
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
        else:
            # connect to tensors
            connect_to_tensors = rng.choice(not_max_order_tensors, size = order, replace = False, shuffle = False)
            for tensor in connect_to_tensors:
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)

        # update tracking
        number_of_connecting_indices_to_do -= order
        number_of_used_indices += 1
        #number_of_spaces -= order
        #number_of_reserved_spaces -= order
        free_spaces_in_not_max_order_tensors -= order # filled order spaces

    # check if all connections have been made
    assert number_of_connecting_indices_to_do == 0, f"The number of created connections = {number_of_connecting_indices-number_of_connecting_indices_to_do} does not fit regularity * number_of_tensors = {regularity * number_of_tensors}."

    # SELF EDGES
    for _ in range(number_of_self_edges):
        index = get_symbol(number_of_used_indices)

        # determine order of self edge:
        max_order = min(max_self_edge_order, 2 + non_reserved_spaces) # respect max order self edge and number of non-reserved spaces
        order = rng.integers(2, max_order + 1)

        # determine tensor for self edge
        tensor = rng.choice(not_max_order_tensors)

        # make sure the tensor has enough spaces left
        while len(tensors[tensor]) + order > max_tensor_order:
            order = rng.integers(2, max_order + 1)
            tensor = rng.choice(not_max_order_tensors)
        
        tensors[tensor] += index * order
            
        # check if max order is reached
        if len(tensors[tensor]) == max_tensor_order:
            not_max_order_tensors.remove(tensor)

        # update tracking
        #number_of_reserved_spaces -= 2 # took care of one self edge
        #number_of_spaces -= order
        non_reserved_spaces -= (order - 2)
        number_of_used_indices += 1
        free_spaces_in_not_max_order_tensors -= order # filled order spaces


    # SINGLE SUMMATION INDICES
    for _ in range(number_of_single_summation_indices):
        index = get_symbol(number_of_used_indices)

        tensor = rng.choice(not_max_order_tensors)
        tensors[tensor] += index

        # check if max order is reached
        if len(tensors[tensor]) == max_tensor_order:
            not_max_order_tensors.remove(tensor)

        # update tracking
        #number_of_reserved_spaces -= 1 # took care of one single summation index
        #number_of_spaces -= 1
        number_of_used_indices += 1
        free_spaces_in_not_max_order_tensors -= 1 # filled 1 space

        
    # OUTPUT INDICES
    for output_index_number in range(1, number_of_output_indices + 1):
        index = get_symbol(number_of_used_indices)

        # determine order of output index:
        
        if diagonals_in_output_indices:
            max_order = min(max_output_index_order, free_spaces_in_not_max_order_tensors, 1 + non_reserved_spaces) # can only fill free spaces in tensors that do not have max order, need to respect the max edge order, how many connections we can still do and how many spaces are not reserved
        else:
            max_order = min(max_output_index_order, len(not_max_order_tensors), 1 + non_reserved_spaces) # respect max order of output index, number of free spaces in tensors and number of output indices left to do (non_reserved_spaces)

        order = rng.integers(1, max_order + 1)

        # determine tensors to connect to
        output += index
        if diagonals_in_output_indices:
            for _ in range(order):
                tensor = rng.choice(not_max_order_tensors) #NOTE:we can get diagonals over one tensor in an output index
                tensors[tensor] += index
                
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
        else:
            connect_to_tensors = rng.choice(not_max_order_tensors, size = order, replace = False, shuffle = False)
            for tensor in connect_to_tensors:
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)

        # update tracking
        number_of_used_indices += 1
        #number_of_reserved_spaces -= 1 # took care of one output index
        #number_of_spaces -= order
        non_reserved_spaces -= (order - 1)
        free_spaces_in_not_max_order_tensors -= order # filled order spaces

    # GLOBAL DIMENSION

    # NOTE: this can now be added to every tensor, as we reserved a spot in every tensor right from the start
    # possibly add the same global dim to every tensor
    if global_dim:
        gdim = get_symbol(number_of_used_indices)
        for i in range(number_of_tensors):
            tensors[i] += gdim
        output += gdim
        number_of_used_indices += 1

    # check length of output and that all specifications are fulfilled
    #assert number_of_reserved_spaces == 0, f"{number_of_reserved_spaces} spaces are still reserved."
    assert len(output) == (number_of_output_indices + 1 * global_dim)

    # randomly shuffle outputs and tensors
    output = "".join(rng.permutation(list(output)))

    # Test if hypergraph is connected #NOTE connected in opt einsum sense means shared output indices is not a connection. In the sense of cotengra's Hypergraph this would be a connection.
    # assert len(_find_disconnected_subgraphs([set(input) for input in tensors], set(output))) == 1, f"the generated hypergraph has {len(_find_disconnected_subgraphs([set(input) for input in tensors], set(output)))} components." #TODO comment out later

    tensors = ["".join(rng.permutation(list(input))) for input in tensors]
    # form equation
    eq = "{}->{}".format(",".join(tensors), output)

    # get random size for an index
    size_dict = {get_symbol(index): rng.integers(min_axis_size, max_axis_size + 1) for index in range(number_of_used_indices)}

    # make the shapes
    shapes = [tuple(size_dict[idx] for idx in op) for op in tensors]

    ret = (eq, shapes)

    if return_size_dict:
        return ret + (size_dict,)
    else:
        return ret