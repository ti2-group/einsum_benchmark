<!-- markdownlint-disable -->

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/random/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `generators.random`




---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/random/randreg.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `regular`

```python
def regular(n, reg, d_min=2, d_max=3, seed=None)
```

Create a random contraction equation that corresponds to a random regular graph. 

The graph must not be connected and for large graphs it will very likely have several components 



**Args:**
 
 - <b>`n`</b> (int):  The number of terms. 
 - <b>`reg`</b> (int):  The degree of the graph. 
 - <b>`d_min`</b> (int, optional):  The minimum size of an index. 
 - <b>`d_max`</b> (int, optional):  The maximum size of an index. 
 - <b>`seed`</b> (None or int, optional):  Seed for `networkx` and `np.random.default_rng` for repeatability. 



**Returns:**
 
 - <b>`Tuple[List[List[str]], List[str], List[Tuple[int]], Dict[str, int]]`</b>:  The inputs, output, shapes, and size dictionary. 



**Example:**
 ```python
 format_string, shapes = randreg_equation(n=100, reg=3, d_min=2, d_max=4, seed=None)
```



---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/random/connected_hypernetwork.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `connected_hypernetwork`

```python
def connected_hypernetwork(
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
    return_size_dict: bool = False
) → Union[Tuple[str, Collection[Tuple[int, ]], Dict[str, int]], Tuple[str, Collection[Tuple[int, ]]]]
```

Generate a random connected Hyper Tensor Network (HTN). 

Returns an einsum expressions string representing the HTN, shapes of the tensors and optionally a dictionary containing the index sizes. 



**Args:**
 
 - <b>`number_of_tensors`</b> (int):  Number of tensors/arrays in the TN. 
 - <b>`regularity`</b> (float):  'Regularity' of the TN. This determines how  many indices/axes each tensor shares with others on average (not counting output indices, global dimensions, self edges and single summation indices). 
 - <b>`max_tensor_order`</b> (int, optional):  The maximum order (number of axes/dimensions) of the tensors. If ``None``, use an upper bound calculated from other parameters. 
 - <b>`max_edge_order`</b> (int, optional):  The maximum order of hyperedges. 
 - <b>`diagonals_in_hyper_edges`</b> (bool, optional):  Whether diagonals can appear in hyper edges, e.g. in "aab,ac,ad -> bcd" a is a hyper edge with a diagonal in the first tensor. 
 - <b>`number_of_output_indices`</b> (int, optional):  Number of output indices/axes (i.e. the number of non-contracted indices) including global dimensions. Defaults to 0 if global_dim = False, i.e., a contraction resulting in a scalar, and to 1 if global_dim = True. 
 - <b>`max_output_index_order`</b> (int, optional):  Restricts the number of times the same output index can occur. 
 - <b>`diagonals_in_output_indices`</b> (bool, optional):  Whether diagonals can appear in output indices, e.g. in "aab,ac -> abc" a is an output index with a diagonal in the first tensor. 
 - <b>`number_of_self_edges`</b> (int, optional):  The number of self edges/traces (e.g. in "ab,bcdd->ac" d represents a self edge). 
 - <b>`max_self_edge_order`</b> (int, optional):  The maximum order of a self edge e.g. in "ab,bcddd->ac" the self edge represented by d has order 3. 
 - <b>`number_of_single_summation_indices`</b> (int, optional):  The number of indices that are not connected to any other tensors and do not show up in the ouput (e.g. in "ab,bc->c" a is a single summation index). 
 - <b>`min_axis_size`</b> (int, optional):  Minimum size of an axis/index (dimension) of the tensors. 
 - <b>`max_axis_size`</b> (int, optional):  Maximum size of an axis/index (dimension) of the tensors. 
 - <b>`seed`</b> (int, optional):  If not None, seed numpy's random generator with this. 
 - <b>`global_dim`</b> (bool, optional):  Add a global, 'broadcast', dimension to every operand. 
 - <b>`return_size_dict`</b> (bool, optional):  Return the mapping of indices to sizes. 



**Returns:**
 
 - <b>`Tuple[str, List[Tuple[int]], Optional[Dict[str, int]]]`</b>:  The einsum expression string, the shapes of the tensors/arrays, and the dict of index sizes (only returned if ``return_size_dict=True``). 



**Examples:**
 'usual' Tensor Hyper Networks 

```python
 eq, shapes, size_dict = random_tensor_hyper_network(
```
         number_of_tensors=10,
         regularity=2.5,
         max_tensor_order=10,
         max_edge_order=5,
         number_of_output_indices=5,
         min_axis_size=2,
         max_axis_size=4,
         return_size_dict=True,
         seed=12345
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
         number_of_tensors=10,
         regularity=2.5,
         max_tensor_order=5,
         max_edge_order=6,
         number_of_output_indices=5,
         max_output_index_order=3,
         number_of_self_edges=4,
         max_self_edge_order=3,
         number_of_single_summation_indices=3,
         global_dim=True,
         min_axis_size=2,
         max_axis_size=4,
         seed=12345
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
         number_of_tensors=10,
         regularity=3.0,
         max_tensor_order=10,
         max_edge_order=3,
         diagonals_in_hyper_edges=True,
         number_of_output_indices=5,
         max_output_index_order=3,
         diagonals_in_output_indices=True,
         number_of_self_edges=4,
         max_self_edge_order=3,
         number_of_single_summation_indices=3,
         global_dim=True,
         min_axis_size=2,
         max_axis_size=4,
         seed=12345
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



---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/random/connected_network.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `connected_network`

```python
def connected_network(
    number_of_tensors: int,
    regularity: float,
    number_of_output_indices: int = 0,
    min_axis_size: int = 2,
    max_axis_size: int = 10,
    seed: Optional[int] = None,
    global_dim: bool = False,
    return_size_dict: bool = False
) → Union[Tuple[str, Collection[Tuple[int, ]], Dict[str, int]], Tuple[str, Collection[Tuple[int, ]]]]
```

Generate a random connected Tensor Network (TN). 

Returns an einsum expressions string representing the TN, shapes of the tensors and optionally a dictionary containing the index sizes. 

Parameters 
---------- number_of_tensors : int  Number of tensors/arrays in the TN. regularity : float  'Regularity' of the TN. This determines how  many indices/axes each tensor shares with others on average (not counting output indices and a global dimension). number_of_output_indices : int, optional  Number of output indices/axes (i.e. the number of non-contracted indices) including the global dimension.  Defaults to 0 in case of no global dimension, i.e., a contraction resulting in a scalar, and to 1 in case there is a global dimension. min_axis_size : int, optional  Minimum size of an axis/index (dimension) of the tensors. max_axis_size : int, optional  Maximum size of an axis/index (dimension) of the tensors. seed: int, optional  If not None, seed numpy's random generator with this. global_dim : bool, optional  Add a global, 'broadcast', dimension to every operand. return_size_dict : bool, optional  Return the mapping of indices to sizes. 

Returns 
------- eq : str  The einsum expression string. shapes : list[tuple[int]]  The shapes of the tensors/arrays. size_dict : dict[str, int]  The dict of index sizes, only returned if ``return_size_dict=True``. 

Example 
-------- ```python
 eq, shapes, size_dict = random_tensor_network(
```
     number_of_tensors = 10,
     regularity = 3.5,
     number_of_output_indices = 5,
     min_axis_size = 2,
     max_axis_size = 4,
     return_size_dict = True,
     global_dim = False,
     seed = 12345
)

```python
 eq
``` 'gafoj,mpab,uhlbcdn,cqlipe,drstk,ve,fk,ongmq,hj,i->sturv' 

```python
 shapes
```
[(3, 4, 4, 2, 3),
(3, 2, 4, 2),
(4, 4, 2, 2, 4, 2, 3),
(4, 2, 2, 4, 2, 2),
(2, 4, 3, 4, 4),
(2, 2), (4, 4),
(2, 3, 3, 3, 2),
(4, 3),
(4,)]

```python
 size_dict
``` {'a': 4, 'b': 2, 'c': 4, 'd': 2, 'e': 2, 'f': 4, 'g': 3, 'h': 4, 'i': 4, 'j': 3, 'k': 4, 'l': 2, 'm': 3, 'n': 3, 'o': 2, 'p': 2, 'q': 2, 'r': 4, 's': 3, 't': 4, 'u': 4, 'v': 2} 


