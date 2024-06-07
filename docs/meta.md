<!-- markdownlint-disable -->

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `meta`




---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/__init__.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `find_path`

```python
def find_path(
    format_string: str,
    *tensors,
    minimize: Literal['flops', 'size'],
    n_trials: int = 128,
    n_jobs: int = 10,
    show_progress_bar: bool = True,
    timeout: Optional[int] = None
)
```

Optimize a path for evaluating an einsum expression. 



**Args:**
 
 - <b>`format_string`</b> (str):  The Einstein summation notation expression. 
 - <b>`*tensors`</b>:  The input tensors. 
 - <b>`minimize`</b> (Literal["flops", "size"]):  The objective to minimize, either "flops" or "size". 
 - <b>`n_trials`</b> (int, optional):  The number of trials for the optimization process. Defaults to 128. 
 - <b>`n_jobs`</b> (int, optional):  The number of parallel jobs to run. Defaults to 10. 
 - <b>`show_progress_bar`</b> (bool, optional):  Whether to show a progress bar during optimization. Defaults to True. 
 - <b>`timeout`</b> (int, optional):  The maximum time in seconds for the optimization process. Defaults to None. 



**Returns:**
 
 - <b>`str`</b>:  An ssa path for evaluating the einsum expression. 


---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/info.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `compute_meta_info_of_einsum_instance`

```python
def compute_meta_info_of_einsum_instance(format_string, tensors)
```

Compute meta information for an einsum instance. 



**Args:**
 
 - <b>`format_string`</b> (str):  The einsum format string. 
 - <b>`tensors`</b> (list):  A list of input tensors. 



**Returns:**
 
 - <b>`Meta_Info_Instance`</b>:  An instance of the Meta_Info_Instance class containing the computed meta information.  It has the following Attributes: 
            - tensors (int): The number of tensors involved in the computation. 
            - different_indices (int): The number of different indices in the computation. 
            - hadamard_products (int): The number of Hadamard products in the computation. 
            - edges (int): The number of contraction edges in the computation. 
            - hyperedges (int): The number of contraction hyperedges in the computation. 
            - tensors_in_largest_hyperedge (int): The number of tensors in the largest hyperedge in the computation. 
            - tensors_with_traces_or_diagonals (int): The number of tensors with traces or diagonals in the computation. 
            - independent_components (int): The number of independent components in the computation. 
            - tensors_in_largest_component (int): The number of tensors in the largest component in the computation. 
            - smallest_dimension_size (int): The size of the smallest dimension in the computation. 
            - largest_dimension_size (int): The size of the largest dimension in the computation. 
            - log2_output_size (float): The logarithm base 2 of the output size of the computation. 


---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `get_ops_and_max_size`

```python
def get_ops_and_max_size(
    format_string,
    annotated_ssa_path,
    *tensors,
    size_dict=None
)
```

Calculates the total number of operations and the maximum size of intermediate tensors. 

 Calculates the total number of operations and the maximum size of intermediate tensors  for a given format string, annotated SSA path, and input tensors. 



**Args:**
 
 - <b>`format_string`</b> (str):  The format string specifying the input and output tensors. 
 - <b>`annotated_ssa_path`</b> (list):  A list of tuples representing the annotated SSA path. 
 - <b>`tensors`</b> (tuple):  Input tensors. 
 - <b>`size_dict`</b> (dict, optional):  A dictionary mapping characters in the format string to their sizes. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the logarithm base 10 of the total number of operations and  the logarithm base 2 of the maximum size of intermediate tensors. 


---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `to_annotated_ssa_path`

```python
def to_annotated_ssa_path(format_string, ssa_path, is_ascii=False)
```

Annote an SSA path with their pairwise einsum format string. 



**Args:**
 
 - <b>`format_string`</b> (str):  The format string representing the einsum expression. 
 - <b>`ssa_path`</b> (list):  A list of tuples representing the SSA path indices. 
 - <b>`is_ascii`</b> (bool, optional):  Flag indicating whether to convert the annotated SSA path  to ASCII characters. Defaults to False. 



**Returns:**
 
 - <b>`list`</b>:  Annotated SSA path, where each element is a tuple containing the indices and  pairwise format_string for each step in the SSA path. 


---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `jensum`

```python
def jensum(annotated_ssa_path, *arguments, debug=False)
```

Perform a series of tensor contractions based on the annotated_ssa_path. 



**Args:**
 
 - <b>`annotated_ssa_path`</b> (list):  A list of tuples representing the annotated SSA path. 
 - <b>`Each tuple contains three elements`</b>:  the indices of the tensors to contract, and the contraction expression. 
 - <b>`*arguments`</b>:  Variable number of tensor arguments. 
 - <b>`debug`</b> (bool, optional):  If True, print debug information during the contractions. 



**Returns:**
 The final tensor resulting from the series of contractions. 



**Raises:**
 
 - <b>`AssertionError`</b>:  If the number of contractions is less than 1. 
 - <b>`AssertionError`</b>:  If the number of tensors in arguments is less than 2. 
 - <b>`RuntimeError`</b>:  If the number of tensors in arguments contradicts the number of entries in annotated_ssa_path. 


---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `jensum_meta`

```python
def jensum_meta(annotated_ssa_path, *arguments, debug=False)
```

Compute the meta information and perform tensor contractions using the jensum algorithm. 



**Args:**
 
 - <b>`annotated_ssa_path`</b> (list):  A list of tuples representing the annotated SSA path for each contraction. 
 - <b>`Each tuple contains three elements`</b>:  the indices of the tensors to contract, and the contraction expression. 
 - <b>`arguments`</b> (tuple):  The input tensors to be contracted. 
 - <b>`debug`</b> (bool, optional):  If True, debug information will be printed. Defaults to False. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the final contracted tensor and a list of meta information for each contraction.  The list of meta information contains for each step, where A and B are input tensors and C is the output tensor: 
            - The number of operations (flops). 
            - The sizes of the input tensors (size_A, size_B, size_C). 
            - The densities of the input and output tensors (density_A, density_B, density_C). 



**Raises:**
 
 - <b>`AssertionError`</b>:  If the number of contractions is less than 1 or the number of tensors in arguments is less than 2. 
 - <b>`RuntimeError`</b>:  If the number of tensors in arguments contradicts the number of entries in annotated_ssa_path. 
 - <b>`RuntimeError`</b>:  If the density of a tensor cannot be computed due to missing size or numel() attribute. 


---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L398"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `linear_path_runtime_meta`

```python
def linear_path_runtime_meta(format_string, linear_path, *arguments, debug=False)
```

Compute the runtime metadata for a linear path. 

This functions returns all metadata as given in the metadata.csv 



**Args:**
 
 - <b>`format_string`</b> (str):  The format string specifying the einsum operation. 
 - <b>`linear_path`</b> (str):  The linear path representing the einsum operation. 
 - <b>`*arguments`</b>:  Variable length arguments for the einsum operation. 
 - <b>`debug`</b> (bool, optional):  Whether to enable debug mode. Defaults to False. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the result of the einsum expression,  the execution time, the minimum density, and the average density. 


