<!-- markdownlint-disable -->

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/language_model/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `generators.language_model`




---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/language_model/generator.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `p_first_and_last`

```python
def p_first_and_last(
    mera_depth: int,
    axis_size_hidden: int,
    axis_size_observable: int
) → tuple[str, list[tuple[int, ]]]
```

Generates an einsum query and shape arguments for computing the distribution of the first and last observable in a model with the given parameters. 



**Args:**
 
 - <b>`mera_depth`</b> (int):  Number of layers in a MERA network with 4th-order tensors. 
 - <b>`axis_size_hidden`</b> (int):  Domain size of hidden variables. 
 - <b>`axis_size_observable`</b> (int):  Domain size of observable variables. 



**Returns:**
 
 - <b>`tuple[str, list[tuple[int, ...]]]`</b>:  The einsum format string, which is needed to compute the distribution, and the shapes of its arguments. 



**Example:**
 ```python
 format_string, argument_shapes = p_first_and_last(mera_depth=1, axis_size_hidden=3, axis_size_observable=11)
```



---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/language_model/generator.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `sliding_likelihood`

```python
def sliding_likelihood(
    mera_depth: int,
    axis_size_hidden: int,
    axis_size_observable: int,
    batch_size: int
) → tuple[str, list[tuple[int, ]]]
```

Generates an einsum query and shape arguments for computing the likelihood of the model on an imaginary batch of training data. 



**Args:**
 
 - <b>`mera_depth`</b> (int):  Number of layers in a MERA network with 4th-order tensors. 
 - <b>`axis_size_hidden`</b> (int):  Domain size of hidden variables. 
 - <b>`axis_size_observable`</b> (int):  Domain size of observable variables. 
 - <b>`batch_size`</b> (int):  Number of context windows to compute the likelihood for. 



**Returns:**
 
 - <b>`tuple[str, list[tuple[int, ...]]]`</b>:  The einsum format string, which is needed to compute the batch likelihood, and the shapes of its arguments. 

**Example:**
 ```python
 format_string, shapes = sliding_likelihood(mera_depth=1, axis_size_hidden=3, axis_size_observable=11, batch_size=100)
```



