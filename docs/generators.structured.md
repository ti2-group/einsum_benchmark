<!-- markdownlint-disable -->

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/structured/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `generators.structured`




---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/structured/matrix_chain.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `generate_mcm`

```python
def generate_mcm(
    num_matrices=10,
    min_dim=10,
    max_dim=1000,
    is_shuffle=False,
    seed=None
)
```

Generate a matrix chain multiplication problem. 



**Args:**
 
 - <b>`num_matrices`</b> (int):  The number of matrices in the chain (default: 10). 
 - <b>`min_dim`</b> (int):  The minimum dimension of each matrix (default: 10). 
 - <b>`max_dim`</b> (int):  The maximum dimension of each matrix (default: 1000). 
 - <b>`is_shuffle`</b> (bool):  Whether to shuffle the einsum string and shapes (default: False). 
 - <b>`seed`</b> (int):  The seed value for reproducibility (default: None). 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the einsum string and the shapes of the matrices. 



**Raises:**
 
 - <b>`AssertionError`</b>:  If the lists of einsum string and shapes have different sizes. 



**Example:**
 ```python
 generate_mcm(num_matrices=10, min_dim=10, max_dim=1000, is_shuffle=True, seed=0)
```



---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/structured/tree.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `tree`

```python
def tree(n=100, d_min=4, d_max=12, n_outer=2, seed=1)
```

Create a random contraction equation that corresponds to a tree. 



**Args:**
 
 - <b>`n`</b> (int):  The number of tensors. 
 - <b>`d_min`</b> (int, optional):  The minimum size of an index. 
 - <b>`d_max`</b> (int, optional):  The maximum size of an index. 
 - <b>`n_outer`</b> (int, optional):  The number of outer indices. 
 - <b>`seed`</b> (int, optional):  Seed for generator. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the contraction equation format string and the shapes of the tensors. 


---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/structured/lattice.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `lattice`

```python
def lattice(dims, cyclic=False, d_min=2, d_max=None, seed=None)
```

Create a random contraction equation that corresponds to a lattice. 



**Args:**
 
 - <b>`dims`</b> (sequence of int):  The size of each dimension, with the dimensionality being the length  of the sequence. 
 - <b>`cyclic`</b> (bool or sequence of bool, optional):  Whether each dimension is cyclic or not. If a sequence,  must be the same length as ``dims``. 
 - <b>`d_min`</b> (int, optional):  The minimum size of an index. 
 - <b>`d_max`</b> (int, optional):  The maximum size of an index. If ``None``, defaults to ``d_min``, i.e.  all indices are the same size. 
 - <b>`seed`</b> (None or int, optional):  Seed for ``random.Random`` for repeatability. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the contraction equation format string and the shapes of the inputs. 



**Raises:**
 
 - <b>`TypeError`</b>:  If ``cyclic`` is a sequence but not the same length as ``dims``. 


