<!-- markdownlint-disable -->

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/quantum_computing/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `generators.quantum_computing`




---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/quantum_computing/mps_product.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `matrix_product_state`

```python
def matrix_product_state(
    n=100,
    phys_dim_min=10,
    phys_dim_max=200,
    bond_dim=20,
    seed=None
)
```

Generates a matrix product state (MPS) for a quantum computing simulation. 



**Args:**
 
 - <b>`n`</b> (int):  The number of sites in the MPS. Default is 100. 
 - <b>`phys_dim_min`</b> (int):  The minimum physical dimension of each site. Default is 10. 
 - <b>`phys_dim_max`</b> (int):  The maximum physical dimension of each site. Default is 200. 
 - <b>`bond_dim`</b> (int):  The bond dimension between neighboring sites. Default is 20. 
 - <b>`seed`</b> (int):  The seed for the random number generator. Default is None. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the einsum string and the shapes of the tensors in the MPS. 



**Example:**
 ```python
 format_string, shapes = matrix_product_state(n=100, phys_dim_min=10, phys_dim_max=200, bond_dim=20, seed=0)
```



---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/generators/quantum_computing/maxcut.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `maxcut`

```python
def maxcut(n=24, reg=3, p=3, seed=1)
```

Generates a Max-Cut quantum circuit using the Quantum Approximate Optimization Algorithm (QAOA). 



**Args:**
 
 - <b>`n`</b> (int):  Number of nodes in the graph (default: 24). 
 - <b>`reg`</b> (int):  Regularity of the graph (default: 3). 
 - <b>`p`</b> (int):  Number of QAOA steps (default: 3). 
 - <b>`seed`</b> (int):  Seed for random number generation (default: 1). 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the input string and the arrays of the quantum circuit. 



**Example:**
 ```python
 format_string, arrays = maxcut(n=24, reg=3, p=3, seed=1)
```



