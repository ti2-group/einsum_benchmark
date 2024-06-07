<!-- markdownlint-disable -->

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/converters/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `converters`




---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/converters/__init__.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `tn_from_uai_file`

```python
def tn_from_uai_file(file_path: str) → tuple[str, list]
```

Parses a UAI file and returns the einsum format string and tensors representing the graphical model. 



**Args:**
 
 - <b>`file_path`</b> (str):  The path to the UAI file. 



**Returns:**
 
 - <b>`tuple[str, list]`</b>:  A tuple containing the format string and a list of tensors. 


---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/converters/__init__.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `tn_from_dimacs_file`

```python
def tn_from_dimacs_file(file_path: str, clause_split_threshold=3) → tuple[str, list]
```

Converts a DIMACS (weighted) model counting file to a tensor network. 



**Args:**
 
 - <b>`file_path`</b> (str):  The path to the DIMACS file. 
 - <b>`clause_split_threshold`</b> (int, optional):  The threshold for splitting clauses. Defaults to 3. 



**Returns:**
 
 - <b>`tuple[str, list]`</b>:  A tuple containing the equation and the list of tensors. 


