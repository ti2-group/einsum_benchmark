<!-- markdownlint-disable -->

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `meta`




---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `get_ops_and_max_size`

```python
def get_ops_and_max_size(
    format_string,
    annotated_ssa_path,
    *tensors,
    size_dict=None
)
```






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






---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `to_annotated_ssa_path`

```python
def to_annotated_ssa_path(format_string, ssa_path, is_ascii=False)
```






---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/info.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `compute_meta_info_of_einsum_instance`

```python
def compute_meta_info_of_einsum_instance(format_string, l)
```






---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `jensum`

```python
def jensum(annotated_ssa_path, *arguments, debug=False)
```






---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `jensum_meta`

```python
def jensum_meta(annotated_ssa_path, *arguments, debug=False)
```






---

<a href="https://github.com/ti2-group/einsum_benchmark/blob/main/src/einsum_benchmark/meta/runtime.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>function</kbd> `linear_path_runtime_meta`

```python
def linear_path_runtime_meta(format_string, linear_path, *arguments, debug=False)
```






