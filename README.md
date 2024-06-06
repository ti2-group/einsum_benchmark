# Einsum Benchmark Data

Benchmark data repository accompanying the paper: "Einsum Benchmark: Enabling the Development of Next-Generation Tensor Execution Engines". It includes a package to easily work with the benmark data and all methods we used to convert and create our instances so you can create new ones yourself.

## Installation

Install our base package, you will need at least python 3.10:

```bash
pip install einsum_benchmark
```

## Basic Usage

Now you can load and run an instance like this:

```python
import opt_einsum as oe
import einsum_benchmark

instance = einsum_benchmark.instances["qc_circuit_n49_m14_s9_e6_pEFGH_simplified"]

opt_size_path_meta = instance.paths.opt_size
print("Size optimized path")
print("log10[FLOPS]:", round(opt_size_path_meta.flops, 2))
print("log2[SIZE]:", round(opt_size_path_meta.size, 2))
result = oe.contract(
    instance.format_string, *instance.tensors, optimize=opt_size_path_meta.path
)
print("sum[OUTPUT]:", result.sum(), instance.result_sum)
```

For more information please read our [docs](https://benchmark.einsum.org/docs/guides/getting-started/).
