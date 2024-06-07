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

## Acknowledgements

The broader data collection process included contributions from individuals whose data was transformed. We duly acknowledge the following for making their data publicly available:

- **Fichte, Johannes; Hecher, Markus; Florim Hamiti**: [Model Counting Competition 2020](https://zenodo.org/records/10031810)
- **Fichte, Johannes; Hecher, Markus**: Model Counting Competition [2021](https://zenodo.org/records/10006441) [2022](https://zenodo.org/records/10014715) [2023](https://zenodo.org/records/10012822)
- **Fichte, Johannes; Hecher, Markus; Woltran, Stefan; Zisser, Markus**: [A Benchmark Collection of #SAT Instances and Tree Decompositions](https://zenodo.org/records/1299752)
- **Meel, Kuldeep S.**: [Model Counting and Uniform Sampling Instances](https://zenodo.org/records/3793090)
- **Automated Reasoning Group at the University of California, Irvine**: [UAI Competitions](https://github.com/dechterlab/uai-competitions)
- **Martinis, John M. et al.**: [Quantum supremacy using a programmable superconducting processor Dataset. Dryad.](https://datadryad.org/stash/dataset/doi:10.5061/dryad.k6t1rj8)

Moreover, we thank the following authors of open source software used to generated instances:

- **Gray, Johnnie**: [quimb](https://quimb.readthedocs.io/en/latest/index.html), [cotengra](https://cotengra.readthedocs.io/en/latest/)
- **Soos, Mate, Meel, Kuldeep S**: [Arjun](https://github.com/meelgroup/arjun)
- **Stoian, Mihail**: [Netzwerk](https://github.com/stoianmihail/Netzwerk)
- **Liu, Jinguo; Lua, Xiuzhe; Wang, Lei**: [Yao.jl](https://github.com/QuantumBFS/Yao.jl)
- **Liu, Jinguo**: [YaoToEinsum.jl](https://github.com/QuantumBFS/YaoToEinsum.jl)
