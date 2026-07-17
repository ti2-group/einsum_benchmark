import unittest

import numpy as np
import opt_einsum as oe

import einsum_benchmark


class PackageSmokeTests(unittest.TestCase):
    def test_top_level_package_imports(self):
        self.assertIsNotNone(einsum_benchmark.generators)
        self.assertIsNotNone(einsum_benchmark.converters)
        self.assertIsNotNone(einsum_benchmark.meta)

    def test_matrix_chain_generator_contracts(self):
        format_string, shapes = einsum_benchmark.generators.structured.matrix_chain(
            num_matrices=3, min_dim=2, max_dim=2, seed=1
        )
        tensors = [np.ones(shape) for shape in shapes]

        result = oe.contract(format_string, *tensors)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
