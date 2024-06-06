import opt_einsum as oe
import numpy as np


# generates inner product of two mps states, phys_dim can vary
# code adapted from: https://optimized-einsum.readthedocs.io/en/stable/ex_large_expr_with_greedy.html
# License at the bottom of this file
def generate_mps_product(
    n=100, phys_dim_min=10, phys_dim_max=200, bond_dim=20, seed=None
):
    # start with the first site
    einsum_str = "ab,ac,"
    shapes = [(phys_dim_min, bond_dim), (phys_dim_min, bond_dim)]

    if seed is not None:
        np.random.seed(seed)

    # generate the einsum string for the middle tensors
    for i in range(1, n - 1):
        j = 3 * i
        ul, ur, m, ll, lr = (oe.get_symbol(i) for i in (j - 1, j + 2, j, j - 2, j + 1))
        einsum_str += "{}{}{},{}{}{},".format(m, ul, ur, m, ll, lr)

        phys_d = np.random.randint(phys_dim_min, phys_dim_max + 1)
        shapes.extend([(phys_d, bond_dim, bond_dim), (phys_d, bond_dim, bond_dim)])

    # finish with the last site
    i = n - 1
    j = 3 * i
    ul, m, ll = (oe.get_symbol(i) for i in (j - 1, j, j - 2))
    einsum_str += "{}{},{}{}".format(m, ul, m, ll)
    shapes.extend([(phys_dim_min, bond_dim), (phys_dim_min, bond_dim)])

    einsum_str += "->"
    return einsum_str, shapes


if __name__ == "__main__":
    from einsum_benchmark.util import compute_oe_path_from_shapes, print_oe_path_metrics

    format_string, shapes = generate_mps_product(
        n=100, phys_dim_min=10, phys_dim_max=200, bond_dim=20, seed=0
    )
    path, path_info = compute_oe_path_from_shapes(format_string, shapes)
    print_oe_path_metrics(path_info)


# The MIT License (MIT)

# Copyright (c) 2014 Daniel Smith

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
