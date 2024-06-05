import quimb as qu
import quimb.tensor as qtn
import networkx as nx


# generates a maxcut network
# code adapted from: https://quimb.readthedocs.io/en/latest/examples/ex_tn_qaoa_energy_bayesopt.html
# License at the bottom of this file
def generate_maxcut_network(n=24, reg=3, p=3, seed=1):
    G = nx.random_regular_graph(reg, n, seed=seed)
    terms = {(i, j): 1 for i, j in G.edges}
    gammas = qu.randn((p,))
    betas = qu.randn((p,))
    circ = qtn.circ_qaoa(terms, p, gammas, betas)

    tn = circ.amplitude_rehearse(simplify_sequence="", optimize=None)["tn"]
    arrays = tn.arrays
    inputs, _, size_dict = tn.get_inputs_output_size_dict()
    return ",".join(inputs) + "->", arrays


if __name__ == "__main__":
    from utils import compute_oe_path_from_arrays, print_oe_path_metrics

    format_string, arrays = generate_maxcut_network(n=24, reg=3, p=3, seed=1)
    path, path_info = compute_oe_path_from_arrays(format_string, arrays)
    print_oe_path_metrics(path_info)


# Copyright 2015-2024 Johnnie Gray

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
