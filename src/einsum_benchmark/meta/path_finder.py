#
# Based on https://github.com/ti2-group/hybrid_contraction_tree_optimizer/
#

from collections import defaultdict
import kahypar
import optuna
import math
import random
from os.path import join
import os
from dataclasses import dataclass, field
from typing import Hashable, Optional, Union, List, Tuple, Dict, Literal
from .runtime import get_ops_and_max_size, to_annotated_ssa_path
from cgreedy.alg import CGreedy

optuna.logging.set_verbosity(optuna.logging.WARNING)

Inputs = List[List[Hashable]]
Output = List[Hashable]
SizeDict = Dict[Hashable, int]
Path = List[Tuple[int, ...]]

dirname = os.path.dirname(__file__)


def safe_log2(x):
    if x < 1:
        return 0
    return math.log2(x)


def get_node_weight_by_log_size(size_dict, node):
    return max(sum([int(safe_log2((size_dict[edge]))) for edge in node]), 1)


def get_node_weight(node, weight_nodes, size_dict):
    if weight_nodes == "const":
        return 1
    elif weight_nodes == "log":
        return get_node_weight_by_log_size(size_dict, node)
    else:
        raise ValueError(f"Unknown weight_nodes: {weight_nodes}")


def partition_tn(
    nodes,
    size_dict,
    imbalance=0.1,
    seed=None,
    profile=None,
    mode="recursive",
    objective="cut",
    weight_nodes="const",
):
    parts = 2
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    hyper_edges = defaultdict(lambda: [])

    # Filter out nodes, that have only output indices
    node_weights = []
    for node in nodes:
        node_weights.append(get_node_weight(node, weight_nodes, size_dict))

    for node_index, input in enumerate(nodes):
        for index in input:
            hyper_edges[index].append(node_index)

    # Filter out open edges
    hyper_edge_list = [e for e in hyper_edges.values() if len(e) > 1]
    hyper_edge_keys = [key for key, e in hyper_edges.items() if len(e) > 1]

    # No edges, just distribute nodes
    if len(hyper_edge_list) == 0:
        num_of_all_nodes = len(nodes)
        return [
            i // (math.ceil(num_of_all_nodes / (parts)))
            for i in range(num_of_all_nodes)
        ]
    index_vector = []
    edge_vector = []

    for e in hyper_edge_list:
        index_vector.append(len(edge_vector))
        edge_vector.extend(e)

    index_vector.append(len(edge_vector))

    edge_weights = [
        int(max(safe_log2((size_dict[edge])), 1)) for edge in hyper_edge_keys
    ]

    hypergraph_kwargs = {
        "num_nodes": len(nodes),
        "num_edges": len(hyper_edge_list),
        "index_vector": index_vector,
        "edge_vector": edge_vector,
        "k": parts,
        "edge_weights": edge_weights,
        "node_weights": node_weights,
    }

    hypergraph = kahypar.Hypergraph(**hypergraph_kwargs)

    if profile is None:
        profile_mode = {"direct": "k", "recursive": "r"}[mode]
        profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"

    context = kahypar.Context()
    context.loadINIconfiguration(join(dirname, "./kahypar_profiles", profile))
    context.setK(parts)
    context.setSeed(seed)
    context.suppressOutput(True)
    context.setEpsilon(imbalance)

    kahypar.partition(hypergraph, context)

    return [hypergraph.blockID(i) for i in hypergraph.nodes()]


@dataclass
class BasicInputNode:
    indices: List[Hashable]


@dataclass
class OriginalInputNode(BasicInputNode):
    id: int

    def get_id(self):
        return self.id


@dataclass
class SubNetworkInputNode(BasicInputNode):
    sub_network: "TensorNetwork"

    def get_id(self):
        return self.sub_network._ssa_id

    def __repr__(self) -> str:
        return f"Sub network Input({self.sub_network.output_indices},"


InputNode = Union[OriginalInputNode, SubNetworkInputNode]
InputNodes = List["InputNode"]


def greedy_optimizer(
    tn: "TensorNetwork", minimize: Literal["flops", "size"]
) -> Tuple[Path]:
    inputs = [set(input.indices) for input in tn.inputs]
    output = set(tn.output_indices)
    size_dict = tn.size_dict

    seed = random.randint(0, 2**31 - 1)

    optimizer = CGreedy(
        seed=seed,
        minimize=minimize,
        max_repeats=64,
        progbar=False,
        threshold_optimal=12,
        threads=20,
        is_linear=False,
    )

    path = optimizer.__call__(inputs, output, size_dict)

    return path, optimizer.flops_log10, optimizer.size_log2


@dataclass
class TensorNetwork:
    name: str
    parent_name: str
    inputs: InputNodes
    size_dict: SizeDict
    output_indices: Output
    _ssa_id: Optional[int] = field(default=None, init=False)


def get_sub_networks(
    tensor_network: TensorNetwork,
    imbalance: float,
    weight_nodes: str = "const",
):
    input_nodes = tensor_network.inputs
    output = tensor_network.output_indices
    num_input_nodes = len(input_nodes)
    assert (
        num_input_nodes > 2
    ), f"You need to pass at least two input nodes, {input_nodes}"

    inputs = [input.indices for input in input_nodes]

    if len(output) > 0:
        inputs.append(output)

    block_ids = partition_tn(
        inputs,
        tensor_network.size_dict,
        imbalance=imbalance,
        weight_nodes=weight_nodes,
        mode="recursive",
        objective="cut",
    )

    # Noramlize block ids

    # Check if all input nodes were assigned to the same block
    input_block_ids = block_ids[:num_input_nodes]
    min_block_id = min(input_block_ids)
    max_block_id = max(input_block_ids)
    if min_block_id == max_block_id:
        # If there is only one block just distribute them with modulo
        block_ids = [i % 2 for i in range(num_input_nodes + 1)]
        input_block_ids = block_ids[:num_input_nodes]
    else:
        if min_block_id != 0 or max_block_id != 1:
            block_ids = [0 if id == min_block_id else 1 for id in block_ids]

    assert (
        len(set(input_block_ids)) == 2
    ), f"There should be two blocks, {input_block_ids}"

    # Group inputs by block id
    block_inputs: list[InputNodes] = [[], []]
    for block_id, input_node in zip(block_ids, input_nodes):
        block_inputs[block_id].append(input_node)

    block_indices = [
        frozenset(set.union(*[set(input_node.indices) for input_node in block]))
        for block in block_inputs
    ]

    # Include output indices in cut, since it is not in block indices
    cut_indices = block_indices[0].intersection(block_indices[1]).union(output)

    if len(output) > 0:
        parent_block_id = block_ids[-1]
    else:
        parent_block_id = random.choice([0, 1])

    child_block_id = 1 - parent_block_id
    child_output = list(cut_indices.intersection(block_indices[child_block_id]))

    parent_sub_network = TensorNetwork(
        f"{tensor_network.name}.{parent_block_id}",
        tensor_network.name,
        block_inputs[parent_block_id],
        tensor_network.size_dict,
        output,
    )

    child_sub_network = TensorNetwork(
        f"{tensor_network.name}.{child_block_id}",
        parent_sub_network.name,
        block_inputs[child_block_id],
        tensor_network.size_dict,
        child_output,
    )

    sub_network_node = SubNetworkInputNode(
        child_sub_network.output_indices,
        child_sub_network,
    )
    parent_sub_network.inputs.append(sub_network_node)

    return parent_sub_network, child_sub_network


def extend_path(tn: TensorNetwork, sub_path: Path, last_id, path: Path):
    n = len(tn.inputs)
    for pair in sub_path:
        new_pair = []
        for element in pair:
            if element < n:
                new_pair.append(int(tn.inputs[element].get_id()))
            else:
                new_pair.append(last_id - n + element + 1)
        path.append(tuple(new_pair))

    return last_id + len(sub_path)


def hybrid_hypercut_greedy(
    inputs: Inputs,
    output: Output,
    size_dict: SizeDict,
    imbalance,
    weight_nodes="const",
    minimize="flops",
    cutoff=15,
):
    # Noramlize parameters
    inputs = [list(input) for input in inputs]
    output = list(output)

    input_nodes: InputNodes = [
        OriginalInputNode(input, id) for id, input in enumerate(inputs)
    ]

    tensor_network = TensorNetwork("tn", None, input_nodes, size_dict, output)

    stack = [tensor_network]
    path = []
    last_id = len(inputs) - 1
    network_by_name = {tensor_network.name: tensor_network}
    while stack:
        tn = stack.pop()
        if len(tn.inputs) <= cutoff:
            sub_path, sub_flops, sub_size = greedy_optimizer(tn, minimize=minimize)
            if isinstance(sub_path, tuple):
                sub_path = [sub_path]
            last_id = extend_path(tn, sub_path, last_id, path)
            tn._ssa_id = last_id
            while tn.parent_name and len(tn.parent_name) < len(tn.name):
                network_by_name[tn.parent_name]._ssa_id = last_id
                tn = network_by_name[tn.parent_name]
            continue
        parent_sub_network, child_sub_network = get_sub_networks(
            tn,
            imbalance=imbalance,
            weight_nodes=weight_nodes,
        )
        stack.append(parent_sub_network)
        network_by_name[parent_sub_network.name] = parent_sub_network
        stack.append(child_sub_network)
        network_by_name[child_sub_network.name] = child_sub_network

    format_string = (
        ",".join(["".join(input) for input in inputs]) + "->" + "".join(output)
    )

    annottated_ssa_path = to_annotated_ssa_path(
        format_string,
        path,
        is_ascii=False,
    )
    flops, max_size = get_ops_and_max_size(
        format_string, annottated_ssa_path, size_dict=size_dict
    )
    return path, flops, max_size


class TrackBestPathCallback:
    def __init__(self, minimize: str = "flops"):
        self.best_path = None
        self.best_size = None
        self.best_flops = None
        self.minimize = minimize

    def update_best(self, trial):
        self.best_path = trial.user_attrs["path"]
        self.best_size = trial.user_attrs["size"]
        self.best_flops = trial.user_attrs["flops"]

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if self.best_path is None:
            self.update_best(trial)
        elif self.minimize == "flops":
            if trial.user_attrs["flops"] < self.best_flops:
                self.update_best(trial)
        else:
            if (
                trial.user_attrs["size"] < self.best_size
                or trial.user_attrs["size"] == self.best_size
                and trial.user_attrs["flops"] < self.best_flops
            ):
                self.update_best(trial)

        trial.set_user_attr("path", None)
        trial.set_user_attr("flops", None)
        trial.set_user_attr("size", None)


def hyper_optimized_hhg(
    inputs: Inputs,
    output: Output,
    size_dict: SizeDict,
    minimize: Literal["flops", "size"],
    n_trials: int = 128,
    n_jobs: int = 10,
    show_progress_bar: bool = True,
    timeout: Optional[int] = None,
):

    tracker_best_path_cb = TrackBestPathCallback(minimize=minimize)

    def object_fn(trial):
        imbalance = trial.suggest_float("imbalance", 0.01, 0.5)
        weight_nodes = trial.suggest_categorical("weight_nodes", ["const", "log"])
        cutoff = trial.suggest_int("cutoff", 10, min(len(inputs), 1500))
        try:
            path, flops, size = hybrid_hypercut_greedy(
                inputs,
                output,
                size_dict,
                imbalance=imbalance,
                weight_nodes=weight_nodes,
                minimize=minimize,
                cutoff=cutoff,
            )
        except Exception as e:
            # Handle the exception here
            print(f"An error occurred 2: {str(e)}")
            trial.set_user_attr("path", [])
            trial.set_user_attr("flops", math.inf)
            trial.set_user_attr("size", math.inf)
            return math.inf
        trial.set_user_attr("path", path)
        trial.set_user_attr("flops", flops)
        trial.set_user_attr("size", size)
        if minimize == "flops":
            return flops
        else:
            return size

    study = optuna.create_study()
    study.optimize(
        object_fn,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
        timeout=timeout,
        callbacks=[tracker_best_path_cb],
        gc_after_trial=True,
    )
    return (
        tracker_best_path_cb.best_path,
        tracker_best_path_cb.best_flops,
        tracker_best_path_cb.best_size,
    )
