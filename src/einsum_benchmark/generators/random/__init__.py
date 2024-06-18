import importlib.util

from .connected_hypernetwork import (
    connected_hypernetwork as connected_hypernetwork,
)
from .connected_network import connected_network as connected_network


if importlib.util.find_spec("networkx") is None:

    def regular(n, reg, d_min=2, d_max=3, seed=None):
        raise ImportError(
            """You need to install the optional dependencies for generators to use this function
            
            You can do this with pip install "einsum_benchmark[generators]"
            """
        )

else:
    from .randreg import regular as regular
