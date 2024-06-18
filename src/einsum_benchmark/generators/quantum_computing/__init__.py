import importlib.util

if (
    importlib.util.find_spec("quimb") is None
    or importlib.util.find_spec("networkx") is None
):

    def maxcut(n=24, reg=3, p=3, seed=1):
        raise ImportError(
            """You need to install the optional dependencies for generators to use this function
            
            You can do this with pip install "einsum_benchmark[generators]"
            """
        )

else:
    from .maxcut import maxcut as maxcut
from .mps_product import matrix_product_state as matrix_product_state
