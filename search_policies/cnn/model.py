from .darts_policy.model_search import Network as CNN
from .darts_policy.model import NetworkCIFAR, NetworkImageNet

from .search_space.nas_bench.model import NasBenchNet
from .search_space.nas_bench.model_search import NasBenchNetSearch

"""
build a search space in DARTS/ENAS, to be a sub-space of NASBench, which is impossible.
The next step is to build a independent model, or just directly apply the DARTS/NAO into NASBench.

Easiest is to adopt DARTS into NASBench.

Because it is pretty hard coded. i need to design a new way to scale it up.
Including many different other

"""
# TODO this file is not necessary anymore. The model resides in one search space.
