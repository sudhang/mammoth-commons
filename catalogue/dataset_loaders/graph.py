import os.path

from mammoth.datasets import Graph
from mammoth.integration import loader


@loader(namespace="mammotheu", version="v0036", python="3.11", packages=("pygrank",))
def data_graph(
    path: str = "",
) -> Graph:
    """Loads the edges of a graph organized as rows of a comma-delimited file.

    Args:
        path: A url from which to load the edges, or a pygrank dataset name to be automatically downloaded, preprocessed and loaded.
    """
    import pygrank as pg

    _, graph, communities = next(pg.load_datasets_multiple_communities([path]))
    return Graph(graph, communities)
