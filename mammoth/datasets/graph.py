from mammoth.datasets.dataset import Dataset


class Graph(Dataset):
    def __init__(self, graph, communities: dict):
        import pygrank as pg

        self.graph = graph
        self.communities = {
            str(k): pg.to_signal(graph, v) for k, v in communities.items()
        }
        self.labels = None
        self.categorical = set(self.communities.keys())

    def to_features(self, sensitive):
        if sensitive is not None:
            return [self.communities[attr] for attr in sensitive]
        return self.graph

    @property
    def data(self):
        return {k: v.np for k, v in self.communities.items()}
