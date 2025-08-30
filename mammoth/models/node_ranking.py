from mammoth.models.predictor import Predictor


class NodeRanking(Predictor):
    def __init__(self, ranker):
        import pygrank as pg

        assert isinstance(ranker, pg.NodeRanking)
        self.ranker = ranker

    def predict_unfair(self, x):
        import networkx as nx
        import pygrank as pg

        assert isinstance(x, nx.Graph) or isinstance(x, pg.Graph)
        return self.ranker(x)

    def predict(self, dataset, sensitive):
        assert (
            len(sensitive) == 1
        ), "fair node ranking algorithms can only account for one sensitive attribute"
        import networkx as nx
        import pygrank as pg

        x = dataset.to_features(None)
        sensitive = dataset.to_features(sensitive)
        assert isinstance(x, nx.Graph) or isinstance(x, pg.Graph)
        return self.ranker(x, sensitive=sensitive[0]).np
