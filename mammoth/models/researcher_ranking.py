from mammoth.models.model import Model


class ResearcherRanking(Model):
    def __init__(self, ranking_function, baseline_ranking_function=None):
        self.rank = ranking_function
        self.baseline_rank = baseline_ranking_function

    def predict(self, dataset, sensitive):
        if len(sensitive) != 1:
            raise Exception("Researcher ranking data cannot have ")
        return self.rank(dataset, sensitive[0])
