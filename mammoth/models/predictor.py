from mammoth.models.model import Model


class Predictor(Model):
    def predict(self, dataset, sensitive):
        raise Exception(
            f"{self.__class__.__name__} has no method predict(dataset, sensitive)"
        )
