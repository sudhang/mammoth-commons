from mammoth.models.model import Model


class EmptyModel(Model):
    def __init__(self):
        pass

    def predict(self, dataset, sensitive):
        raise Exception("Cannot make predictions for an empty model")
