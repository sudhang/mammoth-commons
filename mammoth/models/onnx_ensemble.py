import numpy as np
from mammoth.models.predictor import Predictor
import re


class ONNXEnsemble(Predictor):
    def __init__(
        self,
        models,
        _=None,
        alphas=None,
        classes=None,
        n_classes=None,
        theta=None,
        pareto=None,
        sensitives=None,
    ):
        assert (
            _ is None
        ), "Internal error: ONNXEnsemble was accidentally constructed with more positional arguments than acceptable"
        self.models = models
        self.pareto = pareto
        self.alphas = alphas
        self.sensitive = sensitives
        self.classes = classes
        self.theta = theta
        self.n_classes = n_classes

    def _extract_number(self, filename):
        match = re.search(r"_(\d+)\.onnx$", filename)
        return int(match.group(1)) if match else float("inf")

    def predict(self, dataset, sensitive, theta=None):
        """assert (
            sensitive is None or len(sensitive) == 0
        ), "ONNXEnsemble can only be called with no declared sensitive attributes" """
        theta = theta if self.theta is None else self.theta
        sensitive = sensitive if self.sensitive is None else self.sensitive
        X = dataset.to_pred(sensitive)

        # n_classes = self.params['n_classes']
        classes = self.classes[:, np.newaxis]

        pred = sum(
            (estimator.predict(X, []) == classes).T * alpha
            for estimator, alpha in zip(
                self.models[:theta],
                self.alphas[:theta],
            )
        )
        pred /= self.alphas[:theta].sum()
        pred[:, 0] *= -1
        preds = classes.take(pred.sum(axis=1) > 0, axis=0)
        return np.squeeze(preds, axis=1)
