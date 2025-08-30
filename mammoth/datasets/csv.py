import numpy as np
from mammoth.datasets.dataset import Dataset


def _features(df, numeric, categorical):
    import pandas as pd

    dfs = [df[col] for col in numeric] + [
        pd.get_dummies(df[col]) for col in categorical
    ]
    return pd.concat(dfs, axis=1).values


def _pred_features(df, numeric, categorical, sensitives):
    import pandas as pd

    dfs = [df[col] for col in numeric if col not in sensitives] + [
        pd.get_dummies(df[col]) for col in categorical if col not in sensitives
    ]
    return pd.concat(dfs, axis=1).values


class CSV(Dataset):
    def __init__(self, data, numeric, categorical, labels, sensitives=None):
        import pandas as pd

        self.data = data
        self.numeric = numeric
        self.categorical = categorical
        self.labels = (
            pd.get_dummies(data[labels])
            if isinstance(labels, str)
            else (labels if isinstance(labels, dict) else {"label": labels})
        )
        self.cols = numeric + categorical
        if sensitives is None:
            sensitives = []
        self.pred_cols = [col for col in self.cols if col not in sensitives]

    def to_features(self, sensitive):
        return _features(self.data, self.numeric, self.categorical).astype(np.float64)

    def to_pred(self, sensitive):
        return _pred_features(
            self.data, self.numeric, self.categorical, sensitive
        ).astype(np.float64)
