import numpy as np
from fairbench.bench.loader import features
from typing import Dict
import pandas as pd
from kfp import dsl


class CSV:
    integration = dsl.Output[dsl.Dataset]

    def __init__(self, data, numeric, categorical, labels):
        self.data = data
        self.numeric = numeric
        self.categorical = categorical
        self.labels = labels
        self.cols = numeric + categorical

    def to_features(self):
        return features(self.data, self.numeric, self.categorical).astype(np.float64)
