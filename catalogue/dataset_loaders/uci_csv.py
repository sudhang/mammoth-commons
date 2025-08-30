from mammoth.datasets import CSV
from mammoth.integration import loader, Options
from collections import OrderedDict
import os


@loader(
    namespace="mammotheu",
    version="v0036",
    python="3.11",
    packages=("pandas", "ucimlrepo"),
)
def data_uci(
    dataset_name: Options("Credit", "Bank") = None,
    target=None,
) -> CSV:
    """Loads a dataset from the UCI Machine Learning Repository (<a href="https://archive.ics.uci.edu/ml/index.php" target="_blank">www.uci.org</a>) containing numeric, categorical, and predictive data columns. The dataset is automatically downloaded from the repository, and basic preprocessing is applied to identify the column types. The specified target column is treated as the predictive label.
        To customize the loading process (e.g., use a different target column, load a subset of features, or handle missing data differently), additional parameters or a custom loader can be used.

    Args:
        dataset_name: The name of the dataset.
        target: The name of the predictive label.
    """
    name = dataset_name.lower()
    if name == "credit":
        d_id = 350
    elif name == "bank":
        d_id = 222
    else:
        raise Exception("Unexpected dataset name: " + name)

    import pandas as pd
    from ucimlrepo import fetch_ucirepo

    if d_id is None:
        all_raw_data = fetch_ucirepo(name)
    else:
        all_raw_data = fetch_ucirepo(id=d_id)
    # print(raw_data.metadata.additional_info.summary)
    # print(raw_data.metadata.additional_info.variable_info)
    raw_data = all_raw_data.data.features
    numeric = [
        col
        for col in raw_data
        if pd.api.types.is_any_real_numeric_dtype(raw_data[col])
        and len(set(raw_data[col])) > 10
    ]
    numeric_set = set(numeric)
    categorical = [col for col in raw_data if col not in numeric_set]
    if len(categorical) < 1:
        raise Exception("At least two categorical columns are required.")
    label = all_raw_data.data.targets[target]

    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=label,
    )

    csv_dataset.description = OrderedDict(
        [
            ("Summary", all_raw_data.metadata.additional_info.summary),
            ("Variables", all_raw_data.metadata.additional_info.variable_info),
        ]
    )
    return csv_dataset
