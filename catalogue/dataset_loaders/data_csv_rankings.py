import os.path

from mammoth.datasets import CSV
from mammoth.integration import loader
import pandas as pd
from typing import List, Optional


@loader(namespace="mammotheu", version="v0036", python="3.11")
def data_csv_rankings(path: str = "", delimiter: str = "|") -> CSV:
    """
    This is a Loader to load .csv files with information about researchers
    The `Path` should be given relative to your locally running instance (e.g.: *./data/researchers/Top&#95;researchers.csv*)
    The `Delimiter` should match the CSV file you have (e.g.: '|')
    """
    try:
        raw_data = pd.read_csv(path, on_bad_lines="skip", delimiter=delimiter)
    except:
        raise ValueError(
            "Unable to read the given file.  Please double-check the parameters"
        )

    validate_input(raw_data)

    csv_dataset = CSV(
        raw_data,
        numeric=["Citations", "Productivity"],
        categorical=[
            "Nationality",
            "Nationality_Region",
            "Nationality_IncomeGroup",
            "aff_country",
            "aff_country_Region",
            "aff_country_IncomeGroup",
            "Gender",
        ],
        labels=[
            "id"
        ],  # Just a dummy right now.  We don't do supervised learning and don't "label" anything
    )
    return csv_dataset


def validate_input(data):
    required_columns = [
        "Citations",
        "Productivity",
        "Nationality_Region",
        "Nationality_IncomeGroup",
        "Gender",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"The following columns must be present in the dataset, but they are not: {missing_columns}"
        )
