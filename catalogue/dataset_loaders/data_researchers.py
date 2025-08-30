from mammoth.datasets.graph_csh import Graph_CSH
from mammoth.integration import loader
import pandas as pd
import numpy as np


@loader(
    namespace="mammotheu",
    version="v0037",
    python="3.11",
    packages=("pandas", "networkx"),
)
def data_researchers(
    paper_graph_path: str = "",
    paper_graph_delimiter: str = "|",
    paper_affiliations_path: str = "",
    paper_affiliation_delimiter: str = "|",
) -> Graph_CSH:
    """

    This is a Loader to load .csv files with information about researchers
    The `papers_path` and `papers_affiliations` should be given relative to your locally running instance
    (e.g.: *./data/researchers/Top&#95;researchers.csv*)
    The `Delimiter` should match the CSV file you have (e.g.: '|')

    Args:
        paper_graph_path: The path to the paper citation graph.
        paper_graph_delimiter: The delimiter in the paper graph file.
        paper_affiliations_path: The path to the paper affiliation map.
        paper_affiliation_delimiter: The delimiter of the paper affiliation map file.
    """
    try:
        # Read paper files for information of the papers:
        DF_papers = pd.read_csv(
            paper_graph_path, sep=paper_graph_delimiter
        )

        # Affiliations for the information of the authors
        DF_Affiliations = pd.read_csv(
            paper_affiliations_path, sep=paper_affiliation_delimiter
        )

    except:
        raise ValueError(
            "Unable to read the given file.  Please double-check the parameters"
        )

    validate_papers(DF_papers)
    validate_affiliations(DF_Affiliations)

    # Load country codes and country divisions (static data)
    country_codes = pd.read_csv("data/researchers/Country_codes.csv")
    country_divisions = pd.read_csv(
        "data/researchers/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_14/Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_14.csv",
        delimiter=",",
        header=0,
    )

    # Remove nodes where doi is nan
    DF_Affiliations = DF_Affiliations.dropna(subset=["doi"])

    ##Add year to the affiliations information:
    DF_Affiliations["year"] = [
        DF_papers[DF_papers.doi == doi].iloc[0]["year"] for doi in DF_Affiliations.doi
    ]

    for i in ["Alpha-3 code", "Alpha-2 code"]:
        country_codes[i] = [j.split(" ")[1] for j in country_codes[i]]

    Dict_codes_2_to_3 = {}
    Dict_codes_name_to_3 = {}
    for i in country_codes.index:
        Dict_codes_2_to_3[country_codes["Alpha-2 code"][i]] = {
            "Alpha-3 code": country_codes["Alpha-3 code"][i]
        }
        Dict_codes_name_to_3[country_codes["Country"][i]] = {
            "Alpha-3 code": country_codes["Alpha-3 code"][i]
        }
    Dict_country_divisions = {}
    for i in country_divisions.index:
        Dict_country_divisions[country_divisions["Country Code"][i]] = {
            "Region": country_divisions["Region"][i],
            "IncomeGroup": country_divisions["IncomeGroup"][i],
        }

    def bring_value(i, attribute, category, original_dict):
        try:
            return Dict_country_divisions[
                original_dict[DF_Affiliations[attribute][i]]["Alpha-3 code"]
            ][category]
        except:
            return np.nan

    for category in ["Region", "IncomeGroup"]:
        DF_Affiliations["aff_country_" + category] = [
            bring_value(
                i,
                attribute="aff_country",
                category=category,
                original_dict=Dict_codes_name_to_3,
            )
            for i in DF_Affiliations.index
        ]
        DF_Affiliations["Nationality_" + category] = [
            bring_value(
                i,
                attribute="Nationality",
                category=category,
                original_dict=Dict_codes_2_to_3,
            )
            for i in DF_Affiliations.index
        ]

    graph_data = Graph_CSH(DF_papers, DF_Affiliations, sensitive_columns=["Gender"])
    graph_data.create_coauth_graph()

    return graph_data


def validate_papers(data):
    required_columns = ["doi", "year"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"The following columns must be present in the dataset, but they are not: {missing_columns}"
        )
    len_papers = len(data)
    if len_papers == 0:
        raise ValueError("The papers dataset is empty")

    # TODO: Allow it, but don't do visualisations
    if len_papers > 1500:
        raise ValueError(
            "The papers dataset has too many papers. Please provide a smaller dataset."
        )


def validate_affiliations(data):
    required_columns = ["doi", "researcher_id", "aff_country", "Nationality"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"The following columns must be present in the dataset, but they are not: {missing_columns}"
        )

    len_papers = len(data)
    if len_papers == 0:
        raise ValueError("The affiliations dataset is empty")

    # TODO: Allow it, but don't do visualisations
    if len_papers > 2500:
        raise ValueError(
            "The papers dataset has too many papers. Please provide a smaller dataset."
        )
