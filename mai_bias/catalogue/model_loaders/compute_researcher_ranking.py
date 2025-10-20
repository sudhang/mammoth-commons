import math
from mammoth_commons.integration import loader
from mammoth_commons.models.researcher_ranking import ResearcherRanking
import random

from functools import partial
from mammoth_commons.integration import loader
from mammoth_commons.models.researcher_ranking import ResearcherRanking


def normal_ranking(dataset, ranking_variable, graph):
    """
    Rank a dataset based on a specified variable in descending order.

    This function sorts the input dataset based on the values of the given ranking variable,
    and assigns a ranking score to each entry. The highest value for the ranking variable receives
    a rank of 1, and the ranking increases for lower values.

    Parameters:
    dataset (pandas.DataFrame): The dataset to be ranked, provided as a pandas DataFrame.
    ranking_variable (str): The name of the column in the dataset to use for ranking.

    Returns:
    pandas.DataFrame: A new DataFrame containing the original dataset sorted by ranking_variable
                  in descending order, with an additional column 'Ranking_{ranking_variable}'
                  that contains the corresponding rank for each entry.
    """

    ranked_dataset = dataset.sort_values(ranking_variable, ascending=False)
    ranked_dataset[f"Ranking_{ranking_variable}"] = [
        i + 1 for i in range(ranked_dataset.shape[0])
    ]
    return ranked_dataset


def Compute_mitigation_strategy(
    dataset,
    mitigation_method,
    ranking_variable,
    sensitive_attribute,
    protected_attribute,
    graph,
):
    """
    Computes a ranking adjustment based on selected mitigation strategies to ensure fairness in dataset.

    Parameters:
    -----------
    dataset : pd.DataFrame
        A pandas DataFrame containing the data to be ranked, including sensitive and protected attributes.

    mitigation_method : str
        The method of mitigation to apply. Options include:
        - "Statistical_parity": Adjusts ranking to balance representation between protected and non-protected groups.
        - "Equal_parity": Assumes equal distribution for protected and non-protected groups.
        - "Updated_statistical_parity": Not yet implemented.
        - "Internal_group_fairness": Not yet implemented.

    ranking_variable : str
        The name of the new ranking variable to be added to the dataset.

    sensitive_attribute : str
        The name of the column in the dataset that contains sensitive attribute values.

    protected_attribute : str
        The name of the specific sensitive attribute that is considered protected and requires fairness adjustment.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame with updated rankings based on the specified mitigation strategy.

    Raises:
    -------
    NotImplementedError
        If "Updated_statistical_parity" or "Internal_group_fairness" is selected as the mitigation method.
    """
    from hyperfair.hyperfair import (
        adjust_ranking,
        measure_fairness_multiple_points,
        measure_fairness_single_point,
    )
    from hyperfair.data_loader import load_data_from_pandas_df

    # Only consider rows where the sensitive attribute (eg: "Gender") isn't missing
    Dataframe_ranking = dataset[~dataset[sensitive_attribute].isnull()]

    # Split the data into protected and non-protected groups
    Chosen_groups, Chosen_researchers = {}, {}
    sensitive = set(Dataframe_ranking[sensitive_attribute])
    Ranking_sets = {
        attribute: Dataframe_ranking[
            Dataframe_ranking[sensitive_attribute] == attribute
        ]
        for attribute in sensitive
    }
    non_protected_attribute = [i for i in sensitive if i != protected_attribute][0]
    Len_groups = Dataframe_ranking[sensitive_attribute].value_counts()

    if mitigation_method == "Statistical_parity":
        # Chosen_groups would be a list with a desired ranking of group members
        # eg: ["female", "male", "male", "female", ...]
        Chosen_groups = []
        Len_group_in_ranking = Len_groups
        for i in range(Dataframe_ranking.shape[0]):  # Go through every rank slot

            # Probability that the next slot should be protected
            P_minority = Len_group_in_ranking[protected_attribute] / (
                Len_group_in_ranking[protected_attribute]
                + Len_group_in_ranking[non_protected_attribute]
            )
            # The next chosen group label as per the probability
            Chosen_groups += [
                random.choices(
                    [protected_attribute, non_protected_attribute],
                    [P_minority, 1 - P_minority],
                )[0]
            ]
            # Decrement the remaining-count pool for the chosen group
            Len_group_in_ranking[Chosen_groups[-1]] -= 1

            # Determine which positions each group will occupy
            Positions = {
                non_protected_attribute: [
                    i
                    for i, j in enumerate(Chosen_groups)
                    if j == non_protected_attribute
                ],
                protected_attribute: [
                    i for i, j in enumerate(Chosen_groups) if j == protected_attribute
                ],
            }

            # Pick concrete researcher IDs to fill the positions from above
            Chosen_researchers = {
                i_ranking: Ranking_sets[non_protected_attribute].iloc[i_position]["id"]
                for i_position, i_ranking in enumerate(
                    Positions[non_protected_attribute]
                )
            }
            for i_position, i_ranking in enumerate(Positions[protected_attribute]):
                Chosen_researchers[i_ranking] = Ranking_sets[protected_attribute].iloc[
                    i_position
                ]["id"]

            New_ranking = {r: i for i, r in Chosen_researchers.items()}

    elif mitigation_method == "Equal_parity":
        P_minority = 0.5
    elif mitigation_method == "Updated_statistical_parity":
        raise NotImplementedError(
            "Updated_statistical_parity method is not implemented yet."
        )
    elif mitigation_method == "Internal_group_fairness":
        raise NotImplementedError(
            "Internal_group_fairness method is not implemented yet."
        )
    elif mitigation_method == "Breaking_network":
        Total_size = Dataframe_ranking.shape[0]
        Individuals_waiting_to_be_chosen = list(Dataframe_ranking.id)
        Chosen_researchers = []

        count = 0
        while Total_size > count and len(Individuals_waiting_to_be_chosen) > 1:
            count += 1
            node = Individuals_waiting_to_be_chosen[0]
            Neighbors = [u for u in graph.neighbors(node)]
            Neighbors_in_ranking = [u for u in Neighbors if u in Chosen_researchers]

            # If the node does not have neighbors or all their neighbors have been assigned:
            if (len(Neighbors_in_ranking) == 0) or (
                len(Neighbors_in_ranking) == len(Neighbors)
            ):
                Chosen_researchers += [node]
                Individuals_waiting_to_be_chosen = [
                    i for i in Individuals_waiting_to_be_chosen if i != node
                ]
            else:
                # Select a random position in the list and locate the node there:
                pos_selected = random.randint(0, len(Individuals_waiting_to_be_chosen))
                Individuals_waiting_to_be_chosen_1 = []
                for i in (
                    Individuals_waiting_to_be_chosen[1:pos_selected]
                    + [node]
                    + Individuals_waiting_to_be_chosen[pos_selected:]
                ):
                    if i not in Individuals_waiting_to_be_chosen_1:
                        Individuals_waiting_to_be_chosen_1 += [i]
                Individuals_waiting_to_be_chosen = Individuals_waiting_to_be_chosen_1

        #             Total_size = len(Individuals_waiting_to_be_chosen)
        Chosen_researchers += Individuals_waiting_to_be_chosen
        New_ranking = {r: i - 1 for i, r in enumerate(Chosen_researchers)}
    elif mitigation_method == "Reordering_network":
        Total_size = Dataframe_ranking.shape[0]
        Individuals_waiting_to_be_chosen = list(Dataframe_ranking.id)
        Chosen_researchers = {}

        count = 0
        while Total_size > count and len(Individuals_waiting_to_be_chosen) > 0:

            node = Individuals_waiting_to_be_chosen[0]
            Neighbors = [u for u in graph.neighbors(node)]
            Neighbors_in_ranking = {
                u: Chosen_researchers[u]
                for u in Neighbors
                if u in Chosen_researchers.keys()
            }
            if len(Neighbors_in_ranking) == 0:
                Chosen_researchers[node] = len(Chosen_researchers)
                Individuals_waiting_to_be_chosen = [
                    i for i in Individuals_waiting_to_be_chosen if i != node
                ]
            else:
                Neighbors_in_ranking = dict(
                    sorted(Neighbors_in_ranking.items(), key=lambda item: item[1])
                )
                Neighbors_in_ranking[node] = len(Chosen_researchers)

                Nodes_to_change_ranking_postion = list(Neighbors_in_ranking.keys())
                Ranking_positions = list(Neighbors_in_ranking.values())

                # Before iteration: [(A,0), (B,1), (C,2)]
                # After iteration: [(C,0), (A,1), (B,2)]

                # Change the ranking position of all the neighbors:
                for u in Nodes_to_change_ranking_postion[:-1]:
                    for r in Ranking_positions[1:]:
                        Chosen_researchers[u] = r

                # Move to the front in the new node the ranking of collaborators:
                Chosen_researchers[node] = Ranking_positions[0]

                Individuals_waiting_to_be_chosen = [
                    i for i in Individuals_waiting_to_be_chosen if i != node
                ]
                count += 1

        #             Total_size = len(Individuals_waiting_to_be_chosen)

        #         Chosen_researchers += Individuals_waiting_to_be_chosen
        New_ranking = {i: r - 1 for i, r in Chosen_researchers.items()}

    # Write the rank column
    Dataframe_ranking["Ranking_" + ranking_variable] = [
        New_ranking[i] + 1 for i in Dataframe_ranking.id
    ]

    return Dataframe_ranking


def mitigation_ranking(
    dataset,
    ranking_variable,
    sensitive_attribute,
    protected_attribute,
    graph,
    mitigation_method="Statistical_parity",
):
    """
    Ranks mitigation strategies based on specified parameters to reduce bias in a given dataset.

    Args:
        dataset (pd.DataFrame): The input dataset to be analyzed, typically a Pandas DataFrame.
        ranking_variable (str): The variable used for ranking the mitigation strategies.
        mitigation_method (str, optional): The method used to implement mitigation strategy.
                                            Default is "Statistical_parity".
        sensitive_attribute (str, optional): The attribute that may carry bias, such as "Gender" or "Race".
                                              Default is "Gender".
        protected_attribute (str, optional): The specific value of the sensitive attribute considered
                                             as protected (e.g., "female" for Gender). Default is "female".

    Returns:
        pd.DataFrame: A DataFrame containing the results of the mitigation strategy computation,
                       including rankings based on the specified mitigation method.

    Example:
        result_df = mitigation_ranking(my_dataset, 'income', 'Equalized_odds', 'Gender', 'male')

    Notes:
        This function utilizes the Compute_mitigation_strategy function to perform the actual
        mitigation ranking computation based on the given parameters.
    """

    # Call the computation function with the provided parameters to obtain ranked mitigations
    return Compute_mitigation_strategy(
        dataset,
        mitigation_method,
        ranking_variable,
        sensitive_attribute,
        protected_attribute,
        graph,
    )


def model_normal_ranking() -> ResearcherRanking:
    """
    Load and return a Normal Ranking of researchers.

    This function initializes the normal ranking model and returns
    an instance of the ResearcherRanking class containing the ranking data.

    Returns:
        ResearcherRanking: An instance of the ResearcherRanking class
        populated with normal ranking data.
    """
    return ResearcherRanking("Normal Ranking", normal_ranking)


def _mitigation_ranker_factory(method_name: str):
    """Returns a rank(df, ranking_variable, sensitive, protected, graph, **kwargs) callable."""

    def rank(
        df, ranking_variable, sensitive_attribute, protected_attribute, graph, **kwargs
    ):
        return Compute_mitigation_strategy(
            df,
            method_name,
            ranking_variable,
            sensitive_attribute,
            protected_attribute,
            graph,
        )

    return rank


@loader(namespace="csh", version="v003", python="3.11", packages=("networkx", "pandas"))
def model_breaking_network_ranking() -> ResearcherRanking:
    """
    Load the researcher ranking model incorporating a breaking network strategy.

    Here, we construct a new ranking that avoids placing connected researchers near each other
    in the ranking order. This is achieved by iteratively selecting researchers and ensuring that
    their neighbors in the collaboration network are not ranked closely together, thereby "breaking"
    clusters of connected researchers.
    """
    mitigation = _mitigation_ranker_factory("Breaking_network")
    return ResearcherRanking("Breaking Network", mitigation, normal_ranking)


@loader(namespace="csh", version="v003", python="3.11", packages=("networkx", "pandas"))
def model_reordering_network_ranking() -> ResearcherRanking:
    """
    Load the researcher ranking model incorporating a reordering network strategy.
    """
    mitigation = _mitigation_ranker_factory("Reordering_network")
    return ResearcherRanking("Reordering Network", mitigation, normal_ranking)


@loader(namespace="csh", version="v003", python="3.11", packages=("networkx", "pandas"))
def model_mitigation_ranking() -> ResearcherRanking:
    """
    Load the researcher ranking model incorporating a mitigation strategy.

    This function implements a fair ranking mechanism utilizing a sampling technique. It applies a
    mitigation strategy based on Statistical Parity, which aims to ensure equitable treatment
    across different groups by mitigating bias in the ranking process. Additionally, it compares
    the results of this fair ranking with a standard ranking derived from one of the numerical columns.

    Returns:
        ResearcherRanking: An instance of ResearcherRanking that contains both the mitigation-based
        ranking and the standard ranking for comparison.
    """
    mitigation = _mitigation_ranker_factory("Statistical_parity")

    # Invoke the ResearcherRanking constructor with both mitigation and normal rankings.
    return ResearcherRanking("Statistical Parity", mitigation, normal_ranking)


@loader(
    namespace="csh",
    version="v003",
    python="3.11",
    packages=(
        "pandas",
        "numpy",
        "hyperfair",
        "fairsearchcore",
    ),  # Mammoth will pip-install these
)
def model_hyperfair_ranking(
    alpha: float = 0.05,
    n_exp: int = 10_000,
    test_side: str = "lower",
    k_pc: float = 0.1,
) -> ResearcherRanking:
    """HyperFair-based Ranking Loader"""
    import pandas as pd
    from hyperfair.hyperfair import (
        adjust_ranking,
        measure_fairness_multiple_points,
        measure_fairness_single_point,
    )
    from hyperfair.data_loader import load_data_from_pandas_df

    def hyperfair_mitigation_strategy(
        df: pd.DataFrame,
        ranking_variable,
        sensitive_attribute,
        protected_attribute,
        **kwargs,
    ):
        sensitive_attr = sensitive_attribute
        protected_value = protected_attribute

        df_sorted = df.sort_values(ranking_variable, ascending=False).reset_index(
            drop=True
        )

        # Detect the two distinct labels (drop NaNs)
        labels = [v for v in df_sorted[sensitive_attr].dropna().unique()]
        if len(labels) != 2:
            raise ValueError(f"Expected a binary sensitive attribute, got {labels!r}")

        # Make the protected group = 1, the other = 0
        binary_dict = {
            labels[0]: 1 if labels[0] == protected_value else 0,
            labels[1]: 1 if labels[1] == protected_value else 0,
        }

        # ---- convert to hyperFAIR’s internal structure --------------------
        hf_data, ids = load_data_from_pandas_df(
            df_sorted,
            id_attribute="id",
            order_by=ranking_variable,
            protected_attribute=sensitive_attr,
            binary_dict=binary_dict,
        )

        k = math.floor(len(df_sorted) * float(k_pc)) or 1

        _, generatedData = measure_fairness_multiple_points(
            x_seq=hf_data,
            k=k,
            alpha=float(alpha),
            test_side="lower",
            n_exp=int(n_exp),
            verbose=False,
            plot=False,
        )

        # run the re-ranking
        new_data, new_ids, _ = adjust_ranking(
            hf_data,
            ids=ids,
            k=k,
            alpha=float(alpha),
            n_exp=int(n_exp),
            test_side=test_side,
        )

        df_final = (
            df_sorted.set_index("id", drop=False).loc[new_ids].reset_index(drop=True)
        )
        df_final[f"Ranking_{ranking_variable}"] = range(1, len(df_final) + 1)

        return df_final

    # Note: this relies on the closure giving the right values for k, n_exp etc
    return ResearcherRanking("Hyperfair", hyperfair_mitigation_strategy, normal_ranking)


@loader(
    namespace="csh", version="v003", python="3.11", packages=("networkx", "fairsearch")
)
def model_fair_ranking(
    alpha: float = 0.1, p: float = 0.25, k_pc: float = 0.1
) -> ResearcherRanking:
    """FA*IR mitigation using for minimum protected group representation in top-k."""
    from fairsearchcore import Fair
    from fairsearchcore.models import FairScoreDoc
    import pandas as pd

    def mitigation_strategy(
        df, ranking_variable, sensitive_attribute, protected_attribute, **kwargs
    ):
        sensitive_attribute = kwargs.get("sensitive_attribute", "Gender")
        protected_attribute = kwargs.get("protected_attribute", "female")

        df = df[~df[sensitive_attribute].isnull()].copy()
        df_sorted_full = df.sort_values(ranking_variable, ascending=False).reset_index(
            drop=True
        )

        k = math.floor(len(df_sorted_full) * float(k_pc)) or 10

        # The FAI*R library works on a list consisting of the top-k, so we must work with that subset
        df_sorted = df_sorted_full.head(k)

        docs = [
            FairScoreDoc(
                str(i),
                row[ranking_variable],
                row[sensitive_attribute] == protected_attribute,
            )
            for i, row in df_sorted.iterrows()
        ]

        fair = Fair(k, float(p), float(alpha))
        new_docs = fair.re_rank(docs)

        # TODO: why/when does this happen?
        if new_docs and isinstance(new_docs[0], list):
            new_docs = [d for block in new_docs for d in block]

        ranked_idx = [int(doc.id) for doc in new_docs]

        # Put the FA*IR ranked block on top
        df_fair = df_sorted.loc[ranked_idx].copy()  # these are the top-k
        df_rest = df_sorted_full.drop(index=ranked_idx)

        # Give every row a fresh rank
        df_final = pd.concat([df_fair, df_rest])
        df_final[f"Ranking_{ranking_variable}"] = range(1, len(df_final) + 1)
        df_final = df_final.reset_index(drop=True)

        df_final.reset_index(drop=True)
        return df_final

    return ResearcherRanking("FA*IR", mitigation_strategy, normal_ranking)
