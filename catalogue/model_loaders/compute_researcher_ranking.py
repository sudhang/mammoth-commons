from functools import partial
import math
from mammoth.integration import loader
from mammoth.models.researcher_ranking import ResearcherRanking
import random

import fairsearchcore
from fairsearchcore import Fair
from fairsearchcore.models import FairScoreDoc

import pandas as pd

from hyperfair.hyperfair import adjust_ranking, measure_fairness_multiple_points, measure_fairness_single_point
from hyperfair.data_loader import load_data_from_pandas_df
import numpy as np


def normal_ranking(dataset, ranking_variable):
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
):
    """Function for several mitigation strategies"""

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

    
    # TODO: Remove all other mitigation_methods
    if mitigation_method == "Statistical_parity":
        # Chosen_groups would be a list with a desired ranking of group members
        # eg: ["female", "male", "male", "female", ...]
        Chosen_groups = []
        Len_group_in_ranking = Len_groups
        for i in range(Dataframe_ranking.shape[0]):               # Go through every rank slot

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

    # Determine which positions each group will occupy
    Positions = {
        non_protected_attribute: [
            i for i, j in enumerate(Chosen_groups) if j == non_protected_attribute
        ],
        protected_attribute: [
            i for i, j in enumerate(Chosen_groups) if j == protected_attribute
        ],
    }

    # Pick concrete researcher IDs to fill the positions from above
    Chosen_researchers = {
        i_ranking: Ranking_sets[non_protected_attribute].iloc[i_position]["id"]
        for i_position, i_ranking in enumerate(Positions[non_protected_attribute])
    }
    for i_position, i_ranking in enumerate(Positions[protected_attribute]):
        Chosen_researchers[i_ranking] = Ranking_sets[protected_attribute].iloc[
            i_position
        ]["id"]

    New_ranking = {r: i for i, r in Chosen_researchers.items()}

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
    mitigation_method="Statistical_parity"
):
    return Compute_mitigation_strategy(
        dataset,
        mitigation_method,
        ranking_variable,
        sensitive_attribute,
        protected_attribute,
    )


def model_normal_ranking() -> ResearcherRanking:
    """This is a Normal Ranking loader"""

    return ResearcherRanking(normal_ranking)


@loader(namespace="csh", version="v003", python="3.11", packages=("networkx", "hyperfair"))
def model_mitigation_ranking() -> ResearcherRanking:
    """This is a fair Ranking loader with Sampling. In this model, we will use a mitigation strategy based on Statistical Parity, and compare it with a normal ranking based on one of the Numerical columns"""
    return ResearcherRanking(mitigation_ranking, normal_ranking)

@loader(
    namespace="csh",
    version="v003",
    python="3.11",
    packages=("pandas", "numpy", "hyperfair")   # Mammoth will pip-install these
)
def model_hyperfair_ranking(
    alpha: float = 0.05,
    n_exp: int = 10_000,
    test_side: str = "lower",
    k_pc: float = 0.1

) -> ResearcherRanking:
    """HyperFair-based Ranking Loader"""

    def hyperfair_mitigation_strategy(
            df: pd.DataFrame, 
            ranking_variable,
            sensitive_attribute,
            protected_attribute,
            **kwargs):
        sensitive_attr   = sensitive_attribute
        protected_value  = protected_attribute

        df_sorted = df.sort_values(ranking_variable, ascending=False).reset_index(drop=True)

        # Detect the two distinct labels (drop NaNs)
        labels = [v for v in df_sorted[sensitive_attr].dropna().unique()]
        if len(labels) != 2:
            raise ValueError(
                f"Expected a binary sensitive attribute, got {labels!r}"
            )

        # Make the protected group = 1, the other = 0
        binary_dict = {
            labels[0]: 1 if labels[0] == protected_value else 0,
            labels[1]: 1 if labels[1] == protected_value else 0,
        }


        # ---- convert to hyperFAIR’s internal structure --------------------
        hf_data, ids = load_data_from_pandas_df(
            df_sorted,
            id_attribute="id",
            order_by = ranking_variable,
            protected_attribute = sensitive_attr,
            binary_dict=binary_dict
        )

        k = math.floor(len(df_sorted) * float(k_pc)) or 1

        _, generatedData = measure_fairness_multiple_points(
            x_seq=hf_data, 
            k=k, alpha=float(alpha), 
            test_side='lower', 
            n_exp=int(n_exp), 
            verbose=True, 
            plot=False
        )


        # run the re-ranking
        new_data, new_ids, _ = adjust_ranking(
            hf_data,
            ids=ids,
            k=k,
            alpha     = float(alpha),
            n_exp     = int(n_exp),
            test_side = test_side,
        )

        df_final = df_sorted.set_index("id", drop=False).loc[new_ids].reset_index(drop=True)
        df_final[f"Ranking_{ranking_variable}"] = range(1, len(df_final) +1)

        return df_final

    # Note: this relies on the closure giving the right values for k, n_exp etc
    return ResearcherRanking(hyperfair_mitigation_strategy, normal_ranking)

@loader(namespace="csh", version="v003", python="3.11", packages=("networkx", "fairsearch"))
def model_fair_ranking(
    alpha: float = 0.1,
    p: float = 0.25,
    k_pc: float = 0.1
) -> ResearcherRanking:
    """FA*IR mitigation using for minimum protected group representation in top-k."""

    def mitigation_strategy(
            df, 
            ranking_variable, 
            sensitive_attribute,
            protected_attribute,
            **kwargs):
        sensitive_attribute = kwargs.get("sensitive_attribute", "Gender")
        protected_attribute = kwargs.get("protected_attribute", "female")

        df = df[~df[sensitive_attribute].isnull()].copy()
        df_sorted_full = df.sort_values(ranking_variable, ascending=False).reset_index(drop=True)


        k = math.floor(len(df_sorted_full) * float(k_pc)) or 10

        # The FAI*R library works on a list consisting of the top-k, so we must work with that subset
        df_sorted = df_sorted_full.head(k)

        docs = [
            FairScoreDoc(str(i), row[ranking_variable], row[sensitive_attribute] == protected_attribute)
            for i, row in df_sorted.iterrows()
        ]

        fair = Fair(k, float(p), float(alpha))
        new_docs = fair.re_rank(docs)

        # TODO: why/when does this happen?
        if new_docs and isinstance(new_docs[0], list):
            new_docs = [d for block in new_docs for d in block]

        ranked_idx = [int(doc.id) for doc in new_docs]

        # Put the FA*IR ranked block on top
        df_fair = df_sorted.loc[ranked_idx].copy()   # these are the top-k
        df_rest = df_sorted_full.drop(index=ranked_idx)

        # Give every row a fresh rank
        df_final = pd.concat([df_fair, df_rest])
        df_final[f"Ranking_{ranking_variable}"] = range(1, len(df_final) +1)
        df_final = df_final.reset_index(drop=True)

        df_final.reset_index(drop=True)
        return df_final

    return ResearcherRanking(mitigation_strategy, normal_ranking)