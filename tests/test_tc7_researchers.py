from mammoth_commons import testing
from mai_bias.catalogue.dataset_loaders.data_researchers import data_researchers
from mai_bias.catalogue.model_loaders.compute_researcher_ranking import (
    model_mitigation_ranking,
    model_fair_ranking,
    model_hyperfair_ranking,
    model_breaking_network_ranking,
    model_reordering_network_ranking,
)
from mai_bias.catalogue.metrics.ranking_fairness import exposure_distance_comparison


def test_researchers_ranking_comparison():
    with testing.Env(
        data_researchers,
        model_mitigation_ranking,
        model_breaking_network_ranking,
        model_reordering_network_ranking,
        model_fair_ranking,
        model_hyperfair_ranking,
        exposure_distance_comparison,
    ) as env:
        dataset = env.data_researchers(
            paper_graph_path="./data/researchers/physics_papers.csv.tar.bz2",
            paper_affiliations_path="./data/researchers/affiliations.csv.tar.bz2",
        )

        # model_mitigation = env.model_mitigation_ranking()
        # model_mitigation = env.model_fair_ranking()
        # model_mitigation = env.model_hyperfair_ranking()
        # model_mitigation = env.model_breaking_network_ranking()
        model_mitigation = env.model_reordering_network_ranking()

        analysis_outcome_mitigation = env.exposure_distance_comparison(
            dataset,
            model_mitigation,
            n_runs=10,
            sampling_attribute="Academic_Stage",
            ranking_variable="Citations",
            sensitive=["Gender"],
        )
        analysis_outcome_mitigation.show()


if __name__ == "__main__":
    test_researchers_ranking_comparison()
