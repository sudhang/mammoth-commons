from mammoth.models import NodeRanking
from mammoth.integration import loader, Options


@loader(namespace="mammotheu", version="v0036", python="3.11", packages=("pygrank",))
def model_fair_node_ranking(
    diffusion: float = 0.85,
    redistribution: Options("none", "uniform", "original") = "original",
) -> NodeRanking:
    """
    Constructs a node ranking algorithm that is a variation non-personalized PageRank.
    The base algorithm is often computes a notion of centrality/structural
    importance for each node in the graph, and employs a diffusion parameter in the range [0, 1).
    Find more details on how the algorithm works based on the following seminal paper:

    <i>Page, L. (1999). The PageRank citation ranking: Bringing order to the web. Technical Report.</i>

    The base node ranking algorithm is enriched by fairness-aware interventions implemented
    by the <a href="https://pygrank.readthedocs.io/en/latest/">pygrank</a> library. The latter
    may run on various computational backends, but `numpy` is selected due to its compatibility
    with a broad range of software and hardware. All implemented algorithms transfer node score
    mass from over-represented groups of nodes to those with lesser average mass using different
    strategies that determine the redistribution details. Fairness is imposed in terms of centrality
    scores achieving similar score mass between groups. The three available strategies are
    described here:

    <ul>
    <li>`none` does not employ any fairness intervention and runs the base algorithm.</li>
    <li>`uniform` applies a uniform rank redistribution strategy.</li>
    <li>`original` tries to preserve the order of original node ranks by distributing more score mass to those.</li>
    </ul>

    Args:
        diffusion: The diffusion parameters of the corresponding PageRank algorithm.
        redistribution: The redistribution strategy. Can be none, uniform or original.
    """
    import pygrank as pg

    assert redistribution in [
        "none",
        "original",
        "uniform",
    ], "Invalid node score redistribution strategy."
    diffusion = float(diffusion)
    assert diffusion >= 0, "The diffusion should be non-negative"
    assert diffusion < 1, "The diffusion should be <1"  # careful not to allow 1

    params = {"alpha": diffusion, "tol": 1.0e-9, "max_iters": 3000}
    ranker = (
        pg.PageRank(**params)
        if redistribution == "none"
        else pg.LFPR(redistributor=redistribution, **params)
    )
    return NodeRanking(ranker >> pg.Normalize("max"))
