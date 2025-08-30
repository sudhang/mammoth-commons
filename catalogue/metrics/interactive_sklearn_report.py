import numpy as np

from mammoth.datasets import CSV
from mammoth.models import EmptyModel
from mammoth.exports import HTML
from typing import Dict, List
from mammoth.integration import metric, Options
import fairbench as fb
import sklearn
import numpy as np


@fb.v1.core.Transform
def categories(iterable):
    # print(iterable)
    is_numeric = True
    values = list()
    for value in iterable:
        try:
            values.append(float(value))
        except Exception:
            is_numeric = False
            break
    if is_numeric:
        values = fb.v1.tobackend(values)
        mx = values.max()
        mn = values.min()
        if mx == mn:
            raise Exception(
                "Numerical sensitive attribute has the same value everywhere"
            )
        values = (values - mn) / (mx - mn)
        return {f"fuzzy min ({mn:.3f})": 1 - values, f"fuzzy max ({mx:.3f})": values}
    return fb.categories @ iterable


@metric(
    namespace="mammotheu",
    version="v0036",
    python="3.11",
    packages=(
        "fairbench",
        "scikit-learn",
        "pandas",
        "onnxruntime",
        "ucimlrepo",
        "pygrank",
    ),
)
def interactive_sklearn_report(
    dataset: CSV,
    model: EmptyModel,
    sensitive: List[str],
    predictor: Options("Logistic regression", "Gaussian naive Bayes") = None,
    intersectional: bool = False,
    compare_groups: Options("Pairwise", "To the total population") = None,
    view: Options(
        "Fairness model card",
        "Detailed description",
        "Summary table",
    ) = None,
    minimum_shown_deviation: float = 0,
) -> HTML:
    """Creates an interactive report using the FairBench library, after running an internal training-test split
    on a basic sklearn model. The report creates traceable evaluations that you can shift through to find sources
    of unfairness on a common task.

    Args:
        predictor: Which sklearn predictor should be used.
        intersectional: Whether to consider all non-empty group intersections during analysis. This does nothing if there is only one sensitive attribute.
        compare_groups: Whether to compare groups pairwise, or each group to the whole population.
        view: How to display results. You can choose to view a fairness model card which does not have too many details, a full report, or a summary table.
        minimum_shown_deviation: Show only results where the deviation from ideal values exceeds the given threshold. If nothing is shown, it does not mean that fairness is achieved, but this is a good way to identify the most prominent biases. If value of 0 is set (default) then all results are shown.
    """
    assert len(sensitive) != 0, "Set at least one sensitive attribute"
    X = dataset.to_features(sensitive)
    y = dataset.labels
    if isinstance(y, dict):
        y = y[list(y.keys())[0]]
    else:
        assert (
            y.shape[1] <= 2
        ), "Cannot create an interactive report for non-binary predictions"
        y = y[y.columns[-1]]
    from sklearn import model_selection

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _,
        idx_test,
    ) = model_selection.train_test_split(
        X, y, np.arange(0, y.shape[0], dtype=np.int64), test_size=0.2
    )
    if predictor == "Logistic regression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=10000)
    else:
        from sklearn.naive_bayes import GaussianNB

        model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]

    # declare sensitive attributes
    sensitive = fb.Dimensions(
        {attr + " ": (categories @ dataset.data[attr][idx_test]) for attr in sensitive}
    )

    # change behavior based on arguments
    if intersectional:
        sensitive = sensitive.intersectional()
    report_type = (
        fb.reports.pairwise if compare_groups == "Pairwise" else fb.reports.vsall
    )

    report = report_type(
        predictions=predictions,
        labels=y_test.to_numpy(),
        scores=scores,
        sensitive=sensitive,
    )
    minimum_shown_deviation = float(minimum_shown_deviation)
    assert (
        0 <= minimum_shown_deviation <= 1
    ), "Minimum shown deviation should be in the range [0,1]"
    if minimum_shown_deviation != 0:
        report = report.filter(fb.investigate.DeviationsOver(minimum_shown_deviation))

    if view == "Summary table":
        ret = report.show(env=fb.export.HtmlTable(view=False, filename=None))
    elif view == "Fairness model card":
        ret = report.filter(fb.investigate.Stamps).show(
            env=fb.export.Html(view=False, filename=None), depth=1
        )
    else:
        ret = report.show(env=fb.export.Html(view=False, filename=None))
    return HTML(ret)
