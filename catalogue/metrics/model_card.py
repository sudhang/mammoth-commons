from mammoth.datasets import Dataset
from mammoth.models import Predictor
from mammoth.exports import Markdown
from typing import Dict, List
from mammoth.integration import metric, Options
from fairbench import v1 as fb
import numpy as np


@fb.core.Transform
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
        values = np.array(values)
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
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def model_card(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersectional: bool = False,
    compare_groups: Options("Pairwise", "To the total population") = None,
) -> Markdown:
    """Creates a model card using the <a href="https://github.com/mever-team/FairBench">FairBench</a>
    library. The card includes several fairness stamps; these are specific measures of bias
    or fairness that are commonly used in the algorithmic fairness literature. Only the most prominent
    of those measures are used as stamps, and they correspond to a perfunctory fairness analysis.

    This module computes all applicable FairBench stamps, which
    summarize behavior across all population groups or intersectional
    subgroups.
    Multiple sensitive attributes may be present, such as gender, age, and race.
    Furthermore, each of those attributes may obtain multiple values, as happens when multiple genders or
    races are considered. Numeric attributes, like age, are normalized to
    the range [0,1] and we consider the result as truth values of membership to the group of the maximum
    value - as opposed to membership to the group with minimum value.
    A different stamp is computed for each prediction label.

    You may optionally create intersectional subgroups, that is, create
    a separate subgroup for each combination of sensitive attribute values. Many of those groups will have few
    members if there are too many attributes, and empty groups are ignored during the analysis.

    The created model card contains exact descriptions of methods used to compute fairness under
    the selected stamps, and it lists population groups that were taken into account
    These come alongside an extensive list of
    caveats and recommendations that help the reader get a grasp on how they should
    account for the social context. This material is retrieved from FairBench's
    online socio-technical database generated through MAMMOth's multidisciplinary activities.

    Finally, the generated model card may contain details about out-of-the-box datasets.
    To get the full picture, a detailed fairness report that also allows you to backtrack computations
    is available in the `interactive report` module.

    Args:
        intersectional: Whether to consider all non-empty group intersections during analysis. This does nothing if there is only one sensitive attribute, but may also be computationally intensive if too many group intersections are selected.
        compare_groups: Whether to compare groups pairwise, or each group to the whole population. For example, if the 4/5ths rule stamp is applicable, it computes positive rates and obtains the minimum ratio, either across all pairs of groups (for pairwise comparison) or otherwise between each group and the total population.
    """

    text = ""

    if len(sensitive) == 0:
        raise Exception("At least one sensitive attribute should be selected")

    # obtain predictions
    predictions = model.predict(dataset, sensitive)

    # declare sensitive attributes
    labels = dataset.labels
    sensitive = fb.Fork({attr: categories @ dataset.data[attr] for attr in sensitive})

    # change behavior based on arguments
    if intersectional:
        sensitive = sensitive.intersectional()
    report_type = fb.multireport if compare_groups == "Pairwise" else fb.unireport
    # perform different analysis, depending on whether labels are provided
    if labels is None:
        report = report_type(predictions=predictions, sensitive=sensitive)
        stamps = fb.combine(
            fb.stamps.prule(report),
            fb.stamps.four_fifths(report),
        )
        text += fb.modelcards.tomarkdown(stamps)
    else:
        for label in labels:
            # TODO: the following analysis is only for one class label
            report = report_type(
                predictions=predictions,
                labels=(
                    labels[label].to_numpy()
                    if hasattr(labels[label], "to_numpy")
                    else labels[label]
                ),
                sensitive=sensitive,
            )
            stamps = fb.combine(
                fb.stamps.prule(report),
                fb.stamps.accuracy(report),
                fb.stamps.four_fifths(report),
                fb.stamps.dfpr(report),
                fb.stamps.dfnr(report),
                # fb.stamps.auc(report),
                # fb.stamps.abroca(report),
            )
        text += fb.modelcards.tomarkdown(stamps)

    if hasattr(dataset, "description"):
        text += "\n## Dataset\n"
        if isinstance(dataset.description, str):
            text += text + "\n"
        elif isinstance(dataset.description, dict):
            for key, value in dataset.description.items():
                text += "#### " + key + "\n" + value.replace("\n", "\n\n") + "\n"
        else:
            raise Exception(
                "Since the dataset's description field exist, it should be either string or dict from headers to descriptions"
            )
    return Markdown(text)
