from mammoth.datasets import Image
from mammoth.models import EmptyModel
from mammoth.exports import Markdown
from typing import List
from mammoth.integration import metric, Options
from cvbiasmitigation.suggest import analysis


@metric(
    namespace="mammotheu",
    version="v0036",
    python="3.11",
    packages=("torch", "torchvision", "cvbiasmitigation"),
)
def image_bias_analysis(
    dataset: Image,
    model: EmptyModel,
    sensitive: List[str],
    task: Options("face verification", "image classification") = None,
) -> Markdown:
    """
    This module provides a comprehensive solution for analyzing image bias and recommending effective
    mitigation strategies. It can be used for both classification tasks (e.g., facial attribute
    extraction) and face verification. The core functionality revolves around evaluating how well
    different population groups, defined by a given protected attribute (such as gender, age, or
    ethnicity), are represented in the dataset. Representation bias occurs when some groups are
    overrepresented or underrepresented, leading to models that may perform poorly or unfairly on
    certain groups.

    Additionally, the module detects spurious correlations between the target attribute (e.g., the
    label a model is trying to predict) and other annotated attributes (such as image features like
    color or shape). Spurious correlations are misleading patterns that do not reflect meaningful
    relationships and can cause a model to make biased or inaccurate predictions. By identifying and
    addressing these hidden biases, the module helps improve the fairness and accuracy of your model.

    When you run the analysis, the module identifies specific biases within the dataset and suggests
    tailored mitigation approaches. Specifically, the suitable mitigation methodologies are determined
    based on the task and the types of the detected biases in the data.
    The analysis is conducted based on the <a href="https://github.com/gsarridis/cv-bias-mitigation-library">CV Bias Mitigation Library</a>.


    Args:
        task: The type of predictive task. It should be either face verification or image classification.
    """

    assert task in [
        "face verification",
        "image classification",
    ], "The provided task should be either face verification or image classification"
    md = analysis(dataset.path, task, dataset.target, sensitive)
    return Markdown(md)
