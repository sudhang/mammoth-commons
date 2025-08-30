from mammoth.datasets import Image
from mammoth.models.pytorch import Pytorch
from mammoth.exports import HTML
from typing import List
from mammoth.integration import metric

# install facex lib using: pip install facextool
from facex.component import run_mammoth


@metric(
    namespace="mammotheu",
    version="v0036",
    python="3.11",
    packages=("torch", "torchvision", "timm", "facextool"),
)
def facex_regions(
    dataset: Image,
    model: Pytorch,
    sensitive: List[str],
    target_class: int = None,
    target_layer: str = None,
) -> HTML:
    """<a href="https://github.com/gsarridis/faceX">FaceX</a> is a tool designed to help you understand how
        face attribute classifiers make decisions. It
        provides clear explanations by analyzing 19 key regions of the face, such as the eyes, nose, mouth,
        hair, and skin. This method helps reveal which parts of the face the model focuses on when making
        predictions about attributes like age, gender, or race.

        Rather than explaining each individual image separately, FaceX aggregates information across the
        entire dataset, offering a broader view of the model's behavior. It looks at how the model activates
        different regions of the face for each decision. This aggregated information helps
        you see which facial features are most influential in the model's predictions, and whether certain
        features are being emphasized more than others.

        In addition to providing an overall picture of which regions are important, FaceX also zooms in on
        specific areas of the face - such as a section of the skin or a part of the hair - showing which
        patches of the image have the highest impact on the model's decision. This makes it easier to
        identify potential biases or problems in how the model is interpreting the face.

        Overall, with FaceX, you can quickly and easily get a better understanding of your model's
        decision-making process. This is especially useful for ensuring that your model is fair and
        transparent, and for spotting any potential biases that may
        affect its performance.

        <span class="alert alert-warning alert-dismissible fade show" role="alert"
        style="display: inline-block; padding: 10px;"> <i class="bi bi-exclamation-triangle-fill"></i> GPU
        access is recommended as this analysis can be computationally intensive, especially with large
        datasets. </span>


    Args:
        target_class: This parameter allows you to specify which class you want to analyze. For example, if you are using the model to classify faces by gender, setting the target class to male (i.e., the integer identifier for males class) will show you how the model makes decisions about male faces.
        target_layer: This parameter lets you choose which part of the model's neural network you want to analyze. In simple terms, a model consists of multiple layers that process information at different levels. The target layer refers to the specific layer in the model that you want to explain. The explanation will show you which regions of the face are most important to that layer's decision-making process. For example, the deeper layers of the model may focus on more complex features like facial structure, while earlier layers might focus on simpler features like edges and textures. Typically, you should opt for the last layer prior to the classification layer.
    """

    target_class = int(target_class)
    html = run_mammoth(dataset, sensitive[0], target_class, model.model, target_layer)
    html = (
        """
    <h1>Image Explanations</h1>
    <p>FaceX analysed 19 facial regions and accessories to provide explanations. In the two illustrations below,
    left are face regions and right are hat and glasses. Blue are the least important regions and red the most
    important ones that are taken into account. Based on the outputs, try to the question of “where does a model
    focus on?”. We also show high-impact patches to help understand “what visual features trigger its focus?”.</p>
    """
        + html
    )
    return HTML(html)
