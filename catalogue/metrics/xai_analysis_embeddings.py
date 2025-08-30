from mammoth.datasets import ImagePairs
from mammoth.models.pytorch import Pytorch
from mammoth.exports import HTML
from typing import List
from mammoth.integration import metric

# install facex lib using: pip install facextool
from facex.component import run_embeddings_mammoth


@metric(
    namespace="mammotheu",
    version="v0036",
    python="3.11",
    packages=("torch", "torchvision", "timm", "facextool"),
)
def facex_embeddings(
    dataset: ImagePairs,
    model: Pytorch,
    sensitive: List[str],
    target_class: int = None,
    target_layer: str = None,
) -> HTML:
    """<a href="https://github.com/gsarridis/faceX">FaceX</a> for feature extractors is designed to help you
        understand how face verification models process
        images by comparing feature embeddings. In tasks like face recognition, models
        generate a feature vector (embedding) for each image. These embeddings capture the unique
        characteristics of the image and can be compared to determine how similar or different two images
        are. FaceX helps explain which parts of an image contribute to its similarity or difference to a
        reference embedding, allowing you to understand the model's focus on specific facial
        features during the comparison.

        In this context, the key idea is that FaceX analyzes the facial regions in the image that most
        influence how the model compares the reference embedding with the new image's embedding. Rather than
        providing an explanation for individual images in isolation, FaceX aggregates information across the
        dataset, offering insights into how different parts of the face, such as the eyes, mouth, or hair,
        contribute to the similarity or difference between the reference and the image being compared.

        FaceX works by using a "reference" image (e.g., identity image) and evaluating how other images
        (e.g., selfies) compare to it.
        It looks at how various facial regions of the new image align with the reference image in the
        feature space. The tool then highlights which facial regions are most influential in the comparison,
        showing what aspects of the image are similar to or
        different from the concept embedding.

        Overall, FaceX gives you a better understanding of how the model processes and compares facial
        features by highlighting the specific regions that influence the similarity or difference in feature
        embeddings. This is especially useful for improving transparency and identifying potential biases in
        how face verification models represent and compare faces.

        <span class="alert alert-warning alert-dismissible fade show" role="alert"
            style="display: inline-block; padding: 10px;"> <i class="bi bi-exclamation-triangle-fill"></i> GPU
            access is recommended as this analysis can be computationally intensive, especially with large
            datasets. </span>

    Args:
        target_class: This variable determines what kind of comparison you want to explore. If you set the target class to 1, FaceX will investigate the regions in the image that are similar to the reference embedding (i.e., explanations on why the model consider the two images similar). If you set the target class to 0, FaceX will show the regions that are most different from the reference embedding (i.e., explanations on why the model consider the two images dissimilar).
        target_layer: This parameter lets you choose which part of the model's neural network you want to analyze. In simple terms, a model consists of multiple layers that process information at different levels. The target layer refers to the specific layer in the model that you want to explain. The explanation will show you which regions of the face are most important to that layer's decision-making process. For example, the deeper layers of the model may focus on more complex features like facial structure, while earlier layers might focus on simpler features like edges and textures. Typically, you should opt for the last layer producing the final embeddings.
    """

    target_class = int(target_class)
    html = run_embeddings_mammoth(
        dataset, sensitive[0], target_class, model.model, target_layer
    )

    return HTML(html)
