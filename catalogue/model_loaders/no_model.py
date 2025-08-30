from mammoth.models import EmptyModel
from mammoth.integration import loader


@loader(namespace="mammotheu", version="v0036", python="3.11")
def no_model() -> EmptyModel:
    """Signifies that the analysis should focus solely on the fairness of the dataset."""

    return EmptyModel()
