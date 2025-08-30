import os
from mammoth.models.pytorch import Pytorch
from mammoth.integration import loader
from mammoth.externals import safeexec
import torch


@loader(namespace="mammotheu", version="v0036", python="3.11")
def model_torch(
    state_path: str = "",
    model_path: str = "",
    model_name: str = "model",
    safe_libraries: str = "torch, torchvision",
) -> Pytorch:
    """Loads a pytorch model that comprises a Python code initializing the
    architecture and a file of trained parameters. For safety, the architecture's
    definition is allowed to directly import only specified libraries.

    Args:
        state_path: The path in which the architecture's state is stored.
        model_path: The path in which the architecture's initialization script resides. Alternatively, you may also just paste the initialization code in this field.
        model_name: The variable in the model path's script to which the architecture is assigned.
        safe_libraries: A comma-separated list of libraries that can be imported.
    """

    model = safeexec(
        model_path,
        out=model_name,
        whitelist=[lib.strip() for lib in safe_libraries.split(",")],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(state_path, map_location=device))

    return Pytorch(model)
