from mammoth.models import ONNX
from mammoth.integration import loader
import urllib
import os


@loader(
    namespace="mammotheu", version="v0036", python="3.12", packages=("onnxruntime",)
)
def model_onnx(path: str = "") -> ONNX:
    """Loads an inference model stored in <a href="https://onnx.ai/">ONNx</a> format.
    This is a generic cross-platform format for representing machine learning models with a common set of operations.
    Several machine learning frameworks can export to this format.
    The loaded model should be compatible with the dataset being analysed. For example,
    the same data columns as in the dataset should be used for training on tabular data.

    ONNx supports several different runtimes, but this loader's implementation selects
    the `CPUExecutionProvider` runtime to run on, therefore maintaining compatibility
    with most machines.
    For inference in GPUs, prefer storing and loading models in formats
    that are guaranteed to maintain all features that could be included in the architectures
    of respective frameworks; this can be achieved with different model loaders.

    Here are some quick links on how to export ONNx models from popular frameworks:
    <ul>
    <li><a href="https://onnx.ai/sklearn-onnx">scikit-learn</a></li>
    <li><a href="https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html">PyTorch</a></li>
    <li><a href="https://onnxruntime.ai/docs/tutorials/tf-get-started.html">TensorFlow</a></li>
    </ul>

    Args:
        path: A local path or url pointing to the loaded file. The loader checks for the existence of the local path, and if it does not exist the `urllib.request` module reads the model's bytes from a URL.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            model_bytes = f.read()
    else:
        with urllib.request.urlopen(path) as f:
            model_bytes = f.read()
    return ONNX(model_bytes)
