import os
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import onnxruntime as ort
import tempfile


class TransformWrapper(nn.Module):
    def __init__(self, transform):
        super(TransformWrapper, self).__init__()
        self.transform = transform
        self.input_size = self._get_input_size()
        self.transforms_dict = {}

        # Extract individual transforms dynamically
        for t in self.transform.transforms:
            transform_name = t.__class__.__name__.lower()
            self.transforms_dict[transform_name] = t
        # print(self.transforms_dict.keys())

    def _get_input_size(self):
        # Check for Resize transform in the composition
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                return t.size  # Return the size of the Resize transform
        # Default to a common size if Resize is not found
        return (224, 224)  # Default size

    def forward(self, x):
        # Apply each transform in sequence
        for name, transform in self.transforms_dict.items():
            if name != "totensor":
                x = transform(x)
        return x


def torch2onnx(transforms):
    # Create a wrapper model for the transforms
    transform_model = TransformWrapper(transforms)
    transform_model.eval()
    # Create a dummy input based on the extracted input size
    dummy_input = torch.randn(1, 3, 891, 891)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
        onnx_model_path = temp_file.name
        # Export the model to ONNX format and save it to the temporary file
        torch.onnx.export(
            transform_model,
            dummy_input,
            onnx_model_path,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size"},
            },
        )

    # Load the ONNX model using ONNX Runtime
    onnx_transforms = ort.InferenceSession(onnx_model_path)

    # Optionally delete the temporary file after loading the model
    os.remove(onnx_model_path)
    return onnx_transforms
