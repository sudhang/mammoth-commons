from typing import List
from mammoth.datasets import Dataset


class ImagePairs(Dataset):
    def __init__(
        self,
        path,
        root_dir,
        target,
        data_transform,
        batch_size,
        shuffle,
        cols,
    ):
        """
        Args:
            path (str): Path to the CSV file with annotations (should involve the columns img1_name|img2_name|attribute1|...|attributeN).
            root_dir (str): Root image dataset directory (eg the db_path for the UC2).
            target (str): The target attribute to be predicted (eg the attack for UC2).
            data_transform (callable): A function/transform that takes in an image and returns a transformed version.
            batch_size (int): How many samples per batch to load.
            shuffle (bool): Set to True to have the data reshuffled every time they are obtained.
        """

        self.path = path
        self.root_dir = root_dir
        self.target = target
        self.data_transform = data_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cols = cols
        self.input_size = self._get_input_size(self.data_transform)

    def to_torch(self, sensitive: List[str]):
        # dynamic dependencies here to not force a torch dependency on commons from components that don't need it
        from torch.utils.data import DataLoader
        from mammoth.datasets.backend.torch_implementations import (
            PytorchImagePairsDataset,
        )

        torch_dataset = PytorchImagePairsDataset(
            csv_path=self.path,
            root_dir=self.root_dir,
            target=self.target,
            sensitive=sensitive,
            data_transform=self.data_transform,
        )

        return DataLoader(
            dataset=torch_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def to_numpy(self, sensitive: List[str]):
        from mammoth.datasets.backend.onnx_transforms import torch2onnx
        from mammoth.datasets.backend.onnx_implementations import (
            ONNXImagePairsDataset,
            numpy_dataloader_imagepairs,
        )

        onnx_transforms = torch2onnx(self.data_transform)
        dataset = ONNXImagePairsDataset(
            csv_path=self.path,
            root_dir=self.root_dir,
            target=self.target,
            sensitive=sensitive,
            data_transform=onnx_transforms,
        )

        return numpy_dataloader_imagepairs(
            dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def _get_input_size(self, transform):
        from torchvision import transforms

        # Check for Resize transform in the composition
        for t in transform.transforms:
            if isinstance(t, transforms.Resize):
                return t.size  # Return the size of the Resize transform
        # Default to a common size if Resize is not found
        return (224, 224)  # Default size
