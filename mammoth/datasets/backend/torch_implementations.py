from PIL import Image as PILImage
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from typing import List
import os


class PytorchImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        target: str,
        sensitive: List[str],
        data_transform: transforms.Compose,
    ):
        """
        PyTorch dataset for image data.

        Args:
            csv_path (str): The path to the CSV file containing information about the dataset.
            root_dir (str): The root directory where the actual image files are stored.
            target (str): The name of the column in the CSV file containing the target variable.
            sensitive (List[str]): A list of strings representing columns in the CSV file containing sensitive information.
            transforms (transforms.Compose): A composition of image transformations.
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.target = target
        self.sensitive = sensitive
        self.data_transform = data_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image = PILImage.open(img_path).convert("RGB")
        target = self.data.iloc[idx][self.target]
        protected = [self.data.iloc[idx][attr] for attr in self.sensitive]
        if self.data_transform is not None:
            image = self.data_transform(image)
        return image, target, protected


class PytorchImagePairsDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        target: str,
        sensitive: List[str],
        data_transform: transforms.Compose,
    ):
        """
        PyTorch dataset for image data.

        Args:
            csv_path (str): The path to the CSV file containing information about the dataset.
            root_dir (str): The root directory where the actual image files are stored.
            target (str): The name of the column in the CSV file containing the target variable.
            sensitive (List[str]): A list of strings representing columns in the CSV file containing sensitive information.
            transforms (transforms.Compose): A composition of image transformations.
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.target = target
        self.sensitive = sensitive
        self.data_transform = data_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_name = self.data.iloc[idx, 0]  # ref
        img2_name = self.data.iloc[idx, 1]  # motion

        id1_image_path = os.path.join(self.root_dir, img1_name)
        id2_image_path = os.path.join(self.root_dir, img2_name)

        image1 = PILImage.open(id1_image_path).convert("RGB")
        image2 = PILImage.open(id2_image_path).convert("RGB")

        target = self.data.iloc[idx][self.target]
        protected = [self.data.iloc[idx][attr] for attr in self.sensitive]
        if self.data_transform is not None:
            image1 = self.data_transform(image1)
            image2 = self.data_transform(image2)
        return image1, image2, target, protected
