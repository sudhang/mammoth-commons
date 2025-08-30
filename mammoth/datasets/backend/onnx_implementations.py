import io
from PIL import Image as PILImage
import numpy as np
import pandas as pd
from typing import List
import os


class ONNXImageDataset:
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        target: str,
        sensitive: List[str],
        data_transform: io.BytesIO,
    ):
        """
        ONNX dataset for image data.

        Args:
            csv_path (str): The path to the CSV file containing information about the dataset.
            root_dir (str): The root directory where the actual image files are stored.
            target (str): The name of the column in the CSV file containing the target variable.
            sensitive (List[str]): A list of strings representing columns in the CSV file containing sensitive information.
            transforms (onnx): A composition of image transformations.
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.target = target
        self.sensitive = sensitive
        self.data_transform = data_transform
        # Initialize ONNX runtime session
        self.ort_session = data_transform
        # print(self.ort_session.get_inputs())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image = PILImage.open(img_path).convert("RGB")
        target = self.data.iloc[idx][self.target]
        protected = [self.data.iloc[idx][attr] for attr in self.sensitive]
        if self.data_transform is not None:
            image = self.apply_transform(image)
        return image, target, protected

    def apply_transform(self, image: PILImage) -> np.ndarray:
        """
        Apply the ONNX transformation to the image.

        Args:
            image (PILImage): The image to be transformed.

        Returns:
            np.ndarray: The transformed image as a NumPy array.
        """
        # Convert the image to a NumPy array and preprocess as needed

        image_np = np.array(image).astype(np.float32)  # Convert to float32
        # print(image_np.shape)
        image_np = image_np.transpose(2, 0, 1)  # Change from HWC to CHW format
        image_np = image_np[np.newaxis, ...]  # Add batch dimension
        # print(image_np.shape)
        # Run inference with the ONNX model
        ort_inputs = {self.ort_session.get_inputs()[0].name: image_np}
        transformed_image = self.ort_session.run(None, ort_inputs)
        # print(transformed_image[0].shape)
        return transformed_image[0]  # Assuming the output is in the first position


class ONNXImagePairsDataset:
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        target: str,
        sensitive: List[str],
        data_transform: str,
    ):
        """
        ONNX dataset for image data.

        Args:
            csv_path (str): The path to the CSV file containing information about the dataset.
            root_dir (str): The root directory where the actual image files are stored.
            target (str): The name of the column in the CSV file containing the target variable.
            sensitive (List[str]): A list of strings representing columns in the CSV file containing sensitive information.
            transforms (onnx): A composition of image transformations.
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.target = target
        self.sensitive = sensitive
        self.data_transform = data_transform
        self.ort_session = data_transform

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
            image1 = self.apply_transform(image1)
            image2 = self.apply_transform(image2)
        return image1, image2, target, protected

    def apply_transform(self, image: PILImage) -> np.ndarray:
        """
        Apply the ONNX transformation to the image.

        Args:
            image (PILImage): The image to be transformed.

        Returns:
            np.ndarray: The transformed image as a NumPy array.
        """
        # Convert the image to a NumPy array and preprocess as needed
        image_np = np.array(image).astype(np.float32)  # Convert to float32
        image_np = image_np.transpose(2, 0, 1)  # Change from HWC to CHW format
        image_np = image_np[np.newaxis, ...]  # Add batch dimension

        # Run inference with the ONNX model
        ort_inputs = {self.ort_session.get_inputs()[0].name: image_np}
        transformed_image = self.ort_session.run(None, ort_inputs)
        return transformed_image[0]  # Assuming the output is in the first position


def numpy_dataloader_image(
    dataset: ONNXImageDataset, batch_size: int = 32, shuffle: bool = False
):
    """
    A generator function that yields batches of data from the dataset.

    Args:
        dataset (ONNXImageDataset): The dataset instance.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset at the start of each epoch.
    """
    indices = np.arange(len(dataset))

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_indices = indices[start_idx:end_idx]
        images, targets, protected = [], [], []

        for idx in batch_indices:
            image, target, protected_attr = dataset[idx]
            image = image.squeeze(0)
            images.append(image)
            targets.append(target)
            protected.append(protected_attr)

        yield np.array(images), np.array(targets), protected


def numpy_dataloader_imagepairs(
    dataset: ONNXImagePairsDataset, batch_size: int = 32, shuffle: bool = False
):
    """
    A generator function that yields batches of data from the dataset.

    Args:
        dataset (ONNXImagePairsDataset): The dataset instance.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset at the start of each epoch.
    """
    indices = np.arange(len(dataset))

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_indices = indices[start_idx:end_idx]
        images, images2, targets, protected = [], [], [], []

        for idx in batch_indices:
            image, image2, target, protected_attr = dataset[idx]
            image = image.squeeze(0)
            image2 = image2.squeeze(0)
            images.append(image)
            images2.append(image2)
            targets.append(target)
            protected.append(protected_attr)

        yield np.array(images), np.array(images2), np.array(targets), protected
