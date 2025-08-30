from mammoth.datasets import Image
from mammoth.integration import loader
from mammoth.externals import safeexec


@loader(
    namespace="mammotheu",
    version="v0036",
    python="3.11",
    packages=("torch", "torchvision", "pandas"),
)
def data_images(
    path: str = "",
    image_root_dir: str = "",
    target: str = "",
    batch_size: int = 4,
    shuffle: bool = False,
    data_transform_path: str = "",
    transform_variable: str = "transform",
    safe_libraries="torchvision",
) -> Image:
    """
    Loads image data from a CSV file holding their sensitive and predictive attribute
    data, as well as paths relative to a root directory. Loaded images are subjected to a Python transformation.

    Args:
        path: The path to the CSV file containing information about the dataset.
        image_root_dir: The root directory where the actual image files are stored.
        target: Indicates the predictive attribute in the dataset.
        batch_size: The batch size at which images should be loaded.
        shuffle: Whether to shuffle the loaded images.
        data_transform_path: A path or implementation of a torchvision data transform. Alternatively, paste the transformation code here.
        transform_variable: The transformation target variable that should be extracted after the namesake code runs.
        safe_libraries: A comma-separated list of safe libraries that are allowed in the transformation code.
    """
    import pandas as pd

    batch_size = int(batch_size)

    premature_data = pd.read_csv(path, nrows=1)  # just read one row for verification

    data_transform = safeexec(
        data_transform_path,
        out=transform_variable,
        whitelist=[lib.strip() for lib in safe_libraries.split(",")],
    )

    dataset = Image(
        path=path,
        root_dir=image_root_dir,
        target=target,
        data_transform=data_transform,
        batch_size=batch_size,
        shuffle=shuffle,
        cols=[col for col in premature_data],
    )

    return dataset
