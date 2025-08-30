from mammoth.datasets import CSV
from mammoth.integration import loader
import os


def _download(url, path=None):
    import urllib.request

    # Get the file name from the URL
    if path is None:
        file_name = os.path.basename(url)
    else:
        file_name = path

    try:
        with urllib.request.urlopen(url) as response:
            total_size = response.getheader("Content-Length")
            total_size = int(total_size) if total_size else None

            with open(file_name, "wb") as out_file:
                chunk_size = 1024
                downloaded = 0

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

                    # Print progress if total size is known
                    if total_size:
                        done = int(50 * downloaded / total_size)
                        print(
                            f'\rDownloading {url} [{"=" * done}{" " * (50 - done)}] {downloaded / 1024:.2f} KB',
                            end="",
                        )

        print(f"Downloaded {url}" + " " * 50)
    except Exception as e:
        print(f"Error downloading file: {e}")


def _extract_nested_zip(file, folder):
    import zipfile

    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(file, "r") as zfile:
        zfile.extractall(path=folder)
    os.remove(file)
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".zip"):
                _extract_nested_zip(
                    os.path.join(root, filename), os.path.join(root, filename[:-4])
                )


def read_csv(url, **kwargs):
    import pandas as pd
    import os
    import csv

    url = url.replace("\\", "/")
    if ".zip/" in url:
        url, path = url.split(".zip/", 1)
        extract_to = "data/"
        if "/" not in path:
            extract_to += url.split("/")[-1]
            path = os.path.join(url.split("/")[-1], path)
        path = os.path.join("data", path)
        url += ".zip"
        temp = "data/" + url.split("/")[-1]
        if not os.path.exists(path):
            os.makedirs(os.path.join(*path.split("/")[:-1]), exist_ok=True)
            _download(url, temp)
            _extract_nested_zip(temp, extract_to)
    elif os.path.exists(url):
        path = url
    else:
        shortened = "/".join(url.split("/")[-4:])
        path = "data/" + shortened
        if not os.path.exists(path):
            os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
            _download(url, path)
    if "delimiter" in kwargs:
        return pd.read_csv(path, **kwargs)

    with open(path, "r") as file:
        sample = file.read(1024)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        delimiter = str(delimiter)
    return pd.read_csv(path, delimiter=delimiter, **kwargs)


@loader(
    namespace="mammotheu",
    version="v0036",
    python="3.11",
    packages=("pandas",),
)
def data_auto_csv(path: str = "", max_discrete: int = 10) -> CSV:
    """Loads a CSV file that contains numeric, categorical, and predictive data columns.
    This automatically detects the characteristics of the dataset being loaded,
    namely the delimiter that separates the columns, and whether each column contains
    numeric or categorical data. A <a href="https://pandas.pydata.org/">pandas</a>
    CSV reader is employed internally.
    The last categorical column is used as the dataset label. To load the file using
    different options (e.g., a subset of columns, a different label column) use the
    custom csv loader instead.

    Args:
        path: The local file path or a web URL of the file.
        max_discrete: If a numeric column has a number of discrete entries than is less than this number (e.g., if it contains binary numeric values) then it is considered to hold categorical instead of numeric data. Minimum accepted value is 2.
    """
    if not path.endswith(".csv"):
        raise Exception("A file or url with .csv extension is needed.")
    max_discrete = int(max_discrete)
    if max_discrete < 2:
        raise Exception(
            "The number of numeric levels (the value of max discrete) should be at least 2"
        )
    raw_data = read_csv(
        path,
        on_bad_lines="skip",
    )
    import pandas as pd

    numeric = [
        col for col in raw_data if pd.api.types.is_any_real_numeric_dtype(raw_data[col])
    ]
    numeric = [col for col in numeric if len(set(raw_data[col])) > max_discrete]
    numeric_set = set(numeric)
    categorical = [col for col in raw_data if col not in numeric_set]
    if len(categorical) < 1:
        raise Exception("At least two categorical columns are required.")
    label = categorical[-1]
    categorical = categorical[:-1]

    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=label,
    )
    return csv_dataset
