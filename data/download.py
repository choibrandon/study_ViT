# 이거 데이터를 다운로드 하지않고 그냥 음 transform=get_transforms()
# 반환할 수 있도록 해야함 생각좀 해보자. 1/17

import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from preprocess import get_transforms

#다운로드 일단 데이터셋 

def download_cifar10(root='./datasets', download=True):
    """
    Downloads the CIFAR-10 dataset to the specified root directory.

    Args:
        root (str): Directory where the dataset will be stored.
        download (bool): Whether to download the dataset if not already available.
    """

    if not os.path.exists(root):
        os.makedirs(root)
    
    print("Downloading CIFAR-10 dataset...")

    # Train set

    datasets.CIFAR10(
        root=root,
        train=True,
        transform=get_transforms(),
        download=download
    )

    # Test set

    datasets.CIFAR10(
        root=root,
        train=False,
        transform=get_transforms(),
        download=download
    )

    print(f"CIFAR-10 dataset downloaded and saved at {root}")

if __name__ == "__main__":
    dataset_root = './datasets'

    download_cifar10(root=dataset_root)