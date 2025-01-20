import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from preprocess import get_transforms

#다운로드 일단 데이터셋 

def download_cifar10(root='./datasets', train_transform=None, test_transform=None, download=True):
    if not os.path.exists(root):
        os.mkdir(root)
    
    print("Downloading cifar-10 dataset...")

    train_set = datasets.CIFAR10(
        root=root,
        train=True,
        transform=train_transform,
        download=download
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        transform=test_transform,
        download=download
    )

    print(f"Done! at {root}")

    return train_set, test_set

if __name__ == "__main__":
    download_cifar10()
