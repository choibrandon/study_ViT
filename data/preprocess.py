import torchvision.transforms as transforms

def get_transforms(dataset_name):
    """
    Returns train and test transforms for the specified dataset

    Args:
        dataset_name (str): Name of the dataset (e.g. 'cifar10', "mnist" 이건 너가 만들어보셈)

    Returns:
        (transform_train, transform_test): Tuple of train and test transforms.

    """
    
    if dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), #무작위로 자르기
            transforms.RandomHorizontalFlip(), #좌우반전
            transforms.ToTensor(), #텐서변환
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)), # 정규화
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
    
    elif dataset_name.lower() == 'mnist':
        raise ValueError(f"너가 만들어 보거라")
    
    else:
        raise ValueError(f"not supported dataset: {dataset_name}")
        
    return transform_train, transform_test