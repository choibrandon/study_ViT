"""
전처리 과정
"""

import torchvision.transforms as transforms

def get_transforms():
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 무작위로 자르기
        transforms.RandomHorizontalFlip(),    # 좌우반전
        transforms.ToTensor(),                # 텐서변환
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010)), # 정규화
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                              std=(0.2023, 0.1994, 0.2010))
    ])

        
    return transform_train, transform_test
