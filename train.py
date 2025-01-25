import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from models.vit import ViT
from utils.logger import Logger

def train(epoch, model, train_loader, criterion, optimizer, logger, device):
    model.train()
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            logger.log(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

    return total_loss / len(train_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_config = config['train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True)

    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=5e-5)

    logger = Logger(log_dir=train_config['log_dir'])

    for epoch in range(train_config['epoch']):
        avg_loss = train(epoch, model, train_loader, criterion, optimizer, logger, device)
        logger.log(f"Epoch {epoch} complete | Average Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()