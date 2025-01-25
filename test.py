import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from models.vit import ViT
from utils.logger import Logger

def test(model, test_loader, criterion, logger, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    logger.log(f"Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    test_config = config['test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=test_config['batch_size'], shuffle=False)

    model = ViT().to(device)
    model.load_state_dict(torch.load(f"{test_config['log_dir']}/vit_model.pth"))

    criterion = nn.CrossEntropyLoss()
    logger = Logger(log_dir=test_config['log_dir'])

    test(model, test_loader, criterion, logger, device)

if __name__ == '__main__':
    main()