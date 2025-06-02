import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from config.config import BATCH_SIZE, DATA_DIR

def load_cifar10_data(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def split_dataset(dataset, target_forget):
    labels = [dataset[i][1] for i in range(len(dataset))]
    if not isinstance(target_forget, list):
        target_forget = [target_forget]
    forget_indices = [i for i, label in enumerate(labels) if label in target_forget]
    retain_indices = [i for i, label in enumerate(labels) if label not in target_forget]
    forget_data = Subset(dataset, forget_indices)
    retain_data = Subset(dataset, retain_indices)
    return forget_data, retain_data
