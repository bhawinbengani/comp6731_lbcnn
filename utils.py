import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from tqdm import tqdm

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')

def data_loader(dataset_type="train", dataset="CIFAR10", valid_size=0.1, batch_size=64, 
                shuffle=True, random_seed=42, data_path=DATA_PATH):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    mean, std=(0.5074,0.4867,0.4411), (0.2011,0.1987,0.2025)
    transform_list = [transforms.RandomHorizontalFlip(),
                      transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                      transforms.ToTensor(), 
                      transforms.Normalize(mean=mean, std=std)]
    
    if dataset_type == "train":
        train_set = getattr(torchvision.datasets, dataset)(
                            root=data_path, train=True, download=True,
                            transform=transforms.Compose(transform_list))
        
        valid_set = getattr(torchvision.datasets, dataset)(
                            root=data_path, train=True, download=True,
                            transform=transforms.Compose(transform_list[-2:]))
        
        num_train = len(train_set)
        indicies = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indicies)

        train_idx, valid_idx = indicies[split:], indicies[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=8)
        
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size,
                                  sampler=valid_sampler, num_workers=8)
        return train_loader, valid_loader
    
    if dataset_type == "test":
        dataset = getattr(torchvision.datasets, dataset)(root=data_path, train=True, 
                          download=True, transform=transforms.Compose(transform_list[-2:]))
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 num_workers=8)
        return test_loader


def calculate_metrics(loader, model, criteria):
    device = get_device()
    model = model.to(device)
    steps, loss = 0, 0
    correct, total = 0, 0
    criterion = getattr(torch.nn, criteria)()
    for inputs, labels in loader:
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
            loss += criterion(outputs, labels).cpu().numpy()
            steps += 1
    return loss/steps, correct/total