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
    """
    Function to check and return if any gpu device is available to train the model
    If GPU not present, model is trained on CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def data_loader(dataset_type="train", dataset="CIFAR10", valid_size=0.1, batch_size=64, 
                shuffle=True, random_seed=42, data_path=DATA_PATH):
    """
    Function to create a loader object which loads data in batches for training.
    Params:
        dataset_type: Type of dataset - 'train' or 'test'.
        dataset: Any dataset name provided by torchvision library
        valid_size: Size of validation set
        batch_size: number of data samples to be loaded in each batch
        shuffle: shuffle the data samples before loading
        random_seed:
        data_path: Path to save dataset
    Returns:
        train_loader, valid_loader, test_loader: Returns train & valid loader if dataset type is
                                                 'train' else return test loader
    """
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
    """
    Function to compute & return the loss and accuracy of model
    Params:
        loader: data loader object to evaluate trained model performance
        model: trained model object
        criteria: torch criterion to compute loss
    Returns:
        loss: loss of model
        accuracy: accuracy of model
    """
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

def get_parameters_count(model):
    """
    This function returns the number of parameters of model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """
    This function returns the size of the model
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb