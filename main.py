import os
import sys
import time
import warnings

import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from lbcnn import LBCNN
from utils import calculate_metrics, data_loader, get_device, get_model_size, get_parameters_count
os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")

def train(train_loader, valid_loader, epochs=5, criteria="CrossEntropyLoss",
         optimiser="SGD", lr_scheduler="StepLR", params=None, model_path=MODEL_PATH,
         model_filename=None) -> object:
    
    """
    Return a learnt LBCNN model from the passed dataset.
    Params:
        train_loader (DataLoader): Torch Dataloader object holding training data
        valid_loader (DataLoader): Torch Dataloader object holding validation data
        epochs (int): Number of iterations to train the model
        criteria (str): Torch criterion to compute loss
        optimiser (str): Torch optimizer
        lr_scheduler (str): Torch learning rate scheduler to update the learning rate after every epoch based
                    on some criteria.
        params (dict): Dictionary of parameters to be passed to criterion, optimiser & lr_scheduler
        model_path (str): Path where training model will be saved
        model file name (str): filename to dump the trained model
    Returns:
        model (object): LBCNN model trained on the provided data
    """
    
    device = get_device()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if os.path.exists(os.path.join(model_path, model_filename)):
        model = pickle.load(open(os.path.join(model_path, model_filename), 'rb'))
        print("Loading saved model")
    else:
        model = LBCNN() 
        print("Instantiating model")
    
    print(f"Model Summary: \n {model}")
    model.to(device)
    
    best_accuracy = 0
    train_loss, train_accuracy =  [], []
    validation_loss, validation_accuracy = [], []
    
    criterion = getattr(torch.nn, criteria)()
    optimizer = getattr(torch.optim, optimiser)\
                        (model.parameters(), 
                         lr=params["lr"], 
                         momentum=params["momentum"], 
                         weight_decay=params["weight_decay"],
                         nesterov=True)
    scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler)(optimizer)

    start_time = time.time()
    
    print("Starting Training")
    for epoch in tqdm(range(epochs)):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        t_loss, t_acc = calculate_metrics(train_loader, model, criteria)
        v_loss, v_acc = calculate_metrics(valid_loader, model, criteria)
        
        scheduler.step(v_loss)
        
        train_loss.append(t_loss)
        train_accuracy.append(t_acc)
        validation_loss.append(v_loss)
        validation_accuracy.append(v_acc)
        
        print("\nEpoch {}/{} -> train loss : {} | train accuracy : {} | validation loss : {} | validation accuracy : {}\n"
              .format(epoch+1, epochs, t_loss, t_acc, v_loss, v_acc))

        if v_acc > best_accuracy:
            best_accuracy = validation_accuracy[epoch]
            pickle.dump(model, open(os.path.join(model_path,model_filename), 'wb'))

    end_time = time.time()
    print(f"Finished Training in {end_time - start_time} sec")

    plt.title("Loss: Training vs Validation")
    plt.plot(list(range(epochs)), train_loss, label='train')
    plt.plot(list(range(epochs)), validation_loss, label='valid')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__),'figures/loss.png'))
    
    plt.title("Accuracy: Training vs Validation")
    plt.plot(list(range(epochs)), train_accuracy, label='train')
    plt.plot(list(range(epochs)), validation_accuracy, label='valid')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__),'figures/accuracy.png'))
    
    return model

def test(test_loader, model):
    test_loss, test_accuracy = calculate_metrics(test_loader, model, "CrossEntropyLoss")
    print("Test Loss : {} | Test Accuracy : {}".format(test_loss, test_accuracy))

if __name__ == "__main__":
    train_loader, valid_loader = data_loader(dataset_type="train",
                                             dataset="CIFAR100",
                                             valid_size=0.1, 
                                             batch_size=128)
    test_loader = data_loader(dataset_type="test",dataset="CIFAR100",batch_size=128)

    params = {
        "lr": 1e-2, 
        "lr_scheduler_step": 5, 
        "momentum":0.9, 
        "weight_decay":1e-4
    }

    model = train(train_loader, valid_loader, epochs=80, lr_scheduler="ReduceLROnPlateau", 
                  params=params, model_filename="LBCNN_cifar100.pkl")
    test(test_loader, model)
    print("Number of parameters: {}".format(get_parameters_count(model)))
    print("Model Size: {:.3f}MB".format(get_model_size(model)))