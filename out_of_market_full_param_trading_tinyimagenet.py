import os
import sys
import json
import copy
import random
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import datasets
from torchvision import transforms as T

from models.resnet import ResNet

def train(model, dataloader, optimizer, epoch, agent, softmax=False, log_interval=None):
    
    model.train()
    
    train_loss = 0
    correct = 0
    count = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to("cuda:0"), target.to("cuda:0")
        optimizer.zero_grad()
        output = model(data)
        if softmax:
            output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += loss.item()
        count += 1
        
        loss.backward()
        optimizer.step()
        
        if log_interval != None:
            if batch_idx % log_interval == 0:
                acc = 100. * correct / 55_000
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), 55_000,
                    100. * batch_idx / len(dataloader), loss.item()))
            
    train_loss /= count
    acc = 100. * correct / 55_000
    print('Agent: {}, Train Epoch: {}, Avg. Loss: {:.4f}, Avg. Accuracy: {:.2f}%'.format(agent, epoch, train_loss, acc))
    
    return train_loss, acc

def test(model, dataloader, epoch, agent, softmax=False):
    
    model.eval()
    
    test_loss = 0
    correct = 0
    count = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to("cuda:0"), target.to("cuda:0")
            output = model(data)
            if softmax:
                output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += 1
            
    test_loss /= count
    acc = 100. * correct / len(dataloader.dataset)
    if agent != None:
        print('Agent: {}, Test  Epoch: {}, Avg. Loss: {:.4f}, avg. Accuracy: {:.2f}%'.format(agent, epoch, test_loss, acc))
    
    return test_loss, acc

def generate_dataloader(data, name, transform, sample=False, class_freq=None):
    
    dataset = datasets.ImageFolder(data, transform=transform)
    kwargs = {"pin_memory": True, "num_workers": 1}
    
    if sample == True:
        full_targets = torch.tensor(dataset.targets)
        total_sampled_index = []
        for class_, freq_ in class_freq.items():
            specific_class_total_index = np.where(full_targets == class_)[0]
            sampled_data_index = np.random.choice(specific_class_total_index, size=freq_, replace=False)
            total_sampled_index.extend(sampled_data_index)
        sampled_index = torch.tensor(np.array(total_sampled_index).reshape(len(total_sampled_index), 1))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(sampled_index)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, **kwargs)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(name=="train"), **kwargs)

    return dataloader

np.random.seed(100)

## Exp config setup ##
f = open(sys.argv[1])
config = json.load(f)
f.close()
print(config)

# Define main data directory
DATA_DIR = './tiny-imagenet-200'

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')
val_img_dir = os.path.join(VALID_DIR, 'images')

seeds = config["seeds"]
task = config["task"]
model_type = config["model_type"]
total_epochs = config["total_epochs"]
learning_rate = config["learning_rate"]
optimizer = config["optimizer"]
num_of_class = config["num_of_class"]
datapoint_deduction = config["datapoint_deduction"]
batch_size = config["batch_size"]
log_interval = config["log_interval"]
device = config["device"]
exp_category = config["exp_category"]
if log_interval == -1:
    log_interval = None
variable = str(int(datapoint_deduction * 100)) + "_" + optimizer

## Only for resnet ##
widen_factor = config["widen_factor"]

depth = 22
softmax = True

## Number of data point for each class ##
full_class_freq = dict()
for i in range(num_of_class):
    full_class_freq[i] = 500
    
agent_a_class_sampled_freq = copy.deepcopy(full_class_freq)
agent_b_class_sampled_freq = copy.deepcopy(full_class_freq)
for class_, freq_ in full_class_freq.items():
    if class_ < int(num_of_class / 2):
        agent_a_class_sampled_freq[class_] = int(freq_ * datapoint_deduction)
    else:
        agent_b_class_sampled_freq[class_] = int(freq_ * datapoint_deduction)

print(agent_a_class_sampled_freq)
print(agent_b_class_sampled_freq)

## transformation ##
preprocess_transform_pretrain = T.Compose([
    T.Resize(32),
    T.RandomHorizontalFlip(),
    T.ToTensor(),  # Converting cropped images to tensors
    T.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])

## Dataset and dataloader ##
agent_a_dataloader = generate_dataloader(TRAIN_DIR, "train", \
                                         transform=preprocess_transform_pretrain, \
                                         sample=True, class_freq=agent_a_class_sampled_freq)

agent_b_dataloader = generate_dataloader(TRAIN_DIR, "train", \
                                         transform=preprocess_transform_pretrain, \
                                         sample=True, class_freq=agent_b_class_sampled_freq)

bank_test_dataloader = generate_dataloader(val_img_dir, "val", \
                                           transform=preprocess_transform_pretrain)

## Initialization ##
agent_a_model = ResNet(seed=seeds[0], depth=depth, widen_factor=widen_factor, num_classes=num_of_class).to(device)
agent_b_model = ResNet(seed=seeds[1], depth=depth, widen_factor=widen_factor, num_classes=num_of_class).to(device)
    
if optimizer == "SGD":
    agent_a_optimizer = torch.optim.SGD(agent_a_model.parameters(), lr=learning_rate)
    agent_b_optimizer = torch.optim.SGD(agent_b_model.parameters(), lr=learning_rate)
elif optimizer == "Adam":
    agent_a_optimizer = torch.optim.Adam(agent_a_model.parameters(), lr=learning_rate)
    agent_b_optimizer = torch.optim.Adam(agent_b_model.parameters(), lr=learning_rate)

## Agents' evaluation collection ##
agent_a_evaluation_collection = []
agent_b_evaluation_collection = []

## Training, Trading, and Testing ##
for epoch in tqdm(range(total_epochs), total=total_epochs):
    
    print("--------------------------------------------")
    
    agent_a_train_loss, agent_a_train_acc = train(agent_a_model, agent_a_dataloader, agent_a_optimizer, epoch, agent="A", softmax=softmax, log_interval=log_interval)
    agent_a_test_loss, agent_a_test_acc = test(agent_a_model, bank_test_dataloader, epoch, agent="A", softmax=softmax)
    
    agent_b_train_loss, agent_b_train_acc = train(agent_b_model, agent_b_dataloader, agent_b_optimizer, epoch, agent="B", softmax=softmax, log_interval=log_interval)
    agent_b_test_loss, agent_b_test_acc = test(agent_b_model, bank_test_dataloader, epoch, agent="B", softmax=softmax)
    
    print("--------------------------------------------")
    
    agent_a_evaluation_collection.append([agent_a_train_loss, agent_a_train_acc, agent_a_test_loss, agent_a_test_acc])
    agent_b_evaluation_collection.append([agent_b_train_loss, agent_b_train_acc, agent_b_test_loss, agent_b_test_acc])
    
agent_a_evaluation_collection = np.array(agent_a_evaluation_collection)
agent_b_evaluation_collection = np.array(agent_b_evaluation_collection)

## store all the information ##
agent_a_df = pd.DataFrame(agent_a_evaluation_collection, columns=["Training_Loss", "Training_Acc", "Testing_Loss", "Testing_Acc"])
agent_b_df = pd.DataFrame(agent_b_evaluation_collection, columns=["Training_Loss", "Training_Acc", "Testing_Loss", "Testing_Acc"])

agent_a_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_a_perf_" + variable + ".csv"
agent_b_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_b_perf_" + variable + ".csv"

agent_a_df.to_csv(agent_a_path, index=False)
agent_b_df.to_csv(agent_b_path, index=False)
