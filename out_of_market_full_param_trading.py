import sys
import json
import copy
import random
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm

import torch
import torchvision

from models.mlp import MLP
from models.vgg import VGG
from models.resnet import ResNet

from model_alignment.process_data import MyCustomDataset
from model_alignment.training import train, test, cf_matrix_computation

np.random.seed(100)

## Exp config setup ##
f = open(sys.argv[1])
config = json.load(f)
f.close()
print(config)

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

## Only for mlp ##
hidden_size = config["hidden_size"]

if model_type == "ResNet20":
    depth = 22
elif model_type == "ResNet50":
    depth = 52

if task != "cifar10" and model_type == "MLP":
    n_feature = 28 * 28
elif task == "cifar10" and model_type == "MLP":
    n_feature = 32 * 32 * 3

if model_type in ["ResNet20", "ResNet50", "VGG16"]:
    softmax = True
else:
    softmax = False

## Number of data point for each class ##
if task == "mnist":
    full_class_freq = {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
elif task == "fashion":
    full_class_freq = {0: 6000, 1: 6000, 2: 6000, 3: 6000, 4: 6000, 5: 6000, 6: 6000, 7: 6000, 8: 6000, 9: 6000}
elif task == "kmnist":
    full_class_freq = {0: 6000, 1: 6000, 2: 6000, 3: 6000, 4: 6000, 5: 6000, 6: 6000, 7: 6000, 8: 6000, 9: 6000}
elif task == "cifar10":
    full_class_freq = {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 5000, 5: 5000, 6: 5000, 7: 5000, 8: 5000, 9: 5000}
    
agent_a_class_sampled_freq = copy.deepcopy(full_class_freq)
agent_b_class_sampled_freq = copy.deepcopy(full_class_freq)
bank_class_sampled_freq = copy.deepcopy(full_class_freq)

for class_, freq_ in full_class_freq.items():
    if class_ < int(num_of_class / 2):
        agent_a_class_sampled_freq[class_] = int(freq_ * datapoint_deduction)
    else:
        agent_b_class_sampled_freq[class_] = int(freq_ * datapoint_deduction)

print(agent_a_class_sampled_freq)
print(agent_b_class_sampled_freq)
print(bank_class_sampled_freq)

## Dataset and dataloader ##
agent_a_dataset = MyCustomDataset(task_name=task, train=True, sample=True, class_freq=agent_a_class_sampled_freq)
agent_b_dataset = MyCustomDataset(task_name=task, train=True, sample=True, class_freq=agent_b_class_sampled_freq)
bank_train_dataset = MyCustomDataset(task_name=task, train=True)
bank_test_dataset = MyCustomDataset(task_name=task, train=False)

agent_a_dataloader = torch.utils.data.DataLoader(agent_a_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
agent_b_dataloader = torch.utils.data.DataLoader(agent_b_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
bank_train_dataloader = torch.utils.data.DataLoader(bank_train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
bank_test_dataloader = torch.utils.data.DataLoader(bank_test_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

## Initialization ##
if model_type == "MLP":
    agent_a_model = MLP(seed=seeds[0], n_feature=n_feature, n_hidden=hidden_size, num_classes=num_of_class).to(device)
    agent_b_model = MLP(seed=seeds[1], n_feature=n_feature, n_hidden=hidden_size, num_classes=num_of_class).to(device)
    bank_model = MLP(seed=seeds[2], n_feature=n_feature, n_hidden=hidden_size, num_classes=num_of_class).to(device)
elif model_type in ["ResNet20", "ResNet50"]:
    agent_a_model = ResNet(seed=seeds[0], depth=depth, widen_factor=widen_factor, num_classes=num_of_class).to(device)
    agent_b_model = ResNet(seed=seeds[1], depth=depth, widen_factor=widen_factor, num_classes=num_of_class).to(device)
    bank_model = ResNet(seed=seeds[2], depth=depth, widen_factor=widen_factor, num_classes=num_of_class).to(device)
elif model_type == "VGG16":
    agent_a_model = VGG(seed=seeds[0], vgg_name="VGG16", num_classes=num_of_class).to(device) 
    agent_b_model = VGG(seed=seeds[1], vgg_name="VGG16", num_classes=num_of_class).to(device) 
    bank_model = VGG(seed=seeds[2], vgg_name="VGG16", num_classes=num_of_class).to(device) 
    
if optimizer == "SGD":
    agent_a_optimizer = torch.optim.SGD(agent_a_model.parameters(), lr=learning_rate)
    agent_b_optimizer = torch.optim.SGD(agent_b_model.parameters(), lr=learning_rate)
    bank_optimizer = torch.optim.SGD(bank_model.parameters(), lr=learning_rate)
elif optimizer == "Adam":
    agent_a_optimizer = torch.optim.Adam(agent_a_model.parameters(), lr=learning_rate)
    agent_b_optimizer = torch.optim.Adam(agent_b_model.parameters(), lr=learning_rate)
    bank_optimizer = torch.optim.Adam(bank_model.parameters(), lr=learning_rate)

## Agents' evaluation collection ##
agent_a_evaluation_collection = []
agent_b_evaluation_collection = []
bank_evaluation_collection = []

## Training, Trading, and Testing ##
for epoch in tqdm(range(total_epochs), total=total_epochs):
    
    print("--------------------------------------------")
    
    agent_a_train_loss, agent_a_train_acc = train(agent_a_model, agent_a_dataloader, agent_a_optimizer, epoch, agent="A", softmax=softmax, log_interval=log_interval)
    agent_a_test_loss, agent_a_test_acc = test(agent_a_model, bank_test_dataloader, epoch, agent="A", softmax=softmax)
    
    agent_b_train_loss, agent_b_train_acc = train(agent_b_model, agent_b_dataloader, agent_b_optimizer, epoch, agent="B", softmax=softmax, log_interval=log_interval)
    agent_b_test_loss, agent_b_test_acc = test(agent_b_model, bank_test_dataloader, epoch, agent="B", softmax=softmax)
    
    bank_train_loss, bank_train_acc = train(bank_model, bank_train_dataloader, bank_optimizer, epoch, agent="Bank", softmax=softmax, log_interval=log_interval)
    bank_test_loss, bank_test_acc = test(bank_model, bank_test_dataloader, epoch, agent="Bank", softmax=softmax)
    
    print("--------------------------------------------")
    
    agent_a_evaluation_collection.append([agent_a_train_loss, agent_a_train_acc, agent_a_test_loss, agent_a_test_acc])
    agent_b_evaluation_collection.append([agent_b_train_loss, agent_b_train_acc, agent_b_test_loss, agent_b_test_acc])
    bank_evaluation_collection.append([bank_train_loss, bank_train_acc, bank_test_loss, bank_test_acc])
    
agent_a_evaluation_collection = np.array(agent_a_evaluation_collection)
agent_b_evaluation_collection = np.array(agent_b_evaluation_collection)
bank_evaluation_collection = np.array(bank_evaluation_collection)

agent_a_trade_cf_matrix = cf_matrix_computation(agent_a_model, bank_test_dataloader)
agent_b_trade_cf_matrix = cf_matrix_computation(agent_b_model, bank_test_dataloader)
bank_cf_matrix = cf_matrix_computation(bank_model, bank_test_dataloader)

## store all the information ##
agent_a_df = pd.DataFrame(agent_a_evaluation_collection, columns=["Training_Loss", "Training_Acc", "Testing_Loss", "Testing_Acc"])
agent_b_df = pd.DataFrame(agent_b_evaluation_collection, columns=["Training_Loss", "Training_Acc", "Testing_Loss", "Testing_Acc"])
bank_df = pd.DataFrame(bank_evaluation_collection, columns=["Training_Loss", "Training_Acc", "Testing_Loss", "Testing_Acc"])

agent_a_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_a_perf_" + variable + ".csv"
agent_b_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_b_perf_" + variable + ".csv"
bank_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/bank_perf_" + variable + ".csv"

agent_a_cf_matrix_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_a_cf_matrix_" + variable + ".npy"
agent_b_cf_matrix_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_b_cf_matrix_" + variable + ".npy"
bank_cf_matrix_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/bank_cf_matrix_" + variable + ".npy"

agent_a_df.to_csv(agent_a_path, index=False)
agent_b_df.to_csv(agent_b_path, index=False)
bank_df.to_csv(bank_path, index=False)

np.save(agent_a_cf_matrix_path, agent_a_trade_cf_matrix)
np.save(agent_b_cf_matrix_path, agent_b_trade_cf_matrix)
np.save(bank_cf_matrix_path, bank_cf_matrix)