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
from model_alignment.weight_matching import mlp_permutation_spec, resnet20_permutation_spec, resnet50_permutation_spec, vgg16_permutation_spec, weight_matching, apply_permutation

## Trading modules ##
def try_before_purchase(buyer, seller, softmax=False):
    
    purchased_weights = np.arange(0.05, 1.00, 0.1)
    min_testing_loss = None
    
    testing_loss_list = []
    testing_acc_list = []
    for weight in purchased_weights:
        
        buyer_backup = copy.deepcopy(buyer).to("cuda:0")
        seller_backup = copy.deepcopy(seller).to("cuda:0")
    
        for layer_name in buyer_backup.state_dict().keys():
            hybrid_params = (1 - weight) * buyer_backup.state_dict()[layer_name] + weight * seller_backup.state_dict()[layer_name]
            buyer_backup.state_dict()[layer_name].copy_(hybrid_params)
        
        testing_loss, testing_acc = test(buyer_backup, dataloader=bank_test_dataloader, epoch=None, agent=None, softmax=softmax)
        
        testing_loss_list.append(testing_loss)
        testing_acc_list.append(testing_acc)
        
        if min_testing_loss == None or min_testing_loss > testing_loss:
            min_testing_loss = testing_loss
            purchased_weight_star = weight
            theta_bar = copy.deepcopy(buyer_backup)

    return purchased_weight_star, theta_bar, min_testing_loss, testing_loss_list, testing_acc_list

def weight_alignment(buyer, seller, model_type):
    
    buyer_backup = copy.deepcopy(buyer).to("cpu")
    seller_backup = copy.deepcopy(seller).to("cpu")
    
    if model_type == "MLP":
        permutation_spec = mlp_permutation_spec(4)
    elif model_type == "ResNet20":
        permutation_spec = resnet20_permutation_spec()
    elif model_type == "ResNet50":
        permutation_spec = resnet50_permutation_spec()
    elif model_type == "VGG16":
        permutation_spec = vgg16_permutation_spec()
        
    permutation = weight_matching(permutation_spec, buyer_backup.state_dict(), seller_backup.state_dict())
    aligned_params = apply_permutation(permutation_spec, permutation, seller_backup.state_dict())
    
    seller_backup.load_state_dict(aligned_params)
    
    return seller_backup

def gain_from_trade(before_trade_loss, after_trade_loss):
    
    improvement = (before_trade_loss - after_trade_loss) / before_trade_loss
    
    return improvement

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
trading_frequency = config["trading_frequency"]
optimizer = config["optimizer"]
num_of_class = config["num_of_class"]
datapoint_deduction = config["datapoint_deduction"]
trading_period = config["trading_period"]
model_alignment = config["model_alignment"]
batch_size = config["batch_size"]
purchase_type = config["purchase_type"]
log_interval = config["log_interval"]
device = config["device"]
exp_category = config["exp_category"]
if log_interval == -1:
    log_interval = None

if exp_category == "freq":
    variable = str(trading_frequency)
elif exp_category == "alignment_optimizer":
    variable = str(model_alignment) + "_" + optimizer
elif exp_category == "period":
    variable = str(trading_period)
elif exp_category == "deduction":
    variable = str(int(datapoint_deduction * 100))

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
for class_, freq_ in full_class_freq.items():
    if class_ < int(num_of_class / 2):
        agent_a_class_sampled_freq[class_] = int(freq_ * datapoint_deduction)
    else:
        agent_b_class_sampled_freq[class_] = int(freq_ * datapoint_deduction)

print(agent_a_class_sampled_freq)
print(agent_b_class_sampled_freq)

## Dataset and dataloader ##
agent_a_dataset = MyCustomDataset(task_name=task, train=True, sample=True, class_freq=agent_a_class_sampled_freq)
agent_b_dataset = MyCustomDataset(task_name=task, train=True, sample=True, class_freq=agent_b_class_sampled_freq)
bank_test_dataset = MyCustomDataset(task_name=task, train=False)

agent_a_dataloader = torch.utils.data.DataLoader(agent_a_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
agent_b_dataloader = torch.utils.data.DataLoader(agent_b_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
bank_test_dataloader = torch.utils.data.DataLoader(bank_test_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

## Initialization ##
if model_type == "MLP":
    agent_a_market_model = MLP(seed=seeds[0], n_feature=n_feature, n_hidden=hidden_size, num_classes=num_of_class).to(device)
    agent_b_market_model = MLP(seed=seeds[1], n_feature=n_feature, n_hidden=hidden_size, num_classes=num_of_class).to(device)
elif model_type in ["ResNet20", "ResNet50"]:
    agent_a_market_model = ResNet(seed=seeds[0], depth=depth, widen_factor=widen_factor, num_classes=num_of_class).to(device)
    agent_b_market_model = ResNet(seed=seeds[1], depth=depth, widen_factor=widen_factor, num_classes=num_of_class).to(device)
elif model_type == "VGG16":
    agent_a_market_model = VGG(seed=seeds[0], vgg_name="VGG16", num_classes=num_of_class).to(device) 
    agent_b_market_model = VGG(seed=seeds[1], vgg_name="VGG16", num_classes=num_of_class).to(device) 

if optimizer == "SGD":
    agent_a_trade_optimizer = torch.optim.SGD(agent_a_market_model.parameters(), lr=learning_rate)
    agent_b_trade_optimizer = torch.optim.SGD(agent_b_market_model.parameters(), lr=learning_rate)
elif optimizer == "Adam":
    agent_a_trade_optimizer = torch.optim.Adam(agent_a_market_model.parameters(), lr=learning_rate)
    agent_b_trade_optimizer = torch.optim.Adam(agent_b_market_model.parameters(), lr=learning_rate)

## Agents' evaluation collection ##
agent_a_trade_evaluation_collection = []
agent_b_trade_evaluation_collection = []

trading_log = np.array([np.zeros(total_epochs), np.zeros(total_epochs)])
Delta_collection = np.array([np.zeros(total_epochs), np.zeros(total_epochs)])
agent_a_prebuy_log = []
agent_b_prebuy_log = []

## Training, Trading, and Testing ##
for epoch in tqdm(range(total_epochs), total=total_epochs):
    
    print("--------------------------------------------")
    ## Training model ##
    agent_a_train_loss, agent_a_train_acc = train(agent_a_market_model, agent_a_dataloader, agent_a_trade_optimizer, epoch, agent="A", softmax=softmax, log_interval=log_interval)
    agent_b_train_loss, agent_b_train_acc = train(agent_b_market_model, agent_b_dataloader, agent_b_trade_optimizer, epoch, agent="B", softmax=softmax, log_interval=log_interval)
    
    ## Trading Time ##
    if epoch >= trading_period and epoch % trading_frequency == 0:
        
        agent_a_before_trade_testing_loss, agent_a_before_trade_testing_acc = test(agent_a_market_model, bank_test_dataloader, epoch, agent="A", softmax=softmax)
        agent_b_before_trade_testing_loss, agent_b_before_trade_testing_acc = test(agent_b_market_model, bank_test_dataloader, epoch, agent="B", softmax=softmax)
    
        print("Epoch: ", epoch, "Try-before-purchase")
        
        ## Agent A pre-buys params from Agent B ##
        if model_alignment == 0:
            ## just linear interpolation ##
            print("just linear interpolation")
            agent_a_purchased_weight, agent_a_market_model_bar, agent_a_after_trade_testing_loss, agent_a_prebuy_losses, agent_a_prebuy_accs = try_before_purchase(
                buyer=agent_a_market_model, seller=agent_b_market_model, softmax=softmax
            )
        else:
            ## model alignment + linear interpolation ##
            print("model alignment + linear interpolation")
            agent_b_market_aligned_model = weight_alignment(buyer=agent_a_market_model, seller=agent_b_market_model, model_type=model_type)
            agent_a_purchased_weight, agent_a_market_model_bar, agent_a_after_trade_testing_loss, agent_a_prebuy_losses, agent_a_prebuy_accs = try_before_purchase(
                buyer=agent_a_market_model, seller=agent_b_market_aligned_model, softmax=softmax
            )
        
        ## Agent A gain-from-trade ##
        agent_a_Delta = gain_from_trade(agent_a_before_trade_testing_loss, agent_a_after_trade_testing_loss)
        
        ## Agent B pre-buys params from Agent A ##
        if model_alignment == 0:
            ## just linear interpolation ##
            print("just linear interpolation")
            agent_b_purchased_weight, agent_b_market_model_bar, agent_b_after_trade_testing_loss, agent_b_prebuy_losses, agent_b_prebuy_accs = try_before_purchase(
                buyer=agent_b_market_model, seller=agent_a_market_model, softmax=softmax
            )
        else:
            ## model alignment + linear interpolation ##
            print("model alignment + linear interpolation")
            agent_a_market_aligned_model = weight_alignment(buyer=agent_b_market_model, seller=agent_a_market_model, model_type=model_type)
            agent_b_purchased_weight, agent_b_market_model_bar, agent_b_after_trade_testing_loss, agent_b_prebuy_losses, agent_b_prebuy_accs = try_before_purchase(
                buyer=agent_b_market_model, seller=agent_a_market_aligned_model, softmax=softmax
            )
        
        ## Agent B gain-from-trade ##
        agent_b_Delta = gain_from_trade(agent_b_before_trade_testing_loss, agent_b_after_trade_testing_loss)
        
        ## record trading log ##
        agent_a_prebuy_log.append([agent_a_prebuy_losses, agent_a_prebuy_accs])
        agent_b_prebuy_log.append([agent_b_prebuy_losses, agent_b_prebuy_accs])
        Delta_collection[0][epoch], Delta_collection[1][epoch] = agent_a_Delta, agent_b_Delta
        
        ## Buy it or not, Agent A's side ##
        if agent_a_Delta > 0:
            print("Agent A buys params from Agent B", ", purchased weight, ", round(agent_a_purchased_weight, 2), \
                  "after trade", round(agent_a_after_trade_testing_loss, 4), ", before trade", round(agent_a_before_trade_testing_loss, 4))
            agent_a_market_model = copy.deepcopy(agent_a_market_model_bar)
            if optimizer == "SGD":
                agent_a_trade_optimizer = torch.optim.SGD(agent_a_market_model.parameters(), lr=learning_rate)
            elif optimizer == "Adam":
                agent_a_trade_optimizer = torch.optim.Adam(agent_a_market_model.parameters(), lr=learning_rate)
            trading_log[0][epoch] = 1
        else:
            print("Agent A doesn't buy params from Agent B \t")
        
        ## Buy it or not, Agent B's side ##
        if agent_b_Delta > 0:
            print("Agent B buys params from Agent A", ", purchased weight, ", round(agent_b_purchased_weight, 2), \
                  "after trade", round(agent_b_after_trade_testing_loss, 4), ", before trade", round(agent_b_before_trade_testing_loss, 4))
            agent_b_market_model = copy.deepcopy(agent_b_market_model_bar)
            if optimizer == "SGD":
                agent_b_trade_optimizer = torch.optim.SGD(agent_b_market_model.parameters(), lr=learning_rate)
            elif optimizer == "Adam":
                agent_b_trade_optimizer = torch.optim.Adam(agent_b_market_model.parameters(), lr=learning_rate)
            trading_log[1][epoch] = 1
        else:
            print("Agent B doesn't buy params from Agent A \t")
        
    ## Testing model ##
    agent_a_test_loss, agent_a_test_acc = test(agent_a_market_model, bank_test_dataloader, epoch, agent="A", softmax=softmax)
    agent_b_test_loss, agent_b_test_acc = test(agent_b_market_model, bank_test_dataloader, epoch, agent="B", softmax=softmax)
    
    print("--------------------------------------------")
    
    ## Record loss ##
    agent_a_trade_evaluation_collection.append([agent_a_train_loss, agent_a_train_acc, agent_a_test_loss, agent_a_test_acc])
    agent_b_trade_evaluation_collection.append([agent_b_train_loss, agent_b_train_acc, agent_b_test_loss, agent_b_test_acc])
    
agent_a_trade_evaluation_collection = np.array(agent_a_trade_evaluation_collection)
agent_b_trade_evaluation_collection = np.array(agent_b_trade_evaluation_collection)
agent_a_prebuy_log = np.array(agent_a_prebuy_log)
agent_b_prebuy_log = np.array(agent_b_prebuy_log)

agent_a_trade_cf_matrix = cf_matrix_computation(agent_a_market_model, bank_test_dataloader)
agent_b_trade_cf_matrix = cf_matrix_computation(agent_b_market_model, bank_test_dataloader)

## store all the information ##
agent_a_trade_df = pd.DataFrame(agent_a_trade_evaluation_collection, columns=["Training_Loss", "Training_Acc", "Testing_Loss", "Testing_Acc"])
agent_b_trade_df = pd.DataFrame(agent_b_trade_evaluation_collection, columns=["Training_Loss", "Training_Acc", "Testing_Loss", "Testing_Acc"])
trading_log_df = pd.DataFrame(trading_log.T, columns=["Agent_A_Buys", "Agent_B_Buys"])
gain_from_trade_df = pd.DataFrame(Delta_collection.T, columns=["Agent_A_Delta", "Agent_B_Delta"])

agent_a_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_a_perf_" + variable + ".csv"
agent_b_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_b_perf_" + variable + ".csv"
trading_log_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/trading_log_" + variable + ".csv"
gain_from_trade_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/gain_from_trade_" + variable + ".csv"
agent_a_cf_matrix_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_a_cf_matrix_" + variable + ".npy"
agent_b_cf_matrix_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_b_cf_matrix_" + variable + ".npy"
agent_a_prebuy_log_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_a_prebuy_" + variable + ".npy"
agent_b_prebuy_log_path = "./results2/" + task + "_" + model_type + "/" + exp_category + "/agent_b_prebuy_" + variable + ".npy"

agent_a_trade_df.to_csv(agent_a_path, index=False)
agent_b_trade_df.to_csv(agent_b_path, index=False)
trading_log_df.to_csv(trading_log_path, index=False)
gain_from_trade_df.to_csv(gain_from_trade_path, index=False)
np.save(agent_a_cf_matrix_path, agent_a_trade_cf_matrix)
np.save(agent_b_cf_matrix_path, agent_b_trade_cf_matrix)
np.save(agent_a_prebuy_log_path, agent_a_prebuy_log)
np.save(agent_b_prebuy_log_path, agent_b_prebuy_log)