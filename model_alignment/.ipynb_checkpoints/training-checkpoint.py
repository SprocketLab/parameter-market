import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

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
                acc = 100. * correct / len(dataloader.dataset)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))
            
    train_loss /= count
    acc = 100. * correct / len(dataloader.dataset)
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

def cf_matrix_computation(model, dataloader, softmax=False):

    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to("cuda:0")
            output = model(data)
            if softmax:
                output = F.log_softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True).cpu().numpy().ravel()
            predictions.extend(pred)
            ground_truths.extend(target.numpy().ravel())
    cf_matrix = confusion_matrix(ground_truths, predictions)
    
    return cf_matrix