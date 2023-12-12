import torch

class MLP(torch.nn.Module):

    def __init__(self, seed, n_feature, n_hidden=1024, num_classes=10):
        super(MLP, self).__init__()
        torch.manual_seed(seed)
        self.input = n_feature
        self.layer0 = torch.nn.Linear(n_feature, n_hidden)   
        self.layer1 = torch.nn.Linear(n_hidden, n_hidden)   
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)   
        self.layer3 = torch.nn.Linear(n_hidden, int(n_hidden / 2))
        self.layer4 = torch.nn.Linear(int(n_hidden / 2), num_classes)
        
    def forward(self, x):
        x = x.view(-1, self.input)
        x = torch.nn.functional.relu(self.layer0(x))
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.relu(self.layer3(x))
        x = self.layer4(x)
        
        return torch.nn.functional.log_softmax(x)