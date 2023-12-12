import torch
import numpy as np
import torchvision

class MyCustomDataset(torch.utils.data.dataset.Dataset):
    
    def __init__(self, task_name, train, sample=False, class_freq=None, root_dir='/hdd1/Parameter_Market_Old/datasets'):
        
        self.task_name = task_name
        self.train = train
        self.sample = sample
        self.class_freq = class_freq
        self.root_dir = root_dir
        self.entire_dataset = self.load_dataset()
        self.sampled_dataset = self.sampler()
    
    def load_dataset(self):
        
        if self.task_name == "mnist":
            self.transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            dataset = torchvision.datasets.MNIST(self.root_dir, train = self.train, download = False)
        elif self.task_name == "fashion":
            self.transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            dataset = torchvision.datasets.FashionMNIST(self.root_dir, train = self.train, download = False)
        elif self.task_name == "kmnist":
            self.transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            dataset = torchvision.datasets.KMNIST(self.root_dir, train = self.train, download = False)
        elif self.task_name == "emnist":
            self.transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                                  lambda img: torchvision.transforms.functional.rotate(img, -90),
                                                                  lambda img: torchvision.transforms.functional.hflip(img),
                                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            dataset = torchvision.datasets.EMNIST(self.root_dir, train = self.train, split="balanced", download = False)
        elif self.task_name == "cifar10":
            if self.train == True:
                self.transformation = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(),
                     torchvision.transforms.RandomCrop(32, padding=4),
                     torchvision.transforms.RandomHorizontalFlip(),
                     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                dataset = torchvision.datasets.CIFAR10(self.root_dir, train = self.train, download = False)
            else:
                self.transformation = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                dataset = torchvision.datasets.CIFAR10(self.root_dir, train = self.train, download = False)
            dataset.targets = np.array(dataset.targets)
        return dataset

    def sampler(self):
        
        if self.sample == True:
            targets_np = self.entire_dataset.targets
            total_sampled_index = []
            for class_, freq_ in self.class_freq.items():
                specific_class_total_index = np.where(targets_np == class_)[0]
                sampled_data_index = np.random.choice(specific_class_total_index, size=freq_, replace=False)
                total_sampled_index.extend(sampled_data_index)
            total_sampled_index = np.array(total_sampled_index)
            self.entire_dataset.data = self.entire_dataset.data[total_sampled_index]
            self.entire_dataset.targets = self.entire_dataset.targets[total_sampled_index]
        return self.entire_dataset
        
    def __getitem__(self, index):
        
        if type(self.sampled_dataset.data) == torch.Tensor:
            img = self.transformation(self.sampled_dataset.data[index].numpy())
        else:
            img = self.transformation(self.sampled_dataset.data[index])
        label = self.sampled_dataset.targets[index]
        
        return (img, label)

    def __len__(self):
        
        return len(self.sampled_dataset.targets)