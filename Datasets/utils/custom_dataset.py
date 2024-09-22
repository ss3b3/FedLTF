import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import copy
class client_dataset(Dataset):
    def __init__(self, Subset , dataset_name,num_classes, client_idx, noise_type, noise_ratio,is_train = True, transform = transforms.Compose([transforms.ToTensor()])):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.data, self.label = self.getdata(Subset)
        if type(self.data) != np.array:
            self.data = np.array(self.data)
        if type(self.label) != np.array:
            self.label = np.array(self.label)
        self.label = torch.from_numpy(self.label)
        self.label = self.label.to(torch.int64)
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.client_idx = client_idx
        self.is_train = is_train
        if not self.is_train:
            self.noise_type = 'clean'
            self.noise_ratio = 0
        if self.noise_type not in ['clean', 'real']:
            self.label_noise = self.add_noise()
            self.label_noise = self.label_noise.to(torch.int64)
        self.transform = transform
        self.len = len(self.data)

    def __getitem__(self, index):
        if self.dataset_name in ['clothing1M','tinyimagenet']:
            image = Image.open(self.data[index]).convert('RGB')
        # elif self.dataset_name in ['fmnist','mnist']:
        #     image = torchvision.transforms.ToPILImage(self.data[index])
        else:
            image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        label = self.label[index]
        if self.noise_type not in ['clean', 'real']:
            label_noise = self.label_noise[index]
            return index, image, label, label_noise
        else:
            return index, image, label
    def __len__(self):
        return self.len

    def getdata(self,Subset):
        data = Subset.dataset.data[Subset.indices]
        if type(Subset.dataset.targets) != np.array:
            label = np.array(Subset.dataset.targets)[Subset.indices]
        else:
            label = Subset.dataset.targets[Subset.indices]
        return data, label

    def add_noise(self):
        if self.dataset_name in ['mnist','fmnist']:
            return add_noise_mnist(self.label, self.num_classes, self.noise_type, self.noise_ratio)
        elif self.dataset_name in ['cifar10','cifar100']:
            return add_noise_cifar(self.label, self.num_classes, self.noise_type, self.noise_ratio)
        else:
            return add_noise_pairflip(self.label, self.num_classes, self.noise_type, self.noise_ratio)

class SSL_dataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset_name = dataset.dataset_name
        self.num_classes = dataset.num_classes
        self.data = dataset.data
        self.label = dataset.label
        self.noise_type = dataset.noise_type
        self.noise_ratio = dataset.noise_ratio
        self.client_idx = dataset.client_idx
        self.is_train = dataset.is_train
        if self.noise_type not in ['clean', 'real']:
            self.label_noise = dataset.label_noise
        self.transform = transforms
        self.len = len(self.data)
    def __getitem__(self, index):
        if self.dataset_name in ['clothing1M','tinyimagenet']:
            image = Image.open(self.data[index]).convert('RGB')
        else:
            image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        label = self.label[index]
        if self.noise_type not in ['clean', 'real']:
            label_noise = self.label_noise[index]
            return index, image, label, label_noise
        else:
            return index, image, label


class centralized_dataset(Dataset):
    def __init__(self, Dataset , dataset_name,num_classes, noise_type, noise_ratio,is_train = True, transform = transforms.Compose([transforms.ToTensor()])):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.data = Dataset.data
        self.label = Dataset.targets
        if type(self.data) != np.array:
            self.data = np.array(self.data)
        if type(self.label) != np.array:
            self.label = np.array(self.label)
        self.label = torch.from_numpy(self.label)
        self.label = self.label.to(torch.int64)
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.is_train = is_train
        if not self.is_train:
            self.noise_type = 'clean'
            self.noise_ratio = 0
        if self.noise_type not in ['clean', 'real']:
            self.label_noise = self.add_noise()
            self.label_noise = self.label_noise.to(torch.int64)
        self.transform = transform
        self.len = len(self.data)

    def __getitem__(self, index):
        if self.dataset_name in ['clothing1M','tinyimagenet']:
            image = Image.open(self.data[index]).convert('RGB')
        # elif self.dataset_name in ['fmnist','mnist']:
        #     torchvision.transforms.ToPILImage(self.data[index])
        else:
            image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        label = self.label[index]
        if self.noise_type not in ['clean', 'real']:
            label_noise = self.label_noise[index]
            return index, image, label, label_noise
        else:
            return index, image, label
    def __len__(self):
        return self.len

    def add_noise(self):
        if self.dataset_name in ['mnist','fmnist']:
            return add_noise_mnist(self.label, self.num_classes, self.noise_type, self.noise_ratio)
        elif self.dataset_name in ['cifar10','cifar100']:
            return add_noise_cifar(self.label, self.num_classes, self.noise_type, self.noise_ratio)
        else:
            return add_noise_pairflip(self.label, self.num_classes, self.noise_type, self.noise_ratio)

def add_noise_cifar(y, num_classes , noise_type, noise_rate):
    y_noised = copy.deepcopy(y)
    if noise_type == 'sym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        for i in noisy_idx:
            label_lst = list(range(num_classes))
            label_lst.remove(y_noised[i])
            y_noised[i] = random.sample(label_lst, k=1)[0]
    elif noise_type == 'asym':
        if num_classes == 10:
            transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
            noisy_num = int(noise_rate * len(y))
            noisy_idx = random.sample(range(len(y)), noisy_num)
            for i in noisy_idx:
                y_noised[i] = transition[int(y_noised[i])]
        elif num_classes == 100:
            noisy_num = int(noise_rate * len(y))
            noisy_idx = random.sample(range(len(y)), noisy_num)
            for i in noisy_idx:
                if (y_noised[i] + 1) % 5 == 0 :
                    y_noised[i] = y_noised[i] - 4
                else:
                    y_noised[i] = (y_noised[i] + 1) % 100
    return y_noised

def add_noise_mnist(y, num_classes , noise_type, noise_rate):
    y_noised = copy.deepcopy(y)
    if noise_type == 'sym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        for i in noisy_idx:
            label_lst = list(range(num_classes))
            label_lst.remove(y_noised[i])
            y_noised[i] = random.sample(label_lst, k=1)[0]
    elif noise_type == 'asym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        transition = {0: 0, 1: 1, 2: 7, 3: 8, 4: 4, 5: 6, 6: 5, 7: 7, 8: 8, 9: 9}
        #  2 -> 7, 3 -> 8, 5 <->  6
        for i in noisy_idx:
            y_noised[i] = transition[int(y_noised[i])]
    return y_noised

def add_noise_pairflip(y, num_classes , noise_type, noise_rate):
    y_noised = copy.deepcopy(y)
    if noise_type == 'sym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        for i in noisy_idx:
            y_noised[i] = random.randint(0, num_classes - 1)
    elif noise_type == 'asym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        for i in noisy_idx:
            y_noised[i] = (y_noised[i] + 1) % num_classes
    return y_noised