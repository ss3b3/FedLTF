import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image
class Clothing1M_image_path(Dataset):
    def __init__(self, data, targets,transform=None):
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        label = self.targets[idx]
        if self.transform is not None:
            return self.transform(image), label
        else:
            return image,label

class Tinyimagenet(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)
        self.targets = np.array(imagefolder_obj.targets)
        self.data = np.array(imagefolder_obj.samples)[:, 0]
    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root , train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy()
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class FMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root , train=True, transform=None, target_transform=None, download=False):
        super(FMNIST, self).__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy()
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        return get_mnist()
    elif dataset_name == 'fmnist':
        return get_fmnist()
    elif dataset_name == 'cifar10':
        return get_cifar10()
    elif dataset_name == 'cifar100':
        return get_cifar100()
    elif dataset_name == 'tinyimagenet':
        return get_tinyimagenet()
    elif dataset_name == 'clothing1M':
        return get_clothing1M()
    else:
        raise Exception('Invalid dataset name')

def get_mnist(dir_path = "./Datasets/mnist/",transform = None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    trainset = MNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = MNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)

    return trainset,testset

def get_fmnist(dir_path = "./Datasets/fmnist/",transform = None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    trainset = FMNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = FMNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)

    return trainset,testset

def get_cifar10(dir_path = "./Datasets/cifar10/",transform = None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)

    return trainset,testset

def get_cifar100(dir_path="./Datasets/cifar100/", transform=None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)

    return trainset, testset

def get_tinyimagenet(dir_path="./Datasets/tinyimagenet/", transform = None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    trainset = Tinyimagenet(
        root=dir_path + "rawdata/tiny-imagenet-200/train/", transform=transform)
    testset = Tinyimagenet(
        root=dir_path + "rawdata/tiny-imagenet-200/val/", transform=transform)

    return trainset, testset

def get_clothing1M(dir_path="./Datasets/clothing1M/"):
    rawdata_path = os.path.join(dir_path, "rawdata")
    clean_labels_dict = {}
    noisy_labels_dict = {}
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    with open(os.path.join(rawdata_path, "category_names_eng.txt"), "r") as f:
        classes = f.read().splitlines()
        class_to_idx = {_class: i for i, _class in enumerate(classes)}

    with open(os.path.join(rawdata_path, "clean_label_kv.txt"), "r") as f:
        lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = os.path.join(rawdata_path, entry[0])
            clean_labels_dict[img_path] = int(entry[1])

    with open(os.path.join(rawdata_path, "clean_test_key_list.txt"), "r") as f:
        lines = f.read().splitlines()
        for l in lines:
            img_path = os.path.join(rawdata_path, l)
            test_data.append(img_path)
            test_labels.append(clean_labels_dict[img_path])

    with open(os.path.join(rawdata_path, "noisy_label_kv.txt"), "r") as f:
        lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = os.path.join(rawdata_path, entry[0])
            noisy_labels_dict[img_path] = int(entry[1])

    with open(
            os.path.join(rawdata_path, "noisy_train_key_list.txt"), "r"
    ) as f:
        lines = f.read().splitlines()
        for l in lines:
            img_path = os.path.join(rawdata_path, l)
            train_data.append(img_path)
            train_labels.append(noisy_labels_dict[img_path])

    trainset = Clothing1M_image_path(train_data, train_labels)
    testset = Clothing1M_image_path(test_data, test_labels)

    return trainset, testset