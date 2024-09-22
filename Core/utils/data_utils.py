import os
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
from Datasets.utils.custom_dataset import SSL_dataset

class MnistTransform:
    def __init__(self, size = 28, channel = 1):
        self.trans_mnist1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(28 * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.trans_mnist2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    def __call__(self, x):
        x1 = self.trans_mnist1(x)
        x2 = self.trans_mnist2(x)
        return x1, x2


class MoCoTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, channel = 3):
        if channel == 1:
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(mode='L'),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor()])

            self.test_transform = transforms.Compose([
                transforms.ToTensor()]
            )
        else:
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(mode='RGB'),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor()])

            self.test_transform = transforms.Compose([
                transforms.ToTensor()]
            )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class SimSiamTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size=32, gaussian=False, channel = 3):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        if gaussian:
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(mode='RGB'),
                    # torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(kernel_size=int(0.1 * size)),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif channel == 1:
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(mode='L'),
                    # torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(mode='RGB'),
                    # torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )

        self.test_transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class SimCLRTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    data_format is array or image
    """

    def __init__(self, size=32, gaussian=False, data_format="array", channel = 3):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        if gaussian:
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(mode='RGB'),
                    # torchvision.transforms.Resize(size=size),
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(kernel_size=int(0.1 * size)),
                    # RandomApply(torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif channel == 1:
            if data_format == "array":
                self.train_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToPILImage(mode='L'),
                        # torchvision.transforms.Resize(size=size),
                        torchvision.transforms.RandomResizedCrop(size=size),
                        torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        torchvision.transforms.RandomApply([color_jitter], p=0.8),
                        torchvision.transforms.RandomGrayscale(p=0.2),
                        torchvision.transforms.ToTensor(),
                    ]
                )
            else:
                self.train_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomResizedCrop(size=size),
                        torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        torchvision.transforms.RandomApply([color_jitter], p=0.8),
                        torchvision.transforms.RandomGrayscale(p=0.2),
                        torchvision.transforms.ToTensor(),
                    ]
                )
        else:
            if data_format == "array":
                self.train_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToPILImage(mode='RGB'),
                        # torchvision.transforms.Resize(size=size),
                        torchvision.transforms.RandomResizedCrop(size=size),
                        torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        torchvision.transforms.RandomApply([color_jitter], p=0.8),
                        torchvision.transforms.RandomGrayscale(p=0.2),
                        torchvision.transforms.ToTensor(),
                    ]
                )
            else:
                self.train_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomResizedCrop(size=size),
                        torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        torchvision.transforms.RandomApply([color_jitter], p=0.8),
                        torchvision.transforms.RandomGrayscale(p=0.2),
                        torchvision.transforms.ToTensor(),
                    ]
                )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.fine_tune_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(mode='RGB'),
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img




def get_dataset(path,idx):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    return torch.load(dataset_path)

def get_dataloder(path,idx,batch_size,isTrain,transform = None):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    dataset = torch.load(dataset_path)
    if transform is not None:
        dataset = transform_dataset(dataset, transform)
    if isTrain:
        return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,drop_last=False, shuffle=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=False, shuffle=False)

def get_semi_dataloader(path,idx,batch_size,index,isTrain):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    dataset = torch.load(dataset_path)
    label_dataset, unlabel_dataset = split_dataset_by_index(dataset, index)

    if isTrain:
        return {"label":torch.utils.data.DataLoader(
        label_dataset, batch_size=batch_size,drop_last=False, shuffle=True), "unlabel":torch.utils.data.DataLoader(
        unlabel_dataset, batch_size=batch_size,drop_last=False, shuffle=True)}
    else:
        return {"label":torch.utils.data.DataLoader(
            label_dataset, batch_size=batch_size, drop_last=False, shuffle=False), "unlabel": torch.utils.data.DataLoader(
            unlabel_dataset, batch_size=batch_size, drop_last=False, shuffle=False)}

def get_relabel_dataloader(path,idx,batch_size,index, relabel,isTrain):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    dataset = torch.load(dataset_path)
    dataset = relabel_dataset(dataset, index, relabel)
    if isTrain:
        return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,drop_last=False, shuffle=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=False, shuffle=False)



def get_global_testloader(path,batch_size, transform = None):
    dataset_path = os.path.join(path, 'test_dataset.pkl')
    dataset = torch.load(dataset_path)
    if transform is not None:
        dataset = transform_dataset(dataset, transform)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=False, shuffle=False)

def get_global_trainloader(path,batch_size, transform = None):
    dataset_path = os.path.join(path, 'train_dataset.pkl')
    dataset = torch.load(dataset_path)
    if transform is not None:
        dataset = transform_dataset(dataset, transform)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=False, shuffle=True)

def count_client_data(path,idx):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    dataset = torch.load(dataset_path)
    return len(dataset)

def get_SSL_Transform(method, size, channel = 3):
    if channel == 1:
        transform = MnistTransform(size = size, channel = channel)
    if method == 'moco':
        transform = MoCoTransform(size=size, gaussian=False, channel = channel)
    elif method == 'simsiam':
        transform = SimSiamTransform(size=size, gaussian=False, channel = channel)
    else:
        transform = SimCLRTransform(size=size, gaussian=False, channel = channel)
    return transform

class relabel_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, selected_index , relabel):
        # self.select_data = [dataset[i] for i in selected_index]
        self.data = []
        self.label = []
        if len(dataset[0]) == 3:
            self.Noise = False
            for i in range(len(selected_index)):
                self.data.append(torch.from_numpy(np.transpose(dataset.data[selected_index[i]],(2,0,1))))
                self.label.append(relabel[i])
        else:
            self.Noise = True
            self.label_noise = []
            for i in range(len(selected_index)):
                self.data.append(torch.from_numpy(np.transpose(dataset.data[selected_index[i]],(2,0,1))))
                self.label.append(dataset.label[selected_index[i]])
                self.label_noise.append(relabel[i])
    def __getitem__(self, index):
        if self.Noise:
            image, label, label_noise = self.data[index], self.label[index], self.label_noise[index]
            return index, image, label, label_noise
        else:
            image, label = self.data[index], self.label[index]
            return index, image, label

    def __len__(self):
        return len(self.data)



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class SSL_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.dataset.transform = transforms.ToTensor()
        self.transform = transform
        if len(dataset[0]) == 3:
            self.Noise = False
        else:
            self.Noise = True

    def __getitem__(self, index):
        if self.Noise:
            index, image, label, label_noise = self.dataset[index]
        else:
            index, image, label = self.dataset[index]
        image1, image2 = self.transform(image)
        # return index, image1, image2, label
        return index, image1, image2

    def __len__(self):
        return len(self.dataset)

class transform_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.dataset.transform = transforms.ToTensor()
        self.transform = transform
        if len(dataset[0]) == 3:
            self.Noise = False
        else:
            self.Noise = True

    def __getitem__(self, index):
        if self.Noise:
            index, image, label, label_noise = self.dataset[index]
            image = self.transform(image)
            return index, image, label, label_noise
        else:
            index, image, label = self.dataset[index]
            image = self.transform(image)
            return index, image, label
        # return index, image1, image2, label
        # return index, image, label

    def __len__(self):
        return len(self.dataset)


def get_SSL_dataloder(path,idx,batch_size,ssl_method,isTrain):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    dataset = torch.load(dataset_path)
    channel, size = dataset[0][1].shape[0], dataset[0][1].shape[1]
    # #
    # size = 32
    # #
    transforms = get_SSL_Transform(ssl_method, size = size, channel = channel)
    dataset = SSL_dataset(dataset, transforms)
    if isTrain:
        return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,drop_last=False, shuffle=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=False, shuffle=False)


class Feature_dataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, label_noise = None):
        self.features = features
        self.targets = targets
        self.num_sample = len(features)
        self.label_noise = label_noise
        if label_noise is not None:
            self.label_noise = label_noise

    def __getitem__(self, index):
        if self.label_noise is not None:
            x = self.features[index]
            y = self.targets[index]
            y_noise = self.label_noise[index]
            return index, x, y, y_noise
        x = self.features[index]
        y = self.targets[index]
        return index, x, y

    def __len__(self):
        return self.num_sample


def count_client_data_by_category(path, idx, noise_type):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    dataset = torch.load(dataset_path)
    if noise_type in ['clean', 'real']:
        labels = dataset.label
        labels = np.array(labels)
    else:
        labels = dataset.label_noise
        labels = np.array(labels)
    counts = {}
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    counts = dict(sorted(counts.items(), key=lambda item: item[0]))
    return counts

def count_samples_by_category_with_index(path, idx, noise_type, indexs):
    labels = []
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    dataset = torch.load(dataset_path)
    for index in indexs:
        if noise_type in ['clean', 'real']:
            label = dataset[index][2]
            labels.append(int(label))
        else:
            label = dataset[index][3].int()
            labels.append(int(label))
    counts = {}
    for i in labels:
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1
    #将字典按照键值从小到大排序
    counts = dict(sorted(counts.items(), key=lambda item: item[0]))
    return counts

def get_class_num(counts):
    index = []
    compose = []
    for class_index, j in counts.items():
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, label_noise = None):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label)
        self.label_noise = None
        if label_noise is not None:
            self.label_noise = torch.from_numpy(label_noise)

    def __getitem__(self, index):
        if self.label_noise is not None:
            x = self.data[index]
            y = self.label[index]
            y_noise = self.label_noise[index]
            return index, x, y, y_noise
        x = self.data[index]
        y = self.label[index]
        return index, x, y

    def __len__(self):
        return len(self.data)

def split_dataset_by_index(dataset, labeled_index):
    if len(dataset[0]) == 3:
        data = np.array(dataset.data)
        data = np.transpose(data, (0, 3, 1, 2))
        label = np.array(dataset.label)

        labeled_data = data[labeled_index]
        labeled_targets = label[labeled_index]

        unlabeled_index = np.setdiff1d(np.arange(len(data)), labeled_index)
        unlabeled_data = data[unlabeled_index]
        unlabeled_targets = label[unlabeled_index]

        labeled_dataset = BasicDataset(labeled_data, labeled_targets)
        unlabeled_dataset = BasicDataset(unlabeled_data, unlabeled_targets)

        return labeled_dataset, unlabeled_dataset

    else:
        data = np.array(dataset.data)
        data = np.transpose(data, (0, 3, 1, 2))
        label = np.array(dataset.label)
        label_noise = np.array(dataset.label_noise)

        labeled_data = data[labeled_index]
        labeled_targets = label[labeled_index]
        labeled_noise_targets = label_noise[labeled_index]

        unlabeled_index = np.setdiff1d(np.arange(len(data)), labeled_index)
        unlabeled_data = data[unlabeled_index]
        unlabeled_targets = label[unlabeled_index]
        unlabeled_label_noise = label_noise[unlabeled_index]

        labeled_dataset = BasicDataset(labeled_data, labeled_targets, labeled_noise_targets)
        unlabeled_dataset = BasicDataset(unlabeled_data, unlabeled_targets, unlabeled_label_noise)

        return labeled_dataset, unlabeled_dataset