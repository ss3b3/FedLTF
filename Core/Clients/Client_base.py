import torch
import os
import numpy as np
import pandas as pd
import copy
import time
import random
import matplotlib.pyplot as plt
from Core.utils.data_utils import get_dataloder
from Core.utils.criteria import get_criterion
from Core.utils.optimizers import get_optimizer
class Client(object):
    def __init__(self,args,id,train_samples,test_samples):
        self.args = args
        self.dataset = args.dataset
        self.model = copy.deepcopy(args.model)
        self.device = args.device
        self.id = id
        self.num_clients = args.num_clients


        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.local_epochs = args.local_epochs
        self.warm_up_steps = args.warm_up_steps
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        num_classes = {'mnist': 10, 'fmnist': 10, 'cifar10': 10, 'cifar100': 100, 'clothing1M': 14,
                            'tinyimagenet': 200}
        self.num_classes = num_classes[self.dataset]


        self.criterion = get_criterion(args.criterion, self.num_classes, train_samples, args, True)
        self.optimizer = get_optimizer(args.optimizer, self.model.parameters(),self.learning_rate,self.momentum,self.weight_decay, args)

        self.centralized = args.centralized
        self.globalize = args.globalize
        self.partition = args.partition
        self.balance = args.balance
        self.dir_alpha = args.dir_alpha
        self.class_per_client = args.class_per_client
        self.max_samples = args.max_samples
        self.least_samples = args.least_samples

        self.noise_type = args.noise_type
        self.noise_ratio = args.noise_ratio
        self.min_noise_ratio = args.min_noise_ratio
        self.max_noise_ratio = args.max_noise_ratio
        if self.noise_type == 'clean':
            self.noise_ratio = 0.0
            self.min_noise_ratio = 0.0
            self.max_noise_ratio = 0.0
        self.data_dir = args.data_dir

        balance_str = "balance" if self.balance else "imbalance"

        if self.globalize:
            self.dataset_path = os.path.join(self.data_dir, self.dataset, self.noise_type + "_" + str(
                self.noise_ratio) + "_" + self.partition + "_" + balance_str + "_" + str(self.num_clients))
        else:
            self.dataset_path = os.path.join(self.data_dir, self.dataset,
                                             self.noise_type + "_max" + str(self.max_noise_ratio) + "_min" + str(
                                                 self.min_noise_ratio) + self.partition + "_" + balance_str + "_" + str(
                                                 self.num_clients))
        if not os.path.exists(self.dataset_path):
            try:
                os.makedirs(self.dataset_path)
            except FileNotFoundError:
                os.makedirs(self.dataset_path)

        self.train_path = os.path.join(self.dataset_path, "train")
        self.test_path = os.path.join(self.dataset_path, "test")

        if not os.path.exists(self.train_path):
            try:
                os.makedirs(self.train_path)
            except FileNotFoundError:
                os.makedirs(self.train_path)
        if not os.path.exists(self.test_path):
            try:
                os.makedirs(self.test_path)
            except FileNotFoundError:
                os.makedirs(self.test_path)

        self.result_dir = args.result_dir

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.train_loss = []
        self.test_acc = []

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

    def warm_up_train(self):
        if self.warm_up_steps != 0:
            trainloader = get_dataloder(self.train_path,self.id, self.batch_size,True)
            self.model.train()
            for step in range(self.warm_up_step):
                if self.noise_type  in ['clean', 'real']:
                    for i,(index, image, label) in enumerate(trainloader):
                        label = label.view(-1)
                        image = image.to(self.device)
                        label = label.to(self.device)
                        output = self.model(image)
                        loss = self.criterion(output, label)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                else:
                    for i,(index, image, label, label_noise) in enumerate(trainloader):
                        label_noise = label_noise.view(-1)
                        image = image.to(self.device)
                        label_noise = label_noise.to(self.device)
                        output = self.model(image)
                        loss = self.criterion(output, label_noise)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

    def eval(self):
        testloader = get_dataloder(self.test_path, self.id, self.batch_size, False)
        self.model.eval()
        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for i,(index, image, label) in enumerate(testloader):
                label = label.view(-1)
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.model(image)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                test_num += label.shape[0]
        self.test_acc.append(test_acc / test_num)
        return test_acc , test_num

    def train(self):
        trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)
        self.model.train()
        start_time = time.time()
        train_num = 0
        losses = 0

        for step in range(self.local_epochs):
            if self.noise_type in ['clean', 'real']:
                for i,(index, image, label) in enumerate(trainloader):
                    label = label.view(-1)
                    image = image.to(self.device)
                    label = label.to(self.device)
                    output = self.model(image)
                    loss = self.criterion(output, label)

                    train_num += label.shape[0]
                    losses += loss.item() * label.shape[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                for i,(index, image, label, label_noise) in enumerate(trainloader):
                    label_noise = label_noise.view(-1)
                    image = image.to(self.device)
                    label_noise = label_noise.to(self.device)
                    output = self.model(image)
                    loss = self.criterion(output, label_noise)

                    train_num += label_noise.shape[0]
                    losses += loss.item() * label_noise.shape[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        self.train_loss.append(losses / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return losses, train_num

    def save_result(self):
        result_path = os.path.join(self.result_dir, "client_result" , str(self.id))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_path_csv = os.path.join(result_path, str(self.id) + ".csv")
        result_df = pd.DataFrame({'train_loss': self.train_loss, 'test_acc': self.test_acc})
        result_df.to_csv(result_path_csv, index=False)

        plt.plot(result_df['train_loss'])
        plt.title('train loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(os.path.join(result_path ,'train_loss.png'))

        plt.clf()

        plt.plot(result_df['test_acc'])
        plt.title('test acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        maxx = np.argmax(result_df['test_acc'])
        maxy = np.max(result_df['test_acc'])
        plt.annotate('max_acc' + '(' + str(maxx) + ',' + '{:.3f}'.format(maxy) + ')', xy=(maxx, maxy),xytext=(maxx, maxy))
        plt.savefig(os.path.join(result_path ,'test_acc.png'))

        plt.clf()

    # model->client.model
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()



    ## model -> target
    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()