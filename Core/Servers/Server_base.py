import torch
import os
import numpy as np
import pandas as pd
import copy
import time
import random
import matplotlib.pyplot as plt
from Core.utils.data_utils import count_client_data,get_global_testloader
from Core.utils.criteria import get_criterion

class Server(object):
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.device = args.device
        self.num_clients = args.num_clients
        self.clients = []
        self.selected_clients = []
        self.lr = args.learning_rate
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.random_join_ratio = args.random_join_ratio
        self.client_drop_rate = args.client_drop_rate
        self.warm_up_steps = args.warm_up_steps
        self.result_dir = args.result_dir
        self.eval_gap = args.eval_gap
        self.just_eval_global_model = args.just_eval_global_model
        self.current_round = 0
        self.global_model = copy.deepcopy(args.model)
        self.resume = args.resume
        self.tensorboard = args.tensorboard

        num_classes = {'mnist': 10, 'fmnist': 10, 'cifar10': 10, 'cifar100': 100, 'clothing1M': 14,
                            'tinyimagenet': 200}
        self.num_classes = num_classes[self.dataset]

        self.criterion = get_criterion(args.criterion, self.num_classes, None, args, False)

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

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_acc = []

        self.rs_train_loss = []
        self.rs_test_acc = []

        self.stop_time = 100
    def set_clients(self,clientObj):
        for i in range(self.num_clients):
            train_data_num = count_client_data(self.train_path, i)
            test_data_num = count_client_data(self.test_path, i)
            print("Client {} train data: {}".format(i, train_data_num))
            print("Client {} test data: {}".format(i, test_data_num))
            client = clientObj(self.args,
                               id=i,
                               train_samples=train_data_num,
                               test_samples=test_data_num,
                               )
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.num_join_clients))


        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            # try:
            #     client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
            #                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            # except ZeroDivisionError:
            #     client_time_cost = 0
            # if client_time_cost <= self.time_threthold:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def add_model_parameters(self, w, global_model, client_model):
        for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
        return global_model

    def add_classifier_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_classifier_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def receive_classifier_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.num_join_clients))


        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            # try:
            #     client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
            #                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            # except ZeroDivisionError:
            #     client_time_cost = 0
            # if client_time_cost <= self.time_threthold:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.classifier)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_classifier_parameters(self):
        # self.global_classifier_model.load_state_dict(self._federated_averaging(self.uploaded_models,
        #                                                              self.uploaded_weights).state_dict())
        assert (len(self.uploaded_models) > 0)

        self.global_classifier_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_classifier_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_classifier_parameters(w, client_model)

    def add_classifier_parameters(self, w, client_classifier_model):
        for server_param, client_param in zip(self.global_classifier_model.parameters(), client_classifier_model.parameters()):
            server_param.data += client_param.data.clone() * w


    def save_current_model(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.current_round + ".pt")
        torch.save(self.global_model , model_path)

    def save_global_model(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, "best_model.pt")
        torch.save(self.global_model , model_path)

    def load_model(self):
        model_path = os.path.join(self.result_dir, "model")
        model_path = os.path.join(model_path, "best_model.pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)


    def load_model_from_path(self, path):
        assert (os.path.exists(path))
        self.global_model = torch.load(path)

    def local_train(self):
        id = []
        total_losses = []
        total_train_num = []
        for client in self.clients:
            losses,train_num = client.train()
            id.append(client.id)
            total_losses.append(losses)
            total_train_num.append(train_num)

        averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
        return averaged_loss

    def local_eval(self):
        id = []
        total_acc = []
        total_test_num = []
        for client in self.clients:

            acc,test_num = client.eval()
            id.append(client.id)
            total_acc.append(acc)
            total_test_num.append(test_num)

        averaged_acc = sum(total_acc) * 1.0 / sum(total_test_num)

        return averaged_acc

    def global_eval(self):
        total_acc = 0
        total_test_num = 0

        testloader = get_global_testloader(self.dataset_path, self.batch_size)

        self.global_model.eval()
        with torch.no_grad():
            for i, (index, image, label) in enumerate(testloader):
                label = label.view(-1)
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.global_model(image)
                total_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                total_test_num += label.shape[0]
        averaged_acc = total_acc * 1.0 / total_test_num
        return averaged_acc


    def save_results(self):
        server_result_path = os.path.join(self.result_dir, "server_result")
        if not os.path.exists(server_result_path):
            os.makedirs(server_result_path)
        server_result_csv = os.path.join(server_result_path, "server_result.csv")
        server_result_df = pd.DataFrame({'train_loss':self.rs_train_loss,'test_acc':self.rs_test_acc})
        server_result_df.to_csv(server_result_csv,index=False)

        plt.plot(server_result_df['train_loss'])
        plt.title('train loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(os.path.join(server_result_path ,'global_train_loss.png'))

        plt.clf()

        plt.plot(server_result_df['test_acc'])
        plt.title('test acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        maxx = np.argmax(server_result_df['test_acc'])
        maxy = np.max(server_result_df['test_acc'])
        plt.annotate('max_acc' + '(' + str(maxx) + ',' + '{:.3f}'.format(maxy) + ')', xy=(maxx, maxy),
                     xytext=(maxx, maxy))
        plt.savefig(os.path.join(server_result_path, 'global_test_acc.png'))

        plt.clf()

        for client in self.clients:
            client.save_result()

    def early_stop(self):
        max_acc, max_index = max(enumerate(self.rs_test_acc), key=lambda x: x[1])
        if max_index + self.stop_time >= self.current_round:
            return False
        else:
            return True
