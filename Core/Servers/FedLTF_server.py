import os.path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn
from torch.utils.tensorboard import SummaryWriter
from Core.Servers.Server_base import Server
from Core.Clients.FedLTF_Client import FedLTF_Client
from Core.Networks.SSL_model import MLP
from Core.Networks.ResNet import resnet50, resnet18
import random
import copy
from torch.utils.data.dataloader import DataLoader
from Core.utils.data_utils import TensorDataset
from Core.utils.data_utils import get_global_testloader
class FedLTF(Server):

    def __init__(self, args):
        super().__init__(args)

        self.set_clients(FedLTF_Client)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.rs_classifier_train_loss = []
        self.rs_classifier_test_acc = []
        self.rs_train_acc = []

        self.global_classifier_model = None

        self.aggregate_online_encoder = args.aggregate_online_encoder
        self.aggregate_target_encoder = args.aggregate_target_encoder
        self.aggregate_online_predictor = args.aggregate_online_predictor


        self.uploaded_online_encoder_models = []
        self.uploaded_online_predictor_models = []
        self.uploaded_target_encoder_models = []

        self.finetune_stage_epochs = args.finetune_stage_epochs

        self.linear_stage_epochs = args.linear_stage_epochs


        self.num_of_feature = args.num_of_feature


        self.Budget = []

    def get_encoder_model(self):
        if self.args.dataset in ['mnist', 'fmnist']:
            input_channel = 1
        else:
            input_channel = 3
        if self.args.encoder_network == 'resnet50':
            resnet = resnet50(num_classes=self.args.num_classes, input_channel=input_channel)
        elif self.args.encoder_network == 'resnet18':
            resnet = resnet18(num_classes=self.args.num_classes, input_channel=input_channel)
        else:
            raise NotImplementedError

        resnet.fc = torch.nn.Identity()

        resnet.fc = self.global_model.online_encoder.fc

        if self.args.encoder_model_path is not None:
            resnet.load_state_dict(torch.load(self.args.encoder_model_path, map_location=self.device))
        else:
            raise NotImplementedError

        self.global_model = resnet.to(self.device)

        num_features = list(self.global_model.children())[-1].in_features

        self.features_dim = num_features

        self.global_model.fc = torch.nn.Identity()

        # if self.args.classifier_use_MLP:
        #     self.global_classifier_model = MLP(num_features, self.args.num_classes, 4096)
        #     self.global_classifier_model = self.global_classifier_model.to(self.device)
        # else:
        #     self.global_classifier_model = torch.nn.Sequential(torch.nn.Linear(num_features, self.args.num_classes))
        #     self.global_classifier_model = self.global_classifier_model.to(self.device)

        # self.global_classifier_model = MLP(num_features, self.args.num_classes, 4096)
        # self.global_classifier_model = self.global_classifier_model.to(self.device)

        self.global_classifier_model = torch.nn.Sequential(torch.nn.Linear(num_features, self.args.num_classes))
        self.global_classifier_model = self.global_classifier_model.to(self.device)

        self.classifier_optimizer = torch.optim.Adam(self.global_classifier_model.parameters(), lr= self.args.linear_stage_lr)

    def client_preparation(self):
        for client in self.clients:
            start_time = time.time()

            client.prepare_next_stage(self.global_model, self.global_classifier_model)
            client.get_features()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def prepare_client_finetune(self):

        for client in self.clients:
            start_time = time.time()

            client.prepare_next_stage_finetune(self.global_model, self.global_classifier_model)


            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def train(self):
        if self.tensorboard:
            tensorboard_path = os.path.join(self.result_dir, 'log')
            writer = SummaryWriter(tensorboard_path)
            print('tensorboard log file path: ', tensorboard_path)

        self.get_encoder_model()
        self.client_preparation()

        for i in range(self.linear_stage_epochs):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_classifier_models()

            averaged_loss, train_acc = self.local_classifier_train()

            averaged_acc = self.local_classifier_test()

            self.rs_train_loss.append(averaged_loss)
            self.rs_train_acc.append(train_acc)
            self.rs_test_acc.append(averaged_acc)

            if self.tensorboard:
                writer.add_scalar('train_loss', averaged_loss, i)
                writer.add_scalar('train_acc', train_acc, i)
                writer.add_scalar('test_acc', averaged_acc, i)
            if i % self.eval_gap == 0:
                print(f"\n-------------Linear Evaluation Round number: {i}-------------")
                print("\nEvaluate global model")
                print("Averaged Train loss:{:.4f}".format(averaged_loss))
                print("Averaged Train acc:{:.4f}".format(train_acc))
                print("Averaged Test acc:{:.4f}".format(averaged_acc))

            self.receive_classifier_models()
            self.aggregate_classifier_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

        self.save_classifier_model()

        self.prepare_client_finetune()


        for i in range(self.linear_stage_epochs, self.args.finetune_stage_epochs + self.linear_stage_epochs):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.send_classifier_models()


            averaged_loss, train_acc = self.local_finetune_train()



            averaged_acc = self.local_finetune_test()


            self.rs_train_loss.append(averaged_loss)
            self.rs_train_acc.append(train_acc)
            self.rs_test_acc.append(averaged_acc)

            # print(self.rs_test_acc[-1])

            if self.tensorboard:
                writer.add_scalar('train_loss', averaged_loss, i)
                writer.add_scalar('train_acc', train_acc, i)
                writer.add_scalar('test_acc', averaged_acc, i)
            if i % self.eval_gap == 0:
                print(f"\n-------------Finetune Evaluation Round number: {i - self.linear_stage_epochs}-------------")
                print("\nEvaluate global model")
                print("Averaged Train loss:{:.4f}".format(averaged_loss))
                print("Averaged Train acc:{:.4f}".format(train_acc))
                print("Averaged Test acc:{:.4f}".format(averaged_acc))
            self.receive_models()
            self.aggregate_parameters()

            self.receive_classifier_models()
            self.aggregate_classifier_parameters()



            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        print('\nglobal model test acc: ', self.global_eval())

        self.save_results()
        self.save_whole_model()

    def aggregate_classifier_parameters(self):
        self.global_classifier_model.load_state_dict(self._federated_averaging(self.uploaded_models,
                                                                     self.uploaded_weights).state_dict())
    def global_eval(self,test_loader=None):
        total_acc = 0
        total_test_num = 0
        if test_loader is None:
            test_loader = get_global_testloader(self.dataset_path, self.batch_size)
        self.global_model = self.global_model.to(self.device)
        self.global_classifier_model = self.global_classifier_model.to(self.device)
        self.global_model.eval()
        self.global_classifier_model.eval()
        with torch.no_grad():
            for i, (index, image, label) in enumerate(test_loader):
                label = label.view(-1)
                image = image.to(self.device)
                label = label.to(self.device)
                features = self.global_model(image)
                output = self.global_classifier_model(features)
                total_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                total_test_num += label.shape[0]
        averaged_acc = total_acc * 1.0 / total_test_num
        return averaged_acc

    def global_train(self,train_loader):
        # trainloader = get_global_trainloader(self.dataset_path, self.batch_size)
        self.global_classifier_model.train()
        # start_time = time.time()
        train_num = 0
        losses = 0
        total_acc = 0
        for step in range(self.args.classifier_stage_local_epochs):
            for i, (features, label) in enumerate(train_loader):
                label = label.view(-1)
                features = features.to(self.device)
                label = label.to(self.device)
                output = self.global_classifier_model(features)
                loss = torch.nn.functional.cross_entropy(output, label)
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
                train_num += label.shape[0]
                losses += loss.item() * label.shape[0]
                total_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
        return losses, total_acc, train_num



    def save_results(self):
        server_result_path = os.path.join(self.result_dir, "server_result")
        if not os.path.exists(server_result_path):
            os.makedirs(server_result_path)

        server_result_path = os.path.join(server_result_path, "server_classifier_result.csv")
        server_result_df = pd.DataFrame({'train_loss':self.rs_train_loss,'test_acc':self.rs_test_acc})
        server_result_df.to_csv(server_result_path,index=False)

        for client in self.clients:
            client.save_result()

    def save_ssl_model(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, "ssl_model.pt")
        torch.save(self.global_model , model_path)

    def save_finetuned_model(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = self.global_model
        model.fc = torch.nn.Identity()
        model.fc = self.global_classifier_model
        model = model.to('cpu')
        torch.save(model , os.path.join(model_path, "finetuned_model.pt"))


    def save_classifier_model(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = copy.deepcopy(self.global_model)
        model.fc = torch.nn.Identity()
        model.fc = self.global_classifier_model
        model = model.to('cpu')
        torch.save(model, os.path.join(model_path, "classifier_model.pt"))

    def save_whole_model(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = self.global_model
        model.fc = torch.nn.Identity()
        model.fc = self.global_classifier_model
        model = model.to('cpu')
        torch.save(model, os.path.join(model_path, "whole_model.pt"))



    def local_classifier_train(self):
        id = []
        total_losses = []
        total_train_num = []
        total_acc = []

        for client in self.clients:

            losses, acc ,train_num = client.train_classifier()
            id.append(client.id)
            total_losses.append(losses)
            total_train_num.append(train_num)
            total_acc.append(acc)

        averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
        averaged_train_acc = sum(total_acc) * 1.0 / sum(total_train_num)
        return averaged_loss, averaged_train_acc

    def local_classifier_test(self):
        id = []
        total_acc = []
        total_test_num = []
        for client in self.clients:
            acc,test_num = client.test_classifier()
            id.append(client.id)
            total_acc.append(acc)
            total_test_num.append(test_num)

        averaged_acc = sum(total_acc) * 1.0 / sum(total_test_num)
        return averaged_acc

    def local_finetune_train(self):
        id = []
        total_losses = []
        total_train_num = []
        total_acc = []

        for client in self.clients:

            losses, acc ,train_num = client.train_finetune()
            id.append(client.id)
            total_losses.append(losses)
            total_train_num.append(train_num)
            total_acc.append(acc)

        averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
        averaged_train_acc = sum(total_acc) * 1.0 / sum(total_train_num)
        return averaged_loss, averaged_train_acc

    def local_finetune_test(self):
        id = []
        total_acc = []
        total_test_num = []
        for client in self.clients:
            acc,test_num = client.test_finetune()
            id.append(client.id)
            total_acc.append(acc)
            total_test_num.append(test_num)

        averaged_acc = sum(total_acc) * 1.0 / sum(total_test_num)
        return averaged_acc



    def prepare_client_next_stage(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.prepare_next_stage(self.global_model, self.global_classifier_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def send_classifier_models(self, model=None):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            if model is None:
                client.set_classifier_parameters(self.global_classifier_model)
            else:
                client.set_classifier_parameters(model)

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
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        self.global_model.load_state_dict(self._federated_averaging(self.uploaded_models,
                                                                     self.uploaded_weights).state_dict())


    def _federated_averaging(self, models, weights):
        if models == [] or weights == []:
            return None
        model = copy.deepcopy(models[0])
        for param in model.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, models):
            self.add_model_parameters(w, model, client_model)

        return model


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)


            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)




