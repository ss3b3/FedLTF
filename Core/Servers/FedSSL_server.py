import os.path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn
import random
import copy
from torch.utils.tensorboard import SummaryWriter
from Core.Servers.Server_base import Server
from Core.Clients.FedSSL_client import FedSSLClient
from Core.Networks.SSL_model import MLP
from Core.Networks.ResNet import resnet50, resnet18
from Core.utils.data_utils import get_global_trainloader, get_global_testloader
from Core.utils.knn_monitor import knn_monitor
class FedSSL(Server):

    def __init__(self, args):
        super().__init__(args)

        self.set_clients(FedSSLClient)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.rs_classifier_train_loss = []
        self.rs_classifier_test_acc = []
        self.rs_ssl_acc = []

        self.global_classifier_model = None

        self.aggregate_online_encoder = args.aggregate_online_encoder
        self.aggregate_target_encoder = args.aggregate_target_encoder
        self.aggregate_online_predictor = args.aggregate_online_predictor


        self.uploaded_online_encoder_models = []
        self.uploaded_online_predictor_models = []
        self.uploaded_target_encoder_models = []

        self.linear_stage_epochs = args.linear_stage_epochs

        self.Budget = []

    def get_encoder_model(self):
        if self.dataset in ['mnist', 'fmnist']:
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

        self.global_model.fc = torch.nn.Identity()


        self.global_classifier_model = torch.nn.Sequential(torch.nn.Linear(num_features, self.args.num_classes))
        self.global_classifier_model = self.global_classifier_model.to(self.device)

        # self.classifier_optimizer = torch.optim.Adam(self.global_classifier_model.parameters(), lr= self.args.classifier_stage_lr)

    def train(self):
        if self.tensorboard:
            tensorboard_path = os.path.join(self.result_dir,'log')
            writer = SummaryWriter(tensorboard_path)
            print('tensorboard log file path: ', tensorboard_path)

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()


            averaged_loss = self.local_encoder_train()

            self.rs_train_loss.append(averaged_loss)

            if i % self.eval_gap == 0:
                print(f"\n-------------SSL Round number: {i}-------------")
                print("\nEvaluate global model")
                print("Averaged Train loss:{:.4f}".format(averaged_loss))

            if self.tensorboard:
                writer.add_scalar('train_SSL_loss', averaged_loss, i)



            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            torch.cuda.empty_cache()
            if i % 10 == 0:
                self.save_current_ssl_model()
                torch.cuda.empty_cache()

        self.save_ssl_model()

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()


        torch.cuda.empty_cache()
        torch.cuda.empty_cache()


    def save_results(self):
        server_result_path = os.path.join(self.result_dir, "server_result")
        if not os.path.exists(server_result_path):
            os.makedirs(server_result_path)
        server_result_path = os.path.join(server_result_path, "server_ssl_result.csv")
        server_result_df = pd.DataFrame({'train_loss':self.rs_train_loss})

        server_result_df.to_csv(server_result_path,index=False)

        for client in self.clients:
            client.save_result()

    def save_ssl_model(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path_ = os.path.join(model_path, "ssl_model.pth")
        model_path_all = os.path.join(model_path, "ssl_model_all.pt")
        torch.save(self.global_model.online_encoder.cpu().state_dict(), model_path_)
        torch.save(self.global_model, model_path_all)
        print(f"Save model to {model_path}")

    def save_current_ssl_model(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path,"ssl_model_" + str(self.current_round) + ".pth")
        torch.save(self.global_model.online_encoder.cpu().state_dict(), model_path)
        print(f"Save model to {model_path}")


    def local_encoder_train(self):
        id = []
        total_losses = []
        total_train_num = []
        for client in self.clients:
            losses,train_num = client.train_encoder()
            id.append(client.id)
            total_losses.append(losses)
            total_train_num.append(train_num)

        averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
        return averaged_loss

    def prepare_next_stage(self):
        if self.dataset in ['mnist', 'fmnist']:
            input_channel = 1
        else:
            input_channel = 3
        if self.args.encoder_network == 'resnet50':
            resnet = resnet50(num_classes=self.args.num_classes, input_channel=input_channel)
        elif self.args.encoder_network == 'resnet18':
            resnet = resnet18(num_classes=self.args.num_classes, input_channel=input_channel)
        else:
            raise NotImplementedError
        num_features = list(resnet.children())[-1].in_features
        resnet.fc = torch.nn.Identity()

        resnet.fc = self.global_model.online_encoder.fc

        resnet.load_state_dict(self.global_model.online_encoder.state_dict())

        self.global_model = resnet.to(self.device)

        self.global_model.fc = torch.nn.Identity()

        if self.args.classifier_use_MLP:
            self.global_classifier_model = MLP(num_features, self.args.num_classes, 4096)
            self.global_classifier_model = self.global_classifier_model.to(self.device)
        else:
            self.global_classifier_model = torch.nn.Sequential(torch.nn.Linear(num_features, self.args.num_classes))
            self.global_classifier_model = self.global_classifier_model.to(self.device)


    def prepare_client_next_stage(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.prepare_next_stage()
            client.get_features()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        # self.uploaded_online_encoder_models = []
        # self.uploaded_online_predictor_models = []
        # self.uploaded_target_encoder_models = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
            # if self.aggregate_online_encoder:
            #     self.uploaded_online_encoder_models.append(client.model.online_encoder)
            # if self.aggregate_online_predictor:
            #     self.uploaded_online_predictor_models.append(client.model.online_predictor)
            # if self.aggregate_target_encoder:
            #     self.uploaded_target_encoder_models.append(client.model.target_encoder)

        # for i, w in enumerate(self.uploaded_weights):
        #     self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        if self.aggregate_online_encoder:
            self.global_model.online_encoder.load_state_dict(self._federated_averaging([m.online_encoder for m in self.uploaded_models],
                                                                         self.uploaded_weights).state_dict())
        if self.aggregate_online_predictor:
            self.global_model.online_predictor.load_state_dict(self._federated_averaging([m.online_predictor for m in self.uploaded_models],
                                                                         self.uploaded_weights).state_dict())
        if self.aggregate_target_encoder:
            if self.global_model.target_encoder is None:
                self.global_model.target_encoder = copy.deepcopy(self.global_model.online_encoder)
            self.global_model.target_encoder.load_state_dict(self._federated_averaging([m.target_encoder for m in self.uploaded_models],
                                                                         self.uploaded_weights).state_dict())
        torch.cuda.empty_cache()
    def _federated_averaging(self, models, weights):
        if models == [] or weights == []:
            return None
        model, total_weights = weighted_sum(models, weights)
        model_params = model.state_dict()
        with torch.no_grad():
            for name, params in model_params.items():
                model_params[name] = torch.div(params, total_weights)
        model.load_state_dict(model_params)

        return model


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            if self.aggregate_online_encoder:
                client.set_online_encoder_parameters(self.global_model.online_encoder)
            if self.aggregate_online_predictor:
                client.set_online_predictor_parameters(self.global_model.online_predictor)
            if self.aggregate_target_encoder:
                if self.args.SSL_method == 'byol':
                    client.set_target_encoder_parameters(self.global_model.online_encoder)
                else:
                    client.set_target_encoder_parameters(self.global_model.target_encoder)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


def weighted_sum(models, weights):
    if models == [] or weights == []:
        return None
    model = copy.deepcopy(models[0])
    model_sum_params = copy.deepcopy(models[0].state_dict())
    with torch.no_grad():
        for name, params in model_sum_params.items():
            params *= weights[0]
            for i in range(1, len(models)):
                model_params = dict(models[i].state_dict())
                params += model_params[name] * weights[i]
            model_sum_params[name] = params
    model.load_state_dict(model_sum_params)
    return model, sum(weights)