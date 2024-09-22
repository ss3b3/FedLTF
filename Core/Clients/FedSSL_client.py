import copy
import time
import os
import pandas as pd
import numpy as np

import torch

from Core.Clients.Client_base import Client
from Core.utils.data_utils import get_dataloder, get_SSL_dataloder
from Core.Networks.SSL_model import MLP
from Core.Networks.ResNet import resnet50, resnet18
from Core.utils.optimizers import get_optimizer

class FedSSLClient(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)
        if args.SSL_method != None:
            self.SSL_method = args.SSL_method
        self.classifier = None
        self.classifier_optimizer = None
        self.train_classifier_loss = []
        self.test_acc = []

        self.current_round = 0
        self.optimizer = self.load_optimizer()

        self.train_features_loader = None
        self.test_features_loader = None

    def load_optimizer(self):
        lr = self.args.learning_rate
        if self.args.lr_type == "cosine":
            lr = compute_lr(self.current_round, self.args.global_rounds, 0, self.args.learning_rate)

        params = self.model.parameters()
        if self.args.SSL_method in ['byol']:
            params = [
                {'params': self.model.online_encoder.parameters()},
                {'params': self.model.online_predictor.parameters()}
            ]

        optimizer = get_optimizer(self.args.optimizer, params, lr ,self.momentum,self.weight_decay, self.args)
        self.learning_rate = lr
        return optimizer

    def train_encoder(self):
        self.current_round += 1
        self.optimizer = self.load_optimizer()
        trainloader = get_SSL_dataloder(self.train_path, self.id,self.batch_size,self.SSL_method, True)
        self.model.train()
        start_time = time.time()
        train_num = 0
        losses = 0

        for step in range(self.local_epochs):
            for i,(index, image1, image2) in enumerate(trainloader):
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                if self.SSL_method in ['mocov2', 'moco']:
                    loss = self.model(image1, image2, self.device)
                else:
                    loss = self.model(image1, image2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_num += image1.shape[0]
                losses += loss.item() * image1.shape[0]

                if self.SSL_method in ['mocov2', 'moco', 'byol']:
                    self.model.update_moving_average()

        self.train_loss.append(losses / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        torch.cuda.empty_cache()

        return losses, train_num

    def get_features(self):
        train_features_vector = []
        train_labels_vector = []
        test_features_vector = []
        test_labels_vector = []
        self.model.eval()
        train_loader = get_dataloder(self.train_path, self.id, self.batch_size, True)
        test_loader = get_dataloder(self.test_path, self.id, self.batch_size, False)

        if self.noise_type in ['clean', 'real']:
            for i, (index, image, label) in enumerate(train_loader):
                image = image.to(self.device)
                with torch.no_grad():
                    features = self.model(image)
                features = features.squeeze()
                features = features.detach()
                train_features_vector.extend(features.cpu().detach().numpy())
                train_labels_vector.extend(label.numpy())
        else:
            for i, (index, image, label, label_noise) in enumerate(train_loader):
                image = image.to(self.device)
                with torch.no_grad():
                    features = self.model(image)
                features = features.squeeze()
                features = features.detach()
                train_features_vector.extend(features.cpu().detach().numpy())
                train_labels_vector.extend(label_noise.numpy())
        for i, (index, image, label) in enumerate(test_loader):
            image = image.to(self.device)
            with torch.no_grad():
                features = self.model(image)
            features = features.squeeze()
            features = features.detach()
            test_features_vector.extend(features.cpu().detach().numpy())
            test_labels_vector.extend(label.numpy())

        train_features_vector = np.array(train_features_vector)
        train_labels_vector = np.array(train_labels_vector)
        test_features_vector = np.array(test_features_vector)
        test_labels_vector = np.array(test_labels_vector)
        # print("Features shape {}".format(train_features_vector.shape))

        train_features = torch.utils.data.TensorDataset(
            torch.from_numpy(train_features_vector), torch.from_numpy(train_labels_vector)
        )

        train_features_loader = torch.utils.data.DataLoader(
            train_features, batch_size=self.classifier_stage_batch_size, shuffle=False
        )

        test_features = torch.utils.data.TensorDataset(
            torch.from_numpy(test_features_vector), torch.from_numpy(test_labels_vector)
        )

        test_features_loader = torch.utils.data.DataLoader(
            test_features, batch_size=self.classifier_stage_batch_size, shuffle=False
        )

        self.train_features_loader = train_features_loader
        self.test_features_loader = test_features_loader


    def test_classifier(self):
        testloader = self.test_features_loader
        self.model.eval()
        self.classifier.eval()
        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for i, (feature, label) in enumerate(testloader):
                label = label.view(-1)
                feature = feature.to(self.device)
                label = label.to(self.device)
                output = self.classifier(feature)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                test_num += label.shape[0]
        self.test_acc.append(test_acc / test_num)
        return test_acc, test_num

    def save_result(self):
        result_path = os.path.join(self.result_dir, "client_result" , str(self.id))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        ssl_path = os.path.join(result_path, str(self.id) + "_SSL.csv")
        ssl_df = pd.DataFrame({'train_loss': self.train_loss})
        ssl_df.to_csv(ssl_path, index=False)


    def prepare_next_stage(self):
        if self.args.encoder_network == 'resnet50':
            resnet = resnet50(num_classes=self.args.num_classes)
        elif self.args.encoder_network == 'resnet18':
            resnet = resnet18(num_classes=self.args.num_classes)
        else:
            raise NotImplementedError
        num_features = list(resnet.children())[-1].in_features
        resnet.fc = torch.nn.Identity()

        resnet.fc = self.model.online_encoder.fc

        resnet.load_state_dict(self.model.online_encoder.state_dict())

        self.model = resnet.to(self.device)

        self.model.fc = torch.nn.Identity()

        if self.args.classifier_use_MLP:
            self.classifier = MLP(num_features, self.args.num_classes, 4096)
            self.classifier = self.classifier.to(self.device)
        else:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(num_features, self.args.num_classes))
            self.classifier = self.classifier.to(self.device)



        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr= self.linear_stage_lr)

    def set_online_encoder_parameters(self, model):
        self.model.online_encoder = copy.deepcopy(model)
        self.model = self.model.to(self.device)

    def set_target_encoder_parameters(self, model):
        if self.model.target_encoder == None and self.args.SSL_method == 'byol':
            self.model.target_encoder = copy.deepcopy(self.model.online_encoder)
        # if self.model.target_encoder != None:
        #     for new_param, old_param in zip(model.parameters(), self.model.target_encoder.parameters()):
        #         old_param.data = new_param.data.clone()
        # else:
        #     self.model.target_encoder = copy.deepcopy(self.model.online_encoder)
        self.model = self.model.to(self.device)
    def set_online_predictor_parameters(self, model):
        self.model.online_predictor = copy.deepcopy(model)
        self.model = self.model.to(self.device)

def compute_lr(current_round, rounds=800, eta_min=0, eta_max=0.3):
    """Compute learning rate as cosine decay"""
    pi = np.pi
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (np.cos(pi * current_round / rounds) + 1)
    return eta_t
