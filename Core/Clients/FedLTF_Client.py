import copy
import time
import os
import pandas as pd
import numpy as np

import torch

from Core.Clients.Client_base import Client
from Core.utils.data_utils import get_dataloder,  Feature_dataset, count_client_data_by_category, get_class_num, get_dataset, count_samples_by_category_with_index, get_semi_dataloader


class FedLTF_Client(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)
        if args.SSL_method != None:
            self.SSL_method = args.SSL_method
        self.classifier = None
        self.classifier_optimizer = None
        self.train_classifier_loss = []
        self.train_classifier_acc = []
        self.test_acc = []

        self.train_features_loader = None
        self.test_features_loader = None

        self.semi_dataloader = None

        self.linear_stage_lr = args.linear_stage_lr
        self.classifier_stage_local_epochs = args.classifier_stage_local_epochs
        self.classifier_stage_batch_size = args.classifier_stage_batch_size

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

        train_features = Feature_dataset(
            torch.from_numpy(train_features_vector), torch.from_numpy(train_labels_vector)
        )

        train_features_loader = torch.utils.data.DataLoader(
            train_features, batch_size=self.classifier_stage_batch_size, shuffle=False
        )

        test_features = Feature_dataset(
            torch.from_numpy(test_features_vector), torch.from_numpy(test_labels_vector)
        )

        test_features_loader = torch.utils.data.DataLoader(
            test_features, batch_size=self.classifier_stage_batch_size, shuffle=False
        )

        self.train_features_loader = train_features_loader
        self.test_features_loader = test_features_loader


    def train_classifier(self):
        trainloader = self.train_features_loader
        self.model.eval()
        self.classifier.train()
        start_time = time.time()
        train_num = 0
        losses = 0
        train_acc = 0
        for step in range(self.classifier_stage_local_epochs):
            for i,(index, feature, label) in enumerate(trainloader):
                label = label.view(-1)
                feature = feature.to(self.device)
                label = label.to(self.device)
                output = self.classifier(feature)
                loss = torch.nn.functional.cross_entropy(output, label)
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
                train_num += label.shape[0]
                losses += loss.item() * label.shape[0]
                train_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
        self.train_classifier_loss.append(losses / train_num)
        self.train_classifier_acc.append(train_acc / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return losses,train_acc, train_num

    def test_classifier(self):
        testloader = self.test_features_loader
        self.model.eval()
        self.classifier.eval()
        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for i, (index, feature, label) in enumerate(testloader):
                label = label.view(-1)
                feature = feature.to(self.device)
                label = label.to(self.device)
                output = self.classifier(feature)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                test_num += label.shape[0]
        self.test_acc.append(test_acc / test_num)
        return test_acc, test_num

    def train_finetune(self):
        self.model = self.model.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.encoder_model.eval()
        self.linear_model.eval()
        trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)

        self.model.train()
        self.classifier.train()

        start_time = time.time()
        train_num = 0
        losses = 0
        train_acc = 0

        for step in range(self.classifier_stage_local_epochs):
            if self.noise_type in ['clean', 'real']:
                for i, (index_lb, image_lb, label_lb) in enumerate(trainloader):
                    label_lb = label_lb.view(-1)
                    image_lb = image_lb.to(self.device)
                    label_lb = label_lb.to(self.device)
                    feature = self.model(image_lb)
                    output = self.classifier(feature)

                    t_feature = self.encoder_model(image_lb)
                    t_output = self.linear_model(t_feature)

                    logits = torch.nn.functional.softmax(output, dim=1)
                    logits_t = torch.nn.functional.softmax(t_output, dim=1)

                    logits = torch.clamp(logits, 1e-7, 1 - 1e-7)
                    logits_t = torch.clamp(logits_t, 1e-7, 1 - 1e-7)

                    kl_distance = torch.nn.KLDivLoss(reduction='none')
                    variance = torch.sum(kl_distance(logits_t.log(), logits), dim=1)
                    exp_variance = torch.exp(-variance)

                    logits_temp = logits / 4
                    logits_t_temp = logits_t / 4

                    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(logits_temp.log(), logits_t_temp) * (4**2)

                    ce_loss = self.criterion(output, label_lb)
                    ce_loss = torch.mean(ce_loss * exp_variance) + torch.mean(variance)


                    loss = self.args.ce_s * ce_loss + self.args.kl_s * kl_loss

                    train_num += label_lb.shape[0]
                    losses += loss.item() * label_lb.shape[0]
                    train_acc += (torch.sum(torch.argmax(output, dim=1) == label_lb)).item()

                    self.classifier_optimizer.zero_grad()
                    loss.backward()
                    self.classifier_optimizer.step()
            else:
                for i, (index_lb, image_lb, label_lb, label_noise_lb) in enumerate(trainloader):
                    label_noise_lb = label_noise_lb.view(-1)
                    image_lb = image_lb.to(self.device)
                    label_noise_lb = label_noise_lb.to(self.device)
                    feature = self.model(image_lb)
                    output = self.classifier(feature)

                    t_feature = self.encoder_model(image_lb)
                    t_output = self.linear_model(t_feature)

                    logits = torch.nn.functional.softmax(output, dim=1)
                    logits_t = torch.nn.functional.softmax(t_output, dim=1)

                    logits = torch.clamp(logits, 1e-7, 1 - 1e-7)
                    logits_t = torch.clamp(logits_t, 1e-7, 1 - 1e-7)

                    kl_distance = torch.nn.KLDivLoss(reduction='none')
                    variance = torch.sum(kl_distance(logits_t.log(), logits), dim=1)
                    exp_variance = torch.exp(-variance)

                    logits_temp = logits / 4
                    logits_t_temp = logits_t / 4

                    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(logits_temp.log(), logits_t_temp) * (4**2)


                    ce_loss = self.criterion(output, label_noise_lb)
                    ce_loss = torch.mean(ce_loss * exp_variance) + torch.mean(variance)



                    loss = self.args.ce_s * ce_loss + self.args.kl_s * kl_loss


                    train_num += label_noise_lb.shape[0]
                    losses += loss.item() * label_noise_lb.shape[0]
                    train_acc += (torch.sum(torch.argmax(output, dim=1) == label_noise_lb)).item()
                    self.classifier_optimizer.zero_grad()
                    loss.backward()


                    self.classifier_optimizer.step()
        self.train_classifier_loss.append(losses / train_num)
        self.train_classifier_acc.append(train_acc / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        del trainloader
        torch.cuda.empty_cache()
        return losses,train_acc, train_num



    def test_finetune(self):
        testloader = get_dataloder(self.test_path, self.id, self.batch_size, False)
        self.model.eval()
        self.classifier.eval()
        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for i, (index, image, label) in enumerate(testloader):
                label = label.view(-1)
                label = label.to(self.device)
                image = image.to(self.device)
                feature = self.model(image)
                output = self.classifier(feature)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                test_num += label.shape[0]
        self.test_acc.append(test_acc / test_num)
        return test_acc, test_num

    def save_result(self):
        result_path = os.path.join(self.result_dir, "client_result" , str(self.id))
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        classifier_path = os.path.join(result_path, str(self.id) + "_classifier.csv")
        classifier_df = pd.DataFrame({'train_loss': self.train_classifier_loss, 'test_acc': self.test_acc})
        classifier_df.to_csv(classifier_path, index=False)

    def set_classifier_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.classifier.parameters()):
            old_param.data = new_param.data.clone()

    def prepare_next_stage(self,model, classifier):
        del self.model
        self.model = copy.deepcopy(model)
        self.classifier = copy.deepcopy(classifier).to(self.device)

        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr= self.args.linear_stage_lr)

    def prepare_next_stage_finetune(self, model, classifier):
        del self.model
        self.model = copy.deepcopy(model)
        self.classifier = copy.deepcopy(classifier)
        params = [
            {'params': self.model.parameters()},
            {'params': self.classifier.parameters()}
        ]
        del self.classifier_optimizer
        self.classifier_optimizer = torch.optim.Adam(params, lr= self.args.finetune_stage_lr)

        self.classifier = self.classifier.to(self.device)
        self.encoder_model = copy.deepcopy(model)
        self.linear_model = copy.deepcopy(classifier)
        self.encoder_model = self.encoder_model.to(self.device)
        self.linear_model = self.linear_model.to(self.device)

    def set_online_encoder_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.online_encoder.parameters()):
            old_param.data = new_param.data.clone()
        self.model = self.model.to(self.device)

    def set_target_encoder_parameters(self, model):
        if self.model.target_encoder == None and self.args.SSL_method == 'byol':
            self.model.target_encoder = copy.deepcopy(self.model.online_encoder)
        self.model = self.model.to(self.device)
    def set_online_predictor_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.online_predictor.parameters()):
            old_param.data = new_param.data.clone()
        self.model = self.model.to(self.device)


