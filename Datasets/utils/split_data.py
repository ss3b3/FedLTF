

import numpy as np
import torch
from torch.utils.data import Subset
import random


def separate_data(dataset, num_clients, num_classes, balance=True, partition='iid', class_per_client=2,max_samples=None,least_samples = 25, alpha =0.1,seed=1, split_proportions = None):

    random.seed(seed)
    np.random.seed(seed)
    proportions_save = []
    if type(dataset.targets) == list:
        targets = np.array(dataset.targets)
    elif type(dataset.targets) == torch.Tensor:
        targets = dataset.targets.numpy()
    else:
        targets = dataset.targets
    N = len(dataset.targets)

    if max_samples != None:
        per_class_sample_per_client = max_samples // num_classes // num_clients
        N = max_samples
    else:
        per_class_sample_per_client = len(dataset) // num_classes // num_clients

    if partition == 'iid':
        splits = [[] for _ in range(num_clients)]
        if not balance:
            ratios = np.random.dirichlet(np.ones(20), size=1)[0]
        for i in range(num_classes):
            idxs = np.where(targets == i)[0]
            final_num_per_class_per_client = min(int(len(idxs) / num_clients), per_class_sample_per_client)
            selected_idxs = np.random.choice(idxs, final_num_per_class_per_client * num_clients, replace=False)
            rand_idxs = selected_idxs.copy()
            np.random.shuffle(rand_idxs)
            if balance:
                for client in range(num_clients):
                    split = np.array_split(rand_idxs, num_clients)
                    splits[client].append(split[client])
            else:
                counts = (ratios * len(rand_idxs)).astype(int)
                counts[-1] = len(rand_idxs) - np.sum(counts[:-1])
                split_points = np.cumsum(counts)[:-1]
                for client in range(num_clients):
                    split = np.split(rand_idxs, split_points)
                    splits[client].append(split[client])
        return [Subset(dataset, np.concatenate(splits[client])) for client in range(num_clients)], proportions_save

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        idx_batch = [[] for _ in range(num_clients)]
        proportions_save = []
        for i in range(num_classes):
            idxs = np.where(targets == i)[0]
            final_num_per_class_per_client = min(int(len(idxs) / num_clients), per_class_sample_per_client)
            selected_idxs = np.random.choice(idxs, final_num_per_class_per_client * num_clients, replace=False)
            rand_idxs = selected_idxs.copy()
            np.random.shuffle(rand_idxs)
            if split_proportions != None:
                proportions = split_proportions[i]
            else:
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions_save.append(proportions)
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(rand_idxs)).astype(int)[:-1]
            if max_samples != None:
                # 计算每个客户端的样本数量
                num_samples = np.diff(np.hstack(([0], proportions, [len(rand_idxs)])))
                # 检查每个客户端的样本数量是否超过最大值
                num_samples = np.minimum(num_samples, max_samples // num_clients)
                # 重新计算比例
                proportions = np.cumsum(num_samples).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(rand_idxs, proportions))]

        return [Subset(dataset, idx_batch[client]) for client in range(num_clients)], proportions_save

    # elif partition == "pat":
    #
    #     if int(num_clients / num_classes * class_per_client) == 0:
    #         raise ValueError("Need split more clients for "+ str(num_classes) + " classes. Or bigger class_per_client")
    #
    #     dataidx_map = {}
    #     idx_for_each_class = []
    #     class_num_per_client = [class_per_client for _ in range(num_clients)]
    #
    #     for i in range(num_classes):
    #         idxs = np.where(targets == i)[0]
    #         final_num_per_class_per_client = min(int(len(idxs) / num_clients), per_class_sample_per_client)
    #         selected_idxs = np.random.choice(idxs, final_num_per_class_per_client * num_clients, replace=False)
    #         rand_idxs = selected_idxs.copy()
    #         np.random.shuffle(rand_idxs)
    #         idx_for_each_class.append(rand_idxs)
    #
    #     for i in range(num_classes):
    #         selected_clients = []
    #         for client in range(num_clients):
    #             if class_num_per_client[client] > 0:
    #                 selected_clients.append(client)
    #
    #             selected_clients = selected_clients[:int(num_clients / num_classes * class_per_client)]
    #
    #         num_all_samples = len(idx_for_each_class[i])
    #         num_selected_clients = len(selected_clients)
    #         num_per = num_all_samples / num_selected_clients
    #         if balance:
    #             num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
    #         else:
    #             num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
    #                                             num_selected_clients - 1).tolist()
    #         num_samples.append(num_all_samples - sum(num_samples))
    #         if max_samples != None:
    #             num_samples = [min(max_samples, num) for num in num_samples]
    #
    #         idx = 0
    #         for client, num_sample in zip(selected_clients, num_samples):
    #             if client not in dataidx_map.keys():
    #                 dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
    #             else:
    #                 dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
    #                                                 axis=0)
    #             idx += num_sample
    #             class_num_per_client[client] -= 1
    #     return [Subset(dataset, dataidx_map[client]) for client in range(num_clients)], proportions_save

    else:
        raise NotImplementedError

