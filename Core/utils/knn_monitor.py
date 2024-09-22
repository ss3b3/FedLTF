import torch
import torch.nn.functional as F
from tqdm import tqdm


# code is obtained from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=lzFyFnhbk8hj
# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader,num_classes,noise_type, k=200, t=0.1, hide_progress=False, device=None):
    net.eval()
    classes = num_classes
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        if noise_type in ['clean','real']:
            for index, data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
                if device is None:
                    data = data.cuda(non_blocking=True)
                else:
                    data = data.to(device, non_blocking=True)
                feature = net(data)
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
        else:
            for index, data, target, _ in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
                if device is None:
                    data = data.cuda(non_blocking=True)
                else:
                    data = data.to(device, non_blocking=True)
                feature = net(data)
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = memory_data_loader.dataset.label.clone().detach().to(device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for index, data, target in test_bar:
            if device is None:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy': total_top1 / total_num * 100})
        print("Accuracy: {}".format(total_top1 / total_num * 100))
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
