import torch
from torch.optim import Optimizer


class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])


def get_optimizer(optimizer,parameters,learning_rate,momentum,weight_decay,args):
    """
    FL的优化器后面再说吧，找了别人实现的，现在还用不上，先用pytorch自带的优化器
    :param optimizer:
    :param parameters:
    :param learning_rate:
    :param momentum:
    :param weight_decay:
    :param args:
    :return: optimizer
    """


    if optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "adagrad":
        return torch.optim.Adagrad(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adadelta":
        return torch.optim.Adadelta(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adamax":
        return torch.optim.Adamax(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "asgd":
        return torch.optim.ASGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "lbfgs":
        return torch.optim.LBFGS(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer not found")