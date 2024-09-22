import time
from Core.Clients.Client_base import Client

class FedAvgClient(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)
