import os.path
import time
from torch.utils.tensorboard import SummaryWriter
from Core.Servers.Server_base import Server
from Core.Clients.FedAvg_client import FedAvgClient


class FedAvg(Server):

    def __init__(self, args):
        super().__init__(args)

        self.set_clients(FedAvgClient)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        if self.resume:
            self.load_model()

        self.Budget = []

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
            if self.warm_up_steps > 0:
                print(f"\n-------------Warm up-------------")
                for client in self.selected_clients:
                    client.warm_up_train()
                self.receive_models()
                self.aggregate_parameters()
                self.warm_up_steps = 0
                print(f"\n-------------Warm up end-------------")

            averaged_loss = self.local_train()
            averaged_acc = self.local_eval()

            if self.just_eval_global_model:
                averaged_acc = self.global_eval()


            self.rs_train_loss.append(averaged_loss)
            self.rs_test_acc.append(averaged_acc)
            if self.tensorboard:
                writer.add_scalar('train_loss', averaged_loss, i)
                writer.add_scalar('train_acc', averaged_acc, i)
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                print("Averaged Train loss:{:.4f}".format(averaged_loss))
                print("Averaged Test acc:{:.4f}".format(averaged_acc))

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            # if self.early_stop():
            #     break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()