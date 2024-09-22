from Core.utils.global_setting import read_save_federated_args, setup_seed, get_model, get_server
from Core.Servers.Server_base import Server
from Core.utils.mem_utils import MemReporter
import time



if __name__ == "__main__":

    start_time = time.time()
    args = read_save_federated_args()
    setup_seed(args.seed)
    args.model = get_model(args)
    reporter = MemReporter()
    server = get_server(args)
    server.train()


    print("Time cost:{:.4f}s.".format(time.time() - start_time))
    print("All done!")
    reporter.report()
