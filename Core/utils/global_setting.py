import argparse
import os
import ujson
import random
import numpy as np
import torch
import torchvision
from Core.Networks.models import *
from Core.Networks.AlexNet import *
from Core.Networks.ResNet import *
from Core.Networks.MobileNet_V2 import *
from Core.Networks.SSL_model import get_SSL_model

from Core.Servers.FedAvg_server import FedAvg
from Core.Servers.FedSSL_server import FedSSL
from Core.Servers.FedLTF_server import FedLTF


def read_save_federated_args():
    parser = argparse.ArgumentParser(description="Federated setting")


    parser.add_argument(
        "--global_rounds",
        type=int,
        default=100,
        help="Number of rounds of global training."
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local epochs in each round."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn9l",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="FedAvg",
    )
    parser.add_argument(
        "--join_ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--random_join_ratio",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--eval_gap",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--just_eval_global_model",
        type=bool,
        default=False,
        help="Just evaluate global model in a large test_dataset."
    )
    parser.add_argument(
        "--client_drop_rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--warm_up_steps",
        type=int,
        default=0,
        help="Number of warm up steps for clients before training."
    )
    parser.add_argument(
        "--centralized",
        default=False,
        help="Centralized setting or federated setting. True for centralized "
             "setting, while False for federated setting.",
    )
    # ----Federated Partition----
    parser.add_argument(
        "--partition",
        default="iid",
        type=str,
        choices=["iid", "dir", "pat"],
        help="Data partition scheme for federated setting.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="All clients have the same number of data.",
    )
    parser.add_argument(
        "--num_clients",
        default=20,
        type=int,
        help="Number for clients in federated setting.",
    )
    parser.add_argument(
        "--dir_alpha",
        default=0.1,
        type=float,
        help="Parameter for Dirichlet distribution.",
    )
    parser.add_argument(
        "--class_per_client",
        default=2,
        type=int,
        help="class_per_client number for 'pat' partition.",
    )
    parser.add_argument(
        "--max_samples",
        default=64000,
        help="max_samples sample in one dataset(e.g. clothing1M).",
    )
    parser.add_argument(
        "--least_samples",
        default=25,
        type=int,
        help="least_samples for each client each class.",
    )

    # ----Noise setting options----
    parser.add_argument(
        "--globalize",
        default=True,
        help="Federated noisy label setting, globalized noise or localized noise.",
    )
    parser.add_argument(
        "--noise_type",
        default="sym",
        type=str,
        choices=["clean", "sym", "asym", "real"],
        help="Noise type for centralized setting: 'sym' for symmetric noise; "
             "'asym' for asymmetric noise; 'real' for real-world noise. Only works "
             "if --centralized=True.",
    )

    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="Noise ratio for symmetric noise or asymmetric noise.",
    )

    parser.add_argument(
        "--min_noise_ratio",
        default=0.0,
        type=float,
        help="Minimum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )
    parser.add_argument(
        "--max_noise_ratio",
        default=1.0,
        type=float,
        help="Maximum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )

    # ----Dataset path options----
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["mnist", "fmnist", "cifar10", "cifar100", "clothing1M", "tinyimagenet"],
        help="Dataset for experiment. Current support: ['mnist', 'cifar10', 'cifar100`, 'clothing1m', 'tinyimagenet']",
    )
    parser.add_argument(
        "--data_dir",
        default="./Datasets",
        type=str,
        help="Directory for dataset.",
    )
    parser.add_argument(
        "--result_dir",
        default="./Results",
        type=str,
        help="Directory for results.",
    )
    # ------------------------------------------------------------------------criterion setting_________________________________________________________________________
    parser.add_argument(
        "--criterion",
        type=str,
        default="ce",
    )
    parser.add_argument(
        "--sce_alpha",
        type=float,
        default=0.1,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--sce_beta",
        type=float,
        default=1.0,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=1.0,
        help="scale parameter for loss, for example, scale * RCE, scale * NCE, scale * normalizer * RCE.",
    )
    parser.add_argument(
        "--gce_q",
        type=float,
        default=0.7,
        help="q parametor for Generalized-Cross-Entropy, Normalized-Generalized-Cross-Entropy.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=None,
        help="alpha parameter for Focal loss and Normalzied Focal loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="gamma parameter for Focal loss and Normalzied Focal loss.",
    )
    parser.add_argument(
        "--elr_beta",
        type=float,
        default=0.1,
        help="beta parameter for ELR",
    )
    parser.add_argument(
        "--elr_lamb",
        type=float,
        default=2,
        help="lamb parameter for ELR",
    )




    # ----Miscs options----
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--resume",
        default=False,
        type=bool,
        help="Resume from previous checkpoint.",
    )
    parser.add_argument(
        "--tensorboard",
        default=True,
        type=bool,
        help="Use tensorboard to record training process.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        choices=["cuda", "cpu"],
        help="Device for training.",
    )
    parser.add_argument(
        "--device_id",
        default="0",
        type=str,
        help="GPU device id for training.",
    )

    parser.add_argument(
        "--goal",
        default="test",
        type=str,
        help="goal for this simulation.",
    )

    parser.add_argument(
        "--plot",
        default=True,
        type=bool,
        help="plot result or not.",
    )

    ###########################################################SSL
    parser.add_argument(
        "--SSL_method",
        default=None,
        type=str,
        help="transform_method for SSL.",
    )

    parser.add_argument(
        "--encoder_network",
        default="resnet18",
        type=str,
        help="encoder_network for SSL.",
    )

    parser.add_argument(
        "--aggregate_online_encoder",
        default=True,
        type=bool,
        help="aggregate online_encoder or not.",
    )

    parser.add_argument(
        "--aggregate_online_predictor",
        default=True,
        type=bool,
        help="aggregate online_predictor or not.",
    )

    parser.add_argument(
        "--aggregate_target_encoder",
        default=False,
        type=bool,
        help="aggregate target_encoder or not.",
    )

    parser.add_argument(
        "--linear_stage_epochs",
        default=30,
        type=int,

    )

    parser.add_argument(
        "--finetune_stage_epochs",
        default=100,
        type=int,
        help="classifier_stage_epochs.",
    )

    parser.add_argument(
        "--linear_stage_lr",
        default=1e-3,
        type=float,
        help="classifier_stage_lr.",
    )

    parser.add_argument(
        "--finetune_stage_lr",
        default=1e-3,
        type=float,
        help="classifier_stage_lr.",
    )

    parser.add_argument(
        "--classifier_stage_local_epochs",
        default=1,
        type=int,
        help="classifier_stage_local_epochs.",
    )

    parser.add_argument(
        "--classifier_stage_batch_size",
        default=512,
        type=int,
        help="classifier_stage_batch_size.",
    )


    parser.add_argument(
        "--encoder_model_path",
        default=None,
        type=str,
        help="encoder_model_path.",
    )

    parser.add_argument(
        "--classifier_use_MLP",
        default = False,
        type = bool,
        help = "classifier_use_MLP.",
    )

    parser.add_argument(
        "--lr_type",
        default="cosine",
        type=str,
        help='cosine decay learning rate'
    )

    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--match_epoch",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--dis_metric",
        default="creff",
        type=str,
        help='distance metric'
    )

    parser.add_argument(
        "--num_of_feature",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--lr_feature",
        default=0.1,
        type=float,
        help='learning rate for updating synthetic images'
    )

    parser.add_argument(
        "--crt_epoch",
        default=300,
        type=int,
    )

    parser.add_argument(
        "--confidence_thres",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--relabel_start",
        default=80,
        type=int,
        help = "start epoch for relabel"
    )

    parser.add_argument(
        "--ce_s",
        default=1,
        type=float,
    )

    parser.add_argument(
        "--kl_s",
        default=1,
        type=float,
    )

    parser.add_argument(
        "--temperature",
        default=4,
        type=int,
    )

    # parser.add_argument(
    #     "--ablation",
    #     default=False,
    #     type=bool,
    # )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    if args.device != "cpu":
        args.device = args.device + ":" + str(args.gpu)


    centralized = "centralized" if args.centralized else "federated"
    globalize = "global" if args.globalize else "local"
    if args.balance:
        balanced = "balanced"
    else:
        balanced = "imbalance"
    unique = args.model + "_" + str(args.num_clients) + "_" +str(args.partition) + "_" + balanced + "_" + str(args.noise_type) + "_" + str(args.noise_ratio)+"_lr" + str(args.learning_rate)+"_bs" + str(args.batch_size)+"_ep" + str(args.local_epochs)+"_loss"+ str(args.criterion)+ "_" + args.goal
    if args.criterion == "elr":
        unique = unique + "_beta" + str(args.elr_beta) + "_lamb" + str(args.elr_lamb)
    args.result_dir = os.path.join(args.result_dir,str(args.algorithm) + "_" + str(args.dataset) + "_" + centralized + "_" + globalize,unique)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    config_path = os.path.join(args.result_dir, "config.json")
    with open(config_path, "w") as f:
        ujson.dump(args.__dict__, f, indent=2)


    return args


def setup_seed(seed: int = 0):
    """
    Args:
        seed (int): random seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(args):
    if "mnist" in args.dataset:
        input_channel = 1
    else:
        input_channel = 3
    num_classes = {'mnist': 10, 'fmnist': 10, 'cifar10': 10, 'cifar100': 100, 'clothing1M': 14,
                   'tinyimagenet': 200}
    output_channel = num_classes[args.dataset]
    if args.SSL_method != None:
        # return get_SSL_model(args.SSL_method, args.model, output_channel).to(args.device)
        args.num_classes = output_channel
        return get_SSL_model(args).to(args.device)
    else:
        if args.model == "cnn9l":
            return CNN_9layer(input_channel, output_channel).to(args.device)
        elif args.model == "resnet18":
            return resnet18(num_classes=output_channel, input_channel=input_channel).to(args.device)
        elif args.model == "resnet50":
            base_model = resnet50(input_channel=input_channel)
            base_model.fc = nn.Linear(2048, output_channel)
            base_model = base_model.to(args.device)
            return base_model
        else:
            raise NotImplementedError("Model {} is not implemented.".format(args.model))

def get_server(args):
    if args.algorithm == "FedAvg":
        server = FedAvg(args)
    elif args.algorithm == "FedSSL":
        server = FedSSL(args)
    elif args.algorithm == "FedLTF":
        server = FedLTF(args)
    else:
        raise NotImplementedError("Algorithm {} is not implemented.".format(args.algorithm))
    return server