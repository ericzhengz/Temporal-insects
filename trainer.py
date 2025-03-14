import sys
import os
import logging
import copy
import torch
import numpy as np

from utils.factory import get_model
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, tensor2numpy

def train(args):
    # We assume args["seed"] is a list of seeds.
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    # Set initial class count. If init_cls equals increment, we start from 0.
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]

    # Create log folder and set up logging.
    logs_name = os.path.join("logs", args["model_name"], args["dataset"], str(init_cls), str(args["increment"]))
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    logfilename = os.path.join(
        logs_name,
        f"{args['prefix'].strip()}_{args['seed']}_{args['backbone_type']}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    # DataManager for IIMinsects202.
    data_manager = DataManager(
        shuffle=args["shuffle"],
        seed=args["seed"],
        init_cls=args["init_cls"],
        increment=args["increment"],
        args=args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    # Create model from factory.
    model = get_model(args["model_name"], args)

    # Log some key hyperparameters from mos.json
    logging.info("Final Training Parameters:")
    logging.info("init_cls: {}".format(args["init_cls"]))
    logging.info("increment: {}".format(args["increment"]))
    logging.info("Adapter Momentum: {}".format(args["adapter_momentum"]))
    logging.info("CA Storage Efficient Method: {}".format(args["ca_storage_efficient_method"]))
    logging.info("CA LR: {}".format(args["ca_lr"]))
    logging.info("Batch Size: {}".format(args["batch_size"]))
    logging.info("Tuned Epoch: {}".format(args["tuned_epoch"]))

    cnn_curve = {"top1": [], "top5": []}
    nme_curve = {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    # Incremental learning loop.
    for task in range(data_manager.nb_tasks):
        logging.info("Task {} / {}".format(task+1, data_manager.nb_tasks))
        logging.info("Total parameters: {}".format(count_parameters(model._network)))
        logging.info("Trainable parameters: {}".format(count_parameters(model._network, trainable=True)))
        
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()  # Should return dicts with keys "top1", "top5", "grouped"
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN grouped accuracy: {}".format(cnn_accy["grouped"]))
            logging.info("NME grouped accuracy: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])
            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print("Task {}: CNN Top1: {}, Top5: {}".format(task, cnn_accy["top1"], cnn_accy["top5"]))
            print("Task {}: NME Top1: {}, Top5: {}".format(task, nme_accy["top1"], nme_accy["top5"]))
        else:
            logging.info("NME accuracy not available; using CNN only.")
            logging.info("CNN grouped accuracy: {}".format(cnn_accy["grouped"]))
            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)
            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
            print("Task {}: CNN Top1: {}, Top5: {}".format(task, cnn_accy["top1"], cnn_accy["top5"]))

    # Calculate forgetting if enabled.
    if args.get("print_forget", False):
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([data_manager.nb_tasks, data_manager.nb_tasks])
            for i, line in enumerate(cnn_matrix):
                np_acctable[i, :len(line)] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:-1])
            print("Accuracy Matrix (CNN):")
            print(np_acctable)
            logging.info("Forgetting (CNN): {}".format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([data_manager.nb_tasks, data_manager.nb_tasks])
            for i, line in enumerate(nme_matrix):
                np_acctable[i, :len(line)] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:-1])
            print("Accuracy Matrix (NME):")
            print(np_acctable)
            logging.info("Forgetting (NME): {}".format(forgetting))

    # Final summary output.
    final_cnn_avg = np.mean(cnn_curve["top1"])
    final_nme_avg = np.mean(nme_curve["top1"]) if nme_curve["top1"] else None
    print("Final Average CNN Top1 Accuracy: {:.2f}".format(final_cnn_avg))
    if final_nme_avg is not None:
        print("Final Average NME Top1 Accuracy: {:.2f}".format(final_nme_avg))
    logging.info("Final Average CNN Top1 Accuracy: {:.2f}".format(final_cnn_avg))
    if final_nme_avg is not None:
        logging.info("Final Average NME Top1 Accuracy: {:.2f}".format(final_nme_avg))

def _set_device(args):
    device_list = []
    for d in args["device"]:
        if d == -1:
            device_list.append(torch.device("cpu"))
        else:
            device_list.append(torch.device(f"cuda:{d}"))
    args["device"] = device_list

def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info(f"{key}: {value}")

if __name__ == "__main__":
    # Example args (these should be set in your mos.json and command-line arguments)
    args = {
        "model_name": "mos_modified",
        "dataset": "IIMinsects202",
        "dataset_root": "D:/IL_IIM/Temporal insects/data/IIMinsects202",  
        "meta_file": "D:/IL_IIM/Temporal insects/data/IIMinsects202/meta.json",
        "init_cls": 4,  # 修改为4
        "increment": 4,  # 修改为4
        "shuffle": True,
        "seed": [1993],
        "device": [0],
        "prefix": "insects_experiment",
        "backbone_type": "vit_base_patch16_224_mos",
        "print_forget": True,
        "tuned_epoch": 10,
        "init_lr": 0.03,
        "batch_size": 48,
        "weight_decay": 0.0005,
        "min_lr": 0,
        "optimizer": "sgd",
        "scheduler": "cosine",
        "reinit_optimizer": True,
        "init_milestones": [10],
        "init_lr_decay": 0.1,
        "reg": 0.1,
        "adapter_momentum": 0.1,
        "ensemble": True,
        "crct_epochs": 15,
        "ca_lr": 0.005,
        "ca_storage_efficient_method": "covariance",
        "n_centroids": 10,
        "pretrained": True,
        "drop": 0.0,
        "drop_path": 0.0,
        "ffn_num": 16,
        "memory_size": 100

    }
    train(args)
