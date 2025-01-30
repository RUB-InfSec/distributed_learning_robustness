import argparse
import copy
import datetime
import logging
import os
import random
import tempfile
from functools import partial

import numpy as np
import ray.train as ray_train
import torch
import torch.nn as nn
import torchvision
from ray import tune
from ray.air import CheckpointConfig
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from tabulate import tabulate, TableFormat, Line, DataRow
from torch.utils.data import random_split

from utils import nutils

data_dir = os.path.abspath("data/cifar10")
rootLogger = logging.getLogger()


def get_params(config):
    optim = config["optim"]
    arch = config["arch"]
    sched = config["sched"]
    sched_param = config["scheduler_param"]
    datasplit = config["datasplit"]
    dirichlet_alpha = config["dirichlet_alpha"]
    dataset_seed = config["dataset_seed"]
    batch_size = config["batch_size"]

    optim_params = nutils.OptimizerParameters(optimizer_name=optim, json=
    {
        "optimizer_name": optim,
        "lr": config["lr"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
    })

    if sched is not None:
        sched_params = nutils.SchedulerParameters(scheduler_name=sched, json=
        {
            "scheduler_name": sched,
            "scheduler_param": sched_param,
        })
    else:
        sched_params = None

    params = nutils.DecentralizedParameters(datasplit=datasplit, json=
    {
        "architecture": arch,
        "weights_quantization": False,
        "network_pruning": False,
        "optimizer": optim_params,
        "scheduler": sched_params,
        "epochs": config["epochs"],
        "dropout": config["dropout"],
        "dirichlet_alpha": dirichlet_alpha,
        "nodeID": config["node_id"],
        "totalN": config["node_count"]
    }, dataset_seed=dataset_seed, batch_size=batch_size)

    return params


def train_model(config, args, data_dir, final_training=False):
    fix_randomness(config["trial_seed"])

    params = get_params(config)

    net = nutils.get_model(dataset=args.dataset, arch=params.architecture)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = params.get_optimizer(net.parameters())

    trainset, testset = nutils.get_dataset(dataset=args.dataset, train=True, path=data_dir, params=params,
                                           download=False), nutils.get_dataset(dataset=args.dataset, train=False,
                                                                               path=data_dir, download=False)

    test_abs = int(len(trainset) * 0.8)

    if params.scheduler is not None and params.scheduler.scheduler_name == "CosineAnnealingLR":
        params.scheduler.scheduler_param = params.scheduler.scheduler_param * (test_abs / params.batch_size)

    # ensure we have always the same 80/20 split
    generator1 = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs], generator=generator1
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=params.batch_size, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=params.batch_size, shuffle=True, num_workers=2
    )

    scheduler = params.get_scheduler(optimizer)

    for epoch in range(0, params.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0

        net.train()

        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1

            if scheduler is not None and params.scheduler.scheduler_name == "CosineAnnealingLR":
                scheduler.step()

        if scheduler is not None and params.scheduler.scheduler_name != "CosineAnnealingLR":
            if params.scheduler.scheduler_name == "ReduceLROnPlateau":
                scheduler.step(running_loss / epoch_steps)
            else:
                scheduler.step()

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        net.eval()

        for inputs, labels in valloader:
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        if final_training and (epoch + 1) % 10 == 0:
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(
                    net.state_dict(),
                    os.path.join(checkpoint_dir, "model.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(checkpoint_dir, "optimizer.pt"),
                )
                torch.save(
                    {"epoch": (epoch + 1)},
                    os.path.join(checkpoint_dir, "extra_state.pt"),
                )

                checkpoint = Checkpoint.from_directory(checkpoint_dir)

                ray_train.report(
                    {"loss": val_loss / val_steps, "accuracy": correct / total},
                    checkpoint=checkpoint,
                )
        elif not final_training:
            checkpoint = None
            ray_train.report(
                {"loss": val_loss / val_steps, "accuracy": correct / total},
                checkpoint=checkpoint,
            )

    print("Finished training.")


def accuracy(net, data_dir, args, testloader=None, device="cuda"):
    if testloader is None:
        testloader = nutils.get_loader(dataset=args.dataset, train=False, path=data_dir, download=False)

    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def fine_tuning(ray_config, args, num_samples=100, gpus_per_trial=0.5, final_training=False, name='experiment'):
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=ray_config["epochs"],
        grace_period=3,
        reduction_factor=2,
    )

    # decide whether the redrawn trials worked better
    metric, mode, scope = 'loss', 'min', 'all'

    # start the Ray tuning of the specified training, but limit resources, time budget and number of kept models to reduce the runtime and memory footprint
    result = tune.run(
        partial(train_model, args=args, data_dir=data_dir, final_training=final_training),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        time_budget_s=datetime.timedelta(hours=3),
        config=ray_config,
        checkpoint_config=CheckpointConfig(num_to_keep=3, checkpoint_score_attribute=metric,
                                           checkpoint_score_order=mode),
        storage_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ray_models"),
        num_samples=num_samples,
        scheduler=scheduler,
        log_to_file=True,
        verbose=0,
        fail_fast=True,
        name=name
    )

    best_trial = result.get_best_trial(metric, mode, scope)

    # load and return best fully trained model
    if final_training:
        # Gets best checkpoint for trial based on loss.
        best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric=metric, mode=mode)

        best_trained_model = nutils.get_model(dataset=args.dataset, arch=ray_config["arch"])

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)

        with best_checkpoint.as_directory() as checkpoint_dir:
            file = os.path.join(checkpoint_dir, 'model.pt')
            if os.path.getsize(file) > 0:
                print('Best checkpoint found at {}'.format(os.path.join(checkpoint_dir, "model.pt")))

                model_state_dict = torch.load(
                    os.path.join(checkpoint_dir, "model.pt"),
                )
                best_trained_model.load_state_dict(model_state_dict)

                test_acc = accuracy(net=best_trained_model, args=args, data_dir=data_dir, device=device)
                print("Best trial test set accuracy: {}".format(test_acc))
            else:
                raise FileNotFoundError("No checkpoint found even though there should be one :(")

        return best_checkpoint.path, result.get_best_config(metric=metric, mode=mode, scope=scope), test_acc

    return None, result.get_best_config(metric=metric, mode=mode, scope=scope), None


def finetune(args, seed, config_dictionary):
    train_mode = args.train_mode

    # specifies the number of trials used in the Ray optimization phase
    num_samples = 100

    # train mode specifies if a specific training phase should be executed; if not set all phases will be executed.
    if train_mode is None:
        fix_randomness(seed)

        print("Starting OPTIM-1...")
        best_config_1 = mini_optim1(args, num_samples=num_samples)
        pretty_print_config(args, best_config_1, "OPTIM-1")

        fix_randomness(7423 * seed)
        print("Starting OPTIM-2...")
        best_config_2 = mini_optim2(args, config=copy.deepcopy(best_config_1), num_samples=num_samples)
        pretty_print_config(args, best_config_2, "OPTIM-2")

        fix_randomness(3929 * seed)
        print("Starting FINAL...")
        path, best_config_final, test_acc = final_training(args, copy.deepcopy(best_config_2))

        pretty_print_config(args, best_config_final, "FINAL")
        return path, best_config_final, test_acc

    else:
        best_config = copy.deepcopy(config_dictionary)
        if train_mode == 1:
            fix_randomness(seed)
            print("Starting OPTIM-1...")
            best_config_1 = mini_optim1(args, num_samples=num_samples)
            pretty_print_config(args, best_config_1, "OPTIM-1")
        elif train_mode == 2:
            fix_randomness(7423 * seed)
            print("Starting OPTIM-2...")
            best_config_2 = mini_optim2(args, config=best_config, num_samples=num_samples)
            pretty_print_config(args, best_config_2, "OPTIM-2")
        elif train_mode == 3:
            fix_randomness(3929 * seed)
            print("Starting FINAL...")
            path, best_config_final, test_acc = final_training(args, best_config)
            pretty_print_config(args, best_config_final, "FINAL")
            return path, best_config_final, test_acc


def mini_optim1(args, num_samples=100):
    arch = args.architecture
    optim = args.optimizer
    datasplit = args.data_split
    dirichlet_alpha = args.dirichlet_alpha

    node_id = args.node_id
    node_count = args.node_count
    dataset_seed = (args.round + 1) * 10 + (node_count + 1) * 100

    if optim in nutils.possibleMomentumOptimizers:
        momentums = tune.uniform(0.5, 1.0)
    else:
        momentums = 0.0

    config = {
        "trial_seed": tune.randint(0, 10000),
        "arch": arch,
        "optim": optim,
        "momentum": momentums,
        "sched": None,
        "scheduler_param": None,
        "epochs": 25,
        "weight_decay": 0.0,
        "datasplit": datasplit,
        "dirichlet_alpha": dirichlet_alpha,
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout": 0.0,
        "node_id": node_id,
        "node_count": node_count,
        "dataset_seed": dataset_seed,
        "batch_size": 256
    }

    _, best_config, _ = fine_tuning(config, args=args, num_samples=num_samples, gpus_per_trial=1 / args.tasks_per_gpu,
                                    name='{}_{}_{}_mini_optim_1'.format(args.output.split(',')[1],
                                                                        args.output.split(',')[0], args.dataset))

    config["lr"] = best_config['lr']
    config["momentum"] = best_config['momentum']
    config["trial_seed"] = best_config['trial_seed']

    return config


def mini_optim2(args, config, num_samples=100):
    config["trial_seed"] = tune.randint(0, 10000)

    sched = args.scheduler

    if sched == "CosineAnnealingLR":
        sched_params = tune.uniform(1, 50)
    elif sched == "StepLR":
        sched_params = tune.uniform(10, 30)
    elif sched == "ExponentialLR":
        sched_params = tune.loguniform(1 - 1e-1, 1 - 1e-4)
    elif sched == "ReduceLROnPlateau":
        sched_params = None
    elif sched == "CyclicLR":
        sched_params = None
    elif sched is None:
        sched_params = None
    else:
        raise ValueError("Undefined scheduler.")

    dropout = 0.0

    config["sched"] = sched
    config["scheduler_param"] = sched_params
    config["epochs"] = 50
    config["dropout"] = dropout

    if config["optim"] not in nutils.possibleWeightDecayOptimizers and sched_params is None and dropout == 0.0:
        # skip step 2
        config["weight_decay"] = 0.0
        print("No weight decay optimizer, no adjustable scheduler, and no dropout. Skipping fine-tuning phase 2.")
    else:
        if config["optim"] in nutils.possibleWeightDecayOptimizers:
            config['weight_decay'] = tune.loguniform(1e-5, 1e-3)

        _, best_config, _ = fine_tuning(config, args=args, num_samples=num_samples,
                                        gpus_per_trial=1 / args.tasks_per_gpu,
                                        name='{}_{}_{}_mini_optim_2'.format(args.output.split(',')[1],
                                                                            args.output.split(',')[0], args.dataset))
        config["weight_decay"] = best_config['weight_decay']
        config["scheduler_param"] = best_config['scheduler_param']
        config["dropout"] = best_config['dropout']

        config["trial_seed"] = best_config['trial_seed']

    return config


def final_training(args, config):
    config["trial_seed"] = tune.randint(0, 10000)

    if args.dataset == "cifar10":
        config["epochs"] = 200

    config["batch_size"] = 256

    path, best_config, test_acc = fine_tuning(config, args=args, num_samples=1, gpus_per_trial=1.0, final_training=True,
                                              name='{}_{}_{}_final_training'.format(args.output.split(',')[1],
                                                                                    args.output.split(',')[0],
                                                                                    args.dataset))

    config["trial_seed"] = best_config['trial_seed']

    return path, best_config, test_acc


def parse_args(args_string):
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", "-a", default=None)
    parser.add_argument("--optimizer", "-o", default=None)
    parser.add_argument("--scheduler", "-s", default=None)
    parser.add_argument("--data_split", "-ds", default=None)
    parser.add_argument("--dirichlet_alpha", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--tasks_per_gpu", type=int, default=2)
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--node_id", type=int, default=0)
    parser.add_argument("--node_count", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--train_mode", type=int, default=None)

    if args_string is not None:
        return parser.parse_args(args_string.split())
    else:
        return parser.parse_args()


def fix_randomness(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretty_print_config(args, config, header):
    FORMATT = TableFormat(
        lineabove=Line("╭", "─", "─", "╮"),
        linebelowheader=Line("├", "─", "─", "┤"),
        linebetweenrows=None,
        linebelow=Line("╰", "─", "─", "╯"),
        headerrow=DataRow("│", " ", "│"),
        datarow=DataRow("│", " ", "│"),
        padding=1,
        with_header_hide=None,
    )

    print(
        tabulate([["start_time", args.output.split(",")[1]]] + [[en, config[en]] for en in config],
                 headers=["Task / Type", f"{args.output.split(',')[0]} / {header}"],
                 tablefmt=FORMATT,
                 )
    )


def main(args_string, config_dictionary=None):
    args = parse_args(args_string)

    if args.train_mode == 2 or args.train_mode == 3:
        assert config_dictionary is not None
        config_dictionary["node_id"] = args.node_id
        config_dictionary["node_count"] = args.node_count
        config_dictionary["dataset_seed"] = (args.round + 1) * 10 + (args.node_count + 1) * 100
    else:
        assert args.architecture in nutils.possibleArchitectures[args.dataset]
        assert args.optimizer in nutils.possibleOptimizers
        assert args.scheduler in nutils.possibleSchedulers or args.scheduler is None
        assert args.data_split in nutils.possibleDataSplit
        if args.data_split == "dirichlet":
            assert args.dirichlet_alpha is not None

    seed = (args.round + 1) * 10 + (args.node_count + 1) * 100 + (args.node_id + 1) * 1000 + (
            args.iteration + 1) * 10000

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == "cifar10":
        # load the data once to prevent it being done in every worker
        if not os.path.exists("datasets/cifar10/cifar-10-batches-py"):
            torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
    else:
        raise NotImplementedError("Unknown dataset.")

    path, config, test_acc = finetune(args, seed, config_dictionary)

    return path, config, test_acc
