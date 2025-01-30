import argparse
import copy
import glob
import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime

import ray
from ray.tune import TuneError

from utils import nutils
import train


def experiment(flavor, args):

    rootLogger.warning(f"--- {flavor} ---")

    # finds converging hyperparameter configurations, i.e., master_configs, which is the basis for all heterogeneity experiments.
    find_master = flavor == "find_master"

    for data_split in ["uniform", "dirichlet"]:
        # a non-IID data split is only evaluated for ensemble_diverse
        if data_split == "dirichlet" and not flavor == "ensemble_diverse":
            continue

        if args.rounds is None:
            rounds = [0, 1, 2, 3, 4]
        else:
            rounds = [int(e) if e.isdigit() else e for e in args.rounds.split(',')]

        if find_master:
            possible_architectures = nutils.possibleArchitectures[args.dataset].copy()
            possible_optimizers = nutils.possibleOptimizers.copy()
            possible_schedulers = nutils.possibleSchedulers.copy()

            possible_momentum_optimizers = nutils.possibleMomentumOptimizers.copy()
            possible_momentum_schedulers = nutils.possibleMomentumSchedulers.copy()

        for round in rounds:
            if not find_master:
                possible_architectures = nutils.possibleArchitectures[args.dataset]
                possible_optimizers = nutils.possibleOptimizers
                possible_schedulers = nutils.possibleSchedulers

                possible_momentum_optimizers = nutils.possibleMomentumOptimizers
                possible_momentum_schedulers = nutils.possibleMomentumSchedulers

            random.seed(round)

            # shake things up
            random.shuffle(possible_architectures)
            random.shuffle(possible_optimizers)
            random.shuffle(possible_schedulers)

            if args.node_counts is None:
                node_counts = [1, 3, 5, 7]
            else:
                node_counts = [1] + [int(e) if e.isdigit() else e for e in args.node_counts.split(',')]

            for node_count in node_counts:
                if node_count == 1:
                    if find_master:
                        master_config = None
                        iteration = 0
                        node_id = 0

                        # find best parameters for only one model
                        test_acc_threshold = 0.89
                        success = False

                        if success:
                            break

                        while not success:
                            seed = (round + 1) * 10 + (node_count + 1) * 100 + (node_id + 1) * 1000 + (
                                    iteration + 1) * 10000

                            arch = possible_architectures[iteration % len(possible_architectures)]
                            sched = possible_schedulers[iteration % len(possible_schedulers)]

                            if sched in possible_momentum_schedulers:
                                optim = possible_momentum_optimizers[iteration % len(possible_momentum_optimizers)]
                            else:
                                optim = possible_optimizers[iteration % len(possible_optimizers)]

                            args_string = f"--output {'{}_{},'.format(flavor, data_split) + datetime.now().isoformat()} --dataset {args.dataset} --tasks_per_gpu {args.tasks_per_gpu} --iteration {iteration} --round {round} --node_id {node_id} --node_count {node_count} --architecture {arch} --optimizer {optim} --scheduler {sched} --data_split {data_split}"
                            if data_split == "dirichlet":
                                args_string = args_string + " --dirichlet_alpha 0.9"

                            if args.verbose:
                                print(args_string)

                            master_test_acc = 0.0

                            try:
                                master_path, master_config, master_test_acc = train.main(args_string,
                                                                                         config_dictionary=None)

                                rootLogger.warning(
                                    {"time": time.asctime(), "round": round, "seed": seed, "config": master_config,
                                     "path": master_path,
                                     "test_acc": master_test_acc, "converging_threshold": test_acc_threshold})
                            except TuneError as e:
                                rootLogger.warning(e)
                                rootLogger.warning(
                                    {"status": "ERROR", "time": time.asctime(), "round": round, "seed": seed,
                                     "args": args_string})

                            if master_test_acc > test_acc_threshold:
                                rootLogger.warning(f"Found converging config in it. {iteration}.")
                                success = True

                                # remove successful combination
                                possible_architectures.remove(arch)
                                possible_optimizers.remove(optim)

                                if optim in possible_momentum_optimizers:
                                    possible_momentum_optimizers.remove(optim)

                                possible_schedulers.remove(sched)
                            else:
                                iteration = iteration + 1
                    else:
                        # load the converging configurations and use them for the subsequent hyperparameter variations
                        with open(f"utils/master_configs.json") as file:
                            data = json.loads(file.read())
                            round_data = data[f"{round}"]
                            iteration = round_data["iteration"]
                            master_config = round_data["cfg"]["config"]

                            for key, value in master_config.items():
                                if value == "None":
                                    master_config[key] = None

                            rootLogger.warning(f"Loaded master config with round {round} it. {iteration}.")
                else:
                    if not find_master:
                        assert master_config is not None

                        config = copy.deepcopy(master_config)

                        for node_id in range(0, node_count):
                            done = False

                            while not done:
                                seed = (round + 1) * 10 + (node_count + 1) * 100 + (node_id + 1) * 1000 + (
                                        iteration + 1) * 10000

                                if node_id == 0:
                                    # node0 gets the unmodified master config
                                    pass
                                else:
                                    if flavor == "ensemble":
                                        # nodes take the full config later on and directly train on these parameters, i.e., no individual finetuning
                                        pass
                                    elif flavor == "ensemble_diverse":
                                        # nodes do their own hyperparameter tuning with the master parameters
                                        # but adjust the data type and add dircihlet alpha if needed
                                        if data_split == "dirichlet":
                                            config['datasplit'] = "dirichlet"
                                            config['dirichlet_alpha'] = 0.9

                                    elif "_diverse" in flavor:
                                        # types are architecture+scheduler_diverse, etc..
                                        diversity_type = flavor.replace("_diverse", "")
                                        subtypes = diversity_type.split("+")

                                        for st in subtypes:
                                            # makes sure to cycle through the possible configurations of the considered hyperparameter, while making sure that they are still compatible with one another
                                            if st == "architecture":
                                                config['arch'] = diverse_architecture(
                                                    current_architecture=config['arch'],
                                                    possible_architectures=possible_architectures,
                                                    node_id=node_id, iteration=iteration)
                                            elif st == "scheduler":
                                                config['sched'] = diverse_scheduler(current_scheduler=config['sched'],
                                                                                    current_optimizer=config['optim'],
                                                                                    possible_schedulers=possible_schedulers,
                                                                                    possible_momentum_optimizers=possible_momentum_optimizers,
                                                                                    possible_momentum_schedulers=possible_momentum_schedulers,
                                                                                    node_id=node_id,
                                                                                    iteration=iteration)
                                            elif st == "optimizer":
                                                config['optim'] = diverse_optimizer(current_optimizer=config['optim'],
                                                                                    current_scheduler=config['sched'],
                                                                                    possible_optimizers=possible_optimizers,
                                                                                    possible_momentum_schedulers=possible_momentum_schedulers,
                                                                                    possible_momentum_optimizers=possible_momentum_optimizers,
                                                                                    node_id=node_id,
                                                                                    iteration=iteration)
                                            else:
                                                raise NotImplementedError("Unsupported type!")
                                    else:
                                        raise NotImplementedError("Unsupported mode!")

                                # constructs string for the train module
                                args_string = f"--output {'{}_{},'.format(flavor, data_split) + datetime.now().isoformat()} --dataset {args.dataset} --tasks_per_gpu {args.tasks_per_gpu} --iteration {iteration} --round {round} --node_id {node_id} --node_count {node_count} --data_split {data_split}"

                                if flavor == "ensemble":
                                    args_string += f" --train_mode 3"
                                else:
                                    args_string += f" --architecture {config['arch']} --optimizer {config['optim']} --scheduler {config['sched']}"

                                if data_split == "dirichlet":
                                    args_string = args_string + " --dirichlet_alpha 0.9"

                                if args.verbose:
                                    print(args_string)

                                try:
                                    if flavor == "ensemble":
                                        node_path, node_config, node_test_acc = train.main(args_string,
                                                                                           config_dictionary=config)
                                    else:
                                        node_path, node_config, node_test_acc = train.main(args_string,
                                                                                           config_dictionary=None)

                                    rootLogger.warning(
                                        {"status": "SUCCESS", "time": time.asctime(), "round": round, "seed": seed,
                                         "config": node_config,
                                         "path": node_path,
                                         "test_acc": node_test_acc})

                                    done = True
                                except TuneError as e:
                                    rootLogger.warning(e)
                                    rootLogger.warning(
                                        {"status": "ERROR", "time": time.asctime(), "round": round, "seed": seed,
                                         "config": config})
                                    iteration = iteration + 1
                                    rootLogger.warning(
                                        f"Could not find converging config this round. Trying again with iteration {iteration}!")


def diverse_architecture(current_architecture, possible_architectures, node_id, iteration, remove=False):
    if remove:
        temp_archs = possible_architectures
    else:
        temp_archs = possible_architectures.copy()

    temp_archs.remove(current_architecture)
    arch = temp_archs[(node_id + iteration) % len(temp_archs)]

    return arch


def diverse_scheduler(current_scheduler, current_optimizer, possible_schedulers, possible_momentum_optimizers,
                      possible_momentum_schedulers, node_id, iteration, remove=False):
    if remove:
        temp_sched = possible_schedulers
    else:
        temp_sched = possible_schedulers.copy()

    temp_sched.remove(current_scheduler)

    if current_optimizer in possible_momentum_optimizers:
        sched = temp_sched[(node_id + iteration) % len(temp_sched)]
    else:
        # remove momentum_schedulers
        cleaned_sched = [el for el in temp_sched if
                         el not in possible_momentum_schedulers]
        sched = cleaned_sched[(node_id + iteration) % len(cleaned_sched)]

    return sched


def diverse_optimizer(current_optimizer, current_scheduler, possible_optimizers, possible_momentum_schedulers,
                      possible_momentum_optimizers, node_id, iteration, remove=False):
    if remove:
        temp_optim = possible_optimizers
    else:
        temp_optim = possible_optimizers.copy()

    temp_optim.remove(current_optimizer)

    if current_scheduler in possible_momentum_schedulers:
        temp_optim_mom = possible_momentum_optimizers.copy()
        temp_optim_mom.remove(current_optimizer)

        optim = temp_optim_mom[
            (node_id + iteration) % len(temp_optim_mom)]
    else:
        optim = temp_optim[(node_id + iteration) % len(temp_optim)]

    return optim


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction,
                        help="Prints command lines for scheduled task on the console.")
    parser.add_argument("--task", default=None, help="Task to be executed.")
    parser.add_argument("--output_file", default=None, help="File into which the Ray output is saved.")
    parser.add_argument("--gpus", default="0", help="Comma separated list of GPUs to use. Sorted by PCI_BUS_ID.")
    parser.add_argument("--tasks_per_gpu", type=int, default=2, help="Number of tasks per GPU.")
    parser.add_argument("--dataset", default="cifar10",
                        help="Dataset for training.")

    parser.add_argument("--rounds", default=None,
                        help="Comma separated list of specific rounds (and hence base models) to run the framework for. If nothing is specified all five rounds are executed.")
    parser.add_argument("--node_counts", default=None,
                        help="Comma separated list of specific number of nodes to run the framework for. If nothing is specified all node counts are executed")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.task

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"] = "0"
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    os.makedirs("data/attacks", exist_ok=True)
    os.makedirs("data/cifar10", exist_ok=True)
    os.makedirs("data/processed_data", exist_ok=True)
    os.makedirs("data/ray_logs", exist_ok=True)
    os.makedirs("data/ray_models", exist_ok=True)

    ray.init(
        configure_logging=True,
        logging_level=logging.INFO,
    )

    # logger setup
    rootLogger = logging.getLogger()
    logFormatter = logging.Formatter(f"%(message)s")

    fileHandler = logging.FileHandler("{0}/{1}.log".format(os.path.abspath("./data/ray_logs"), args.output_file), mode='a')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stderr)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    if args.task is not None:
        experiment(args.task, args)

        # just keep the final trained models
        other_experiments = glob.glob("data/ray_models/*_mini_optim_*")
        print("Cleanup of unused folders...")

        for f in other_experiments:
            shutil.rmtree(f)

    print("All workers finished!")
