from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

from models.cifar10_models import VGG, ResNet18, PreActResNet18, GoogLeNet, DenseNet121, ResNeXt29_2x64d, \
    MobileNetV2, DPN92, \
    ShuffleNetG2, SENet18, EfficientNetB0, RegNetX_200MF, SimpleDLA

possibleDataSplit = ["uniform", "dirichlet"]

possibleArchitectures = {
    "cifar10": ["vgg19", "resnet18", "preactresnet18", "googlenet", "densenet121", "resnext29_2x64d",
                "mobilenet_v2", "dpn92", "senet18", "efficientnet_b0", "regnetx_200mf",
                "simpledla"]
}

possibleOptimizers = ["SGD", "SGD-M", "SGD-N", "Adam", "NAdam", "Adagrad", "ASGD", "Rprop", "RMSprop"]
possibleMomentumOptimizers = ["SGD", "SGD-M", "SGD-N", "RMSprop"]
possibleWeightDecayOptimizers = ["SGD", "SGD-M", "SGD-N", "Adam", "NAdam", "Adagrad", "ASGD", "RMSprop"]

possibleSchedulers = ["CosineAnnealingLR", "StepLR", "ExponentialLR", "ReduceLROnPlateau", "CyclicLR"]
possibleMomentumSchedulers = ["CyclicLR"]


class SchedulerParameters:
    def __init__(self, scheduler_name, json=None):
        if json is not None:
            if scheduler_name != json["scheduler_name"]:
                raise ValueError(
                    f"scheduler name {scheduler_name} does not match with the json file {json['scheduler_name']}")

        self.scheduler_name = scheduler_name
        self.scheduler_param = json["scheduler_param"]

    def toJSON(self):
        return self.__dict__

    def __str__(self):
        return str(self.toJSON())

    def __repr__(self):
        return str(self.toJSON())

    def __eq__(self, other):
        return self.toJSON() == other.toJSON()

    def __ne__(self, other):
        return self.toJSON() != other.toJSON()


class OptimizerParameters:
    def __init__(self, optimizer_name, json=None):
        self.optimizer_name = optimizer_name
        if json is not None:
            if optimizer_name != json["optimizer_name"]:
                raise ValueError(
                    f"optimizer name {optimizer_name} does not match with the json file {json['optimizer_name']}")

            self.lr = json["lr"]
            self.weight_decay = json["weight_decay"]

            if "momentum" in json:
                self.momentum = json["momentum"]  

    def toJSON(self):
        return self.__dict__

    def __str__(self):
        return str(self.toJSON())

    def __repr__(self):
        return str(self.toJSON())

    def __eq__(self, other):
        return self.toJSON() == other.toJSON()

    def __ne__(self, other):
        return self.toJSON() != other.toJSON()


class DecentralizedParameters:

    def __init__(self, datasplit, dirichlet_alpha=None, json=None, dataset_seed=None, batch_size=512, nodeID=None,
                 totalN=None):

        self.dataset_seed = dataset_seed
        self.batch_size = batch_size

        self.datasplit = datasplit
        self.dirichlet_alpha = dirichlet_alpha
        self.nodeID = nodeID
        self.totalN = totalN

        if json is not None:
            for key in json:
                self.__dict__[key] = json[key]

    def get_scheduler(self, optimizer):
        if self.scheduler is None:
            return None

        name = self.scheduler.scheduler_name

        if name == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.scheduler.scheduler_param)
        elif name == "StepLR":
            return optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler.scheduler_param)
        elif name == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler.scheduler_param)
        elif name == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        elif name == "CyclicLR":
            return optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.optimizer.lr, max_lr=10 * self.optimizer.lr)

    def get_optimizer(self, model_params):
        name = self.optimizer.optimizer_name
        lr = self.optimizer.lr
        wd = self.optimizer.weight_decay

        if name == "SGD":
            return optim.SGD(model_params, lr=lr, weight_decay=wd)
        elif name == "SGD-M":
            return optim.SGD(model_params, lr=lr, momentum=self.optimizer.momentum, weight_decay=wd)
        elif name == "SGD-N":
            return optim.SGD(model_params, lr=lr, momentum=self.optimizer.momentum, nesterov=True, weight_decay=wd)
        elif name == "Adam":
            return optim.Adam(model_params, lr=lr, weight_decay=wd)
        elif name == "NAdam":
            return optim.NAdam(model_params, lr=lr, weight_decay=wd)
        elif name == "Adagrad":
            return optim.Adagrad(model_params, lr=lr, weight_decay=wd)
        elif name == "ASGD":
            return optim.ASGD(model_params, lr=lr, weight_decay=wd)
        elif name == "Rprop":
            return optim.Rprop(model_params, lr=lr)
        elif name == "RMSprop":
            return optim.RMSprop(model_params, lr=lr, momentum=self.optimizer.momentum, weight_decay=wd)

    def toJSON(self):
        return self.__dict__

    def __str__(self):
        return str(self.toJSON())

    def __repr__(self):
        return str(self.toJSON())

    def __eq__(self, other):
        return self.toJSON() == other.toJSON()

    def __ne__(self, other):
        return self.toJSON() != other.toJSON()


def get_dataset(dataset, train=True, path=None, params=None, download=True, force_train=False):
    if dataset == "cifar10":
        if not train:
            transform = transforms.Compose(
                [transforms.ToImage(),
                 transforms.ToDtype(torch.float32, scale=True),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

            testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                                   download=download, transform=transform)
            return testset
        else:
            transform = transforms.Compose(
                [transforms.ToImage(),
                 transforms.ToDtype(torch.uint8, scale=True),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomResizedCrop(size=(32, 32), antialias=True),
                 transforms.ToDtype(torch.float32, scale=True),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

            trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                                    download=download, transform=transform)

            if force_train:
                return trainset

    # split data into chunks when training
    if train:
        nmodels = params.totalN
        nid = params.nodeID
        if params.datasplit == "uniform":
            indices = list(range(len(trainset)))
            # split indices in nmodels
            # shuffle indices
            np.random.seed(params.dataset_seed)
            np.random.shuffle(indices)
            indices = np.array_split(indices, nmodels)
            node_indices = indices[nid]
            splittedDataset = SplitByIndexDataset(node_indices, trainset)

        elif params.datasplit == "dirichlet":
            alpha = params.dirichlet_alpha
            dataset_classes = {}
            np.random.seed(params.dataset_seed)
            for ind, label in enumerate(trainset):
                label = label[1]
                if label in dataset_classes:
                    dataset_classes[label].append(ind)
                else:
                    dataset_classes[label] = [ind]

            per_node_indices = defaultdict(list)
            no_classes = len(dataset_classes.keys())

            for n in range(no_classes):
                np.random.shuffle(dataset_classes[n])
                class_size = len(dataset_classes[n])
                sampled_probabilities = class_size * np.random.dirichlet(
                    np.array(nmodels * [alpha]))
                for node in range(nmodels):
                    no_imgs = int(round(sampled_probabilities[node]))
                    sampled_list = dataset_classes[n][:min(len(dataset_classes[n]), no_imgs)]
                    per_node_indices[node].extend(sampled_list)
                    dataset_classes[n] = dataset_classes[n][min(len(dataset_classes[n]), no_imgs):]
            splittedDataset = SplitByIndexDataset(per_node_indices[nid], trainset)

        return splittedDataset


def get_loader(dataset, train=True, path=None, params=None, download=True, force_train=False, batch_size=512):
    data = get_dataset(dataset=dataset, train=train, path=path, params=params, download=download,
                       force_train=force_train)
    if not train:
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


def get_model(dataset, arch):
    name = arch

    if dataset == "cifar10" and name in possibleArchitectures[dataset]:
        if name == "vgg19":
            mod = VGG('VGG19')
        elif name == "resnet18":
            mod = ResNet18()
        elif name == "preactresnet18":
            mod = PreActResNet18()
        elif name == "googlenet":
            mod = GoogLeNet()
        elif name == "densenet121":
            mod = DenseNet121()
        elif name == "resnext29_2x64d":
            mod = ResNeXt29_2x64d()
        elif name == "mobilenet_v2":
            mod = MobileNetV2()
        elif name == "dpn92":
            mod = DPN92()
        elif name == "shufflenetg2":
            mod = ShuffleNetG2()
        elif name == "senet18":
            mod = SENet18()
        elif name == "efficientnet_b0":
            mod = EfficientNetB0()
        elif name == "regnetx_200mf":
            mod = RegNetX_200MF()
        elif name == "simpledla":
            mod = SimpleDLA()

    return mod


class SplitByIndexDataset(Dataset):
    def __init__(self, idx, dataset):
        self.idx = idx
        self.ds = dataset

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        realid = self.idx[idx]
        img, label = self.ds[realid]
        return img, label


class SelfDataset(Dataset):
    def __init__(self, images, labels, transform) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.t = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.t(self.images[index]), self.labels[index]
