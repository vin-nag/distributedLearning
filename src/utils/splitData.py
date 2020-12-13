"""
This file contains the code for splitting the data.

Date:
    December 10, 2020

Project:
    ECE751 Final Project: Distributed Neural Network Learning

Authors:
    name: Vineel Nagisetty, Husayn Kara
    contact: vineel.nagisetty@uwaterloo.ca
"""

import sys
sys.path.append("../../src")

from os import path
import torch
import pickle
import random
from torchvision import datasets, transforms


def splitData(dataSize, numWorkers, method="random"):
    """
    This function splits the data.
    :param dataSize:
    :param numWorkers:
    :param method:
    :return:
    """
    if numWorkers == 1:
        return [x for x in range(dataSize)]

    if method == "random":
        lst = [x for x in range(dataSize)]
        random.shuffle(lst)
        size = dataSize//numWorkers
        splits = []
        for i in range(numWorkers-1):
            splits.append(lst[i*size: (i+1)*size])
        splits.append(lst[(i+1)*size:])
        return splits

    elif method == "class":
        return caculateClassIndices()

    elif method == "equal":
        splits = caculateClassIndices()
        result = []
        minSize = min([len(x) for x in splits])//numWorkers
        size = min(minSize,(dataSize//numWorkers)//10)
        print(size, minSize, dataSize, numWorkers)
        print([len(x) for x in splits])
        for i in range(numWorkers):
            temp = []
            for j in range(len(splits)):
                temp.extend(splits[j][i*size:(i+1)*size])
            result.append(temp)
        print([(len(x), len(set(x))) for x in result])
        return result

    else:
        print(f"the data splitting method {method} is not supported")


def caculateClassIndices():
    fname = "./../data/class.pickle"
    if path.exists(fname):
        with open(fname, "rb") as fp:
            splits = pickle.load(fp)
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = datasets.MNIST(
            root="./../data",
            train=True,
            download=True,
            transform=transform
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        splits = [[0] for i in range(10)]
        labels = [[0] for i in range(10)]
        i = 0
        for batch_idx, (data, target) in enumerate(loader):
            splits[target.item()].append(i)
            labels[target.item()].append(target.item())
            i += 1
        splits = [x[1:] for x in splits]
        with open(fname, "wb") as fp:
            pickle.dump(splits, fp)
    return splits

