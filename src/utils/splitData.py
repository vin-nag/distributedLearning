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

import torch
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
    lst = [x for x in range(dataSize)]
    if numWorkers == 1:
        return [lst]

    if method == "random":
        splits = []
        random.shuffle(lst)
        size = dataSize//numWorkers
        for i in range(numWorkers-1):
            splits.append(lst[i*size: (i+1)*size])
        splits.append(lst[(i+1)*size:])
        return splits
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
        i = 0
        for batch_idx, (data, target) in enumerate(loader):
            splits[target.item()].append(i)
            i += 1
            if (i==dataSize and method=="class") or (i==dataSize*3 and method=="equal"):
                break
        if method == "class":
            return splits
        elif method =="equal":
            result = []
            minSize = min([len(x) for x in splits])
            size = (dataSize // numWorkers)//10
            for i in range(numWorkers):
                temp = []
                for sublist in splits:
                    temp.extend(sublist[i*size:(i+1)*size])
                result.append(temp)
            print([len(x) for x in result])
            return result
        else:
            print(f"the data splitting method {method} is not supported")
