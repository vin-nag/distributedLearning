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

from torch.utils.data import Sampler
import torch


def splitData(dataSize, numWorkers, method="default"):
    splits = []
    if method == "default":
        if numWorkers == 1:
            return [[x for x in range(dataSize)]]
        size = dataSize//numWorkers
        for i in range(numWorkers-1):
            splits.append([x for x in range(i*size, (i+1)*size)])
        splits.append([x for x in range((i+1)*size, dataSize)])
        return splits
