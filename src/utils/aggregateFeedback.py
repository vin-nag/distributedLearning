"""
This file contains the code for aggregating feedback.

Date:
    December 10, 2020

Project:
    ECE751 Final Project: Distributed Neural Network Learning

Authors:
    name: Vineel Nagisetty, Husayn Kara
    contact: vineel.nagisetty@uwaterloo.ca
"""

from utils.models import Net
import torch
import copy


def aggregateFeedback(stateDictFiles, method="default"):
    if method == "default":
        numWorkers = len(stateDictFiles)
        stateDicts = [torch.load(file) for file in stateDictFiles]
        averageStateDict = copy.deepcopy(stateDicts[0])

        for k,v in averageStateDict.items():
            v = 0
            for state in stateDicts:
                v += state[k]
            v /= numWorkers
        print('completed aggregation')
        model = Net()
        model.load_state_dict(averageStateDict)
        return model
