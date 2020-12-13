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
import sys
sys.path.append("../../src")

from utils.models import Net
import torch
import copy


def aggregateFeedback(stateDictFiles, loss, method="average"):
    """

    :param stateDictFiles:
    :param loss:
    :param method:
    :return:
    """
    assert len(stateDictFiles) == len(loss), f"loss has {len(loss)} size but stateDict has {len(stateDictFiles)} size."
    numWorkers = len(stateDictFiles)
    stateDicts = [torch.load(file) for file in stateDictFiles]
    if len(stateDicts) == 1:
        result = stateDicts[0]
    else:
        result = copy.deepcopy(stateDicts[0])
        if method == "weighted":
            minVal, maxVal = min(loss), max(loss)
            weights = [(x-minVal)/(maxVal-minVal) for x in loss]
        elif method == "average":
            weights = [1.0 for x in loss]
        else:
            raise Exception(f"method: {method} not implemented.")
        for k,v in result.items():
            v = 0
            for i in range(len(stateDicts)):
                state = stateDicts[i]
                v += weights[i] * state[k]
            v /= numWorkers
    model = Net()
    model.load_state_dict(result)
    return model
