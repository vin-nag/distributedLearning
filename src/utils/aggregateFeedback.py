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

def aggregateFeedback(stateDictFiles, method="default"):
    if method == "default":
        # model, model2, model3 = Net(), Net(), Net()
        # model.load_state_dict({
        #     k: (v1 + v2) / 2 for (k, v1), (_, v2) in zip(model2.load_state_dict(torch.load(stateDictFiles[0])).items(),
        #                                                  model3.load_state_dict(torch.load(stateDictFiles[1].items())))
        # })
        model = Net()
        model.load_state_dict(torch.load(stateDictFiles[0]))
        return model
