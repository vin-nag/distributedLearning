"""
This file contains the code for the back-end service.

Date:
    December 10, 2020

Project:
    ECE751 Final Project: Distributed Neural Network Learning

Authors:
    name: Vineel Nagisetty, Husayn Kara
    contact: vineel.nagisetty@uwaterloo.ca
"""

import sys
sys.path.append('gen-py')

from project.BackEnd import Client
from utils.models import Net
import torch
from torch.utils.data import Subset
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms


class BackEndHandler(Client):
    def __init__(self):
        self.nodeList = {}
        self.model = Net()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.trainset = datasets.MNIST(
            root="./../data",
            train=True,
            download=True,
            transform=self.transform
        )
        self.learning_rate = 0.01
        self.momentum = 0.5

    def trainNetworkBE(self, stateDictFile, indices, outputFile):
        """

        :param stateDictFile:
        :param indices:
        :param outputFile:
        :return:
        """
        print(f"Received {stateDictFile}, {len(indices)}, {outputFile} as input.")

        # load model
        self.model.load_state_dict(torch.load(stateDictFile))

        # load dataset
        data = Subset(self.trainset, indices)
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

        # initialize optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        # train model
        self.model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # save model state
        torch.save(self.model.state_dict(), outputFile)

        # return
        print("completed training")
        return total_loss
