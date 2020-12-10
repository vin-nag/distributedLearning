import sys
import glob

import time

from project.FrontEnd import Client
from project.ttypes import ResultFE
import torch
from model import Net
import torch.optim as optim
import torch.functional as F


class FrontEndHandler(Client):
    def __init__(self):
        self.nodeList = {}
        self.batchSize = 64
        self.learning_rate = 0.01
        self.momentum = 0.5

    def registerNode(self, hostVal, portNum):
        """
        This function registers a back-end node to the front-end
        :param hostVal: string
        :param portNum: int
        :return:
        """
        self.nodeList[(hostVal, portNum)] = 1

    def trainNetwork(self, epochs):
        return ResultFE([0],[1],[2])

