"""
This file contains the code for the front-end service.

Date:
    December 10, 2020

Project:
    ECE751 Final Project: Distributed Neural Network Learning

Authors:
    name: Vineel Nagisetty, Husayn Kara
    contact: vineel.nagisetty@uwaterloo.ca
"""

import sys
sys.path.append("gen-py")

from project.FrontEnd import Client
from project.ttypes import ResultFE
from backEnd.serverBE import BENode


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
        print(f"[Front-end] Received notification from {hostVal}: {portNum}")
        if (hostVal, portNum) not in self.nodeList:
            node = BENode(hostName=hostVal, portNum=portNum)
            self.nodeList[(hostVal, portNum)] = node
            return True
        else:
            print(f"{hostVal}: {portNum} already in node list")
            return False

    def trainNetwork(self, epochs):
        return ResultFE([0],[1],[2])

