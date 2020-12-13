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
sys.path.append("../../src")


from project.FrontEnd import Client
from project.ttypes import ResultFE
from backEnd.BENode import BENode
from utils.models import Net
from torch import save, load
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.splitData import splitData
import time
from concurrent.futures import ThreadPoolExecutor
from utils.aggregateFeedback import aggregateFeedback
import concurrent
from thrift.Thrift import TException


class FrontEndHandler(Client):
    def __init__(self):
        self.nodeMap = {}
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
        if (hostVal, portNum) not in self.nodeMap:
            node = BENode(hostName=hostVal, portNum=portNum)
            self.nodeMap[(hostVal, portNum)] = node
            return True
        else:
            print(f"{hostVal}: {portNum} already in node list")
            return False

    def delegateEachWork(self, key, modelFile, indices, outputFile):
        """

        :param key:
        :param modelFile:
        :param indices:
        :param outputFile:
        :return:
        """
        transport = self.nodeMap[key].getTransport()
        if not transport.isOpen():
            transport.open()
        client = self.nodeMap[key].getClient()
        return client.trainNetworkBE(modelFile, indices, outputFile)

    def trainNetwork(self, epochs, splitMethod, aggregateMethod):
        """
        This function performs the main Downpour-SGD algorithm
        :param epochs:
        :param splitMethod:
        :param aggregateMethod:
        :return:
        """
        batch_size = 64
        learning_rate = 1.0
        gamma = 0.7
        modelFile = "./../data/state.pt"

        model = Net()
        save(model.state_dict(), modelFile)

        numOfWorkers = len(self.nodeMap)

        keys = list(self.nodeMap.keys())
        indices = splitData(60000, numWorkers=numOfWorkers, method=splitMethod)
        outputFiles = [f"./../data/state{self.nodeMap[key].getInfo()}.pt" for key in keys]
        clients = [self.nodeMap[key].getClient() for key in keys]

        # test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_data = datasets.MNIST('./../data', train=False, transform=transform)
        test_loader = DataLoader(test_data, batch_size)

        accuracy = []

        start_time = time.time()
        for j in range(epochs):

            # train the model
            model.train()

            with ThreadPoolExecutor(max_workers=numOfWorkers) as executor:
                loss = []
                future_to_BE = {executor.submit(self.delegateEachWork, keys[i], modelFile, indices[i], outputFiles[i]): i for i in range(numOfWorkers)}
                for future in concurrent.futures.as_completed(future_to_BE):
                    result = future_to_BE[future]
                    try:
                        loss.append(future.result())
                    except Exception as exc:
                        print(f"generated exception {exc}")

            model = aggregateFeedback(outputFiles, loss, method=aggregateMethod)
            save(model.state_dict(), modelFile)

            # test the model
            model.eval()
            test_loss = 0
            correct = 0
            number_classes = 10
            confusion_matrix = torch.zeros(number_classes, number_classes)
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    for t, p in zip(target.view(-1), pred.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            val = 100. * correct / len(test_loader.dataset)

            print(confusion_matrix.diag() / confusion_matrix.sum(1))
            print(f"epoch: {j+1} accuracy: {val}")
            accuracy.append(val)

        return ResultFE(accuracy, time.time()-start_time, numOfWorkers)

