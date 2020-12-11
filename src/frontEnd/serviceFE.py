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
from backEnd.BENode import BENode
from utils.models import Net
from torch import save, load
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from utils.splitData import splitData
import time
from concurrent.futures import ThreadPoolExecutor
from utils.aggregateFeedback import aggregateFeedback
import concurrent
from thrift.Thrift import TException
import asyncio


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

    async def delegateEachWork(self, key, modelFile, indices, outputFile):
        transport = self.nodeMap[key].getTransport()
        if not transport.isOpen():
            transport.open()
        client = self.nodeMap[key].getClient()
        return client.trainNetworkBE(modelFile, indices, outputFile)

    async def delegateWork(self, keys, modelFile):
        tasks = []
        numWorkers = len(keys)
        indices = splitData(60000, numWorkers=numWorkers)
        outputFiles = [f"./../data/state{self.nodeMap[key].getInfo()}.pt" for key in keys]
        for i in range(numWorkers):
            tasks.append(asyncio.create_task(self.delegateEachWork(keys[i], modelFile, indices[i], outputFiles[i])))
            print('task added')
        for task in tasks:
            await task
        # return await asyncio.gather(*[self.delegateEachWork(keys[i], modelFile, indices[i], outputFiles[i]) for i in range(numWorkers)])
        return

    def trainNetwork(self, epochs):

        start_time = time.time()

        time_taken = 0

        # experiment constants
        batch_size = 64
        learning_rate = 1.0
        gamma = 0.7
        modelFile = "./../data/state.pt"

        model = Net()
        save(model.state_dict(), modelFile)

        numOfWorkers = len(self.nodeMap)
        print(numOfWorkers)

        keys = list(self.nodeMap.keys())
        indices = splitData(60000, numWorkers=numOfWorkers)
        outputFiles = [f"./../data/state{self.nodeMap[key].getInfo()}.pt" for key in keys]
        clients = [self.nodeMap[key].getClient() for key in keys]

        # test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_data = datasets.MNIST('./../data', train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size)

        accuracy = []

        for j in range(epochs):
            # train the model
            model.train()
            print(f"epoch: {j+1}")

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(self.delegateWork(keys, modelFile))

            model = aggregateFeedback(outputFiles)
            save(model.state_dict(), modelFile)

        loop.close()
        # test the model
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        val = 100. * correct / len(test_loader.dataset)
        print(f"epoch: {j+1} accuracy: {val}")
        accuracy.append(val)

        print(accuracy, time.time() - start_time)

