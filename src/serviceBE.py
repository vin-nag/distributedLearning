import sys
sys.path.append('gen-py')

from BackEnd import Client
from model import Net
import torch
from splitData import DataSampler
from torch.utils.data import Subset
from torch import optim
import torch.functional as F


class BackEndHandler(Client):

    def __init__(self):
        self.nodeList = {}
        self.model = Net()
        self.trainset = None
        self.learning_rate = 0.99
        self.momentum = 0.5

    def trainNetworkBE(self, stateDictFile, indices, outputFile):

        # load model
        self.model.load_state_dict(torch.load(stateDictFile))

        # load dataset
        data = Subset(self.trainset, indices)
        loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)

        # initialize optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        # train model
        self.model.train()
        accuracy = 0
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            accuracy += loss / len(output)
            loss.backward()
            optimizer.step()

        # save model state
        torch.save(self.model.state_dict(), outputFile)

        # return accuracy
        return accuracy





