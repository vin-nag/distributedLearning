"""
This file contains the code for the client server.

Date:
    December 10, 2020

Project:
    ECE751 Final Project: Distributed Neural Network Learning

Authors:
    name: Vineel Nagisetty, Husayn Kara
    contact: vineel.nagisetty@uwaterloo.ca
"""

import sys
sys.path.append("../gen-py")

from project import FrontEnd
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import argparse
import pandas as pd


class Client:
    def __init__(self, hostName, portFE, epochs, runs, splitMethod, aggregateMethod, outputFile):
        self.hostName = hostName
        self.portFE = portFE
        self.epochs = epochs
        self.runs = runs
        self.splitMethod = splitMethod
        self.aggregateMethod = aggregateMethod
        self.outputFile = outputFile
        self.results = pd.DataFrame(columns=('numWorkers', 'splitMethod', 'aggregateMethod', 'epochs', 'accuracies', 'time'))


    def run(self):
        trans = TSocket.TSocket(self.hostName, self.portFE)
        trans = TTransport.TFramedTransport(trans)
        proto = TBinaryProtocol.TBinaryProtocol(trans)
        client = FrontEnd.Client(proto)

        splitMethods = ["random", "class", "equal"]
        aggregateMethods = ["average", "weighted"]


        trans.open()
        # for splitMethod in splitMethods:
        #     for aggregateMethod in aggregateMethods:
        for i in range(self.runs):
            result = client.trainNetwork(self.epochs, splitMethod=self.splitMethod, aggregateMethod=self.aggregateMethod)
            print(f"[Client] received: {result}, splitMethod: {self.splitMethod}, aggregateMethod: {self.aggregateMethod}")
            self.results.loc[i] = [result.numWorkers, self.splitMethod, self.aggregateMethod, self.epochs,
                              result.accuracies, result.time]
        trans.close()
        self.results.to_pickle(self.outputFile)


def main():
    parser = argparse.ArgumentParser(description='Client for Distributed Neural Network Learning')
    parser.add_argument('--host', type=str, default="localhost",
                        help='host name (default: localhost)')
    parser.add_argument('--portFE', type=int, default=9090,
                        help='port number (default: 9090)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs (default: 2)')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of runs for each experiment (default: 1)')
    parser.add_argument('--splitMethod', type=str, default="class",
                        help='the type of split method (choose between random [default], class and equal)')
    parser.add_argument('--aggregateMethod', type=str, default="average",
                        help='the type of aggregation (choose between average [default] and weighted)')
    parser.add_argument('--outputFile', type=str, default="./../results/nodeSeven.pkl",
                        help='the name of the file to save results in.')

    args = parser.parse_args()
    node = Client(args.host, args.portFE, args.epochs, args.runs, args.splitMethod, args.aggregateMethod, args.outputFile)
    node.run()


if __name__ == "__main__":
    main()