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
    def __init__(self, hostName, portFE, epochs):
        self.hostName = hostName
        self.portFE = portFE
        self.epochs = epochs
        self.results = pd.DataFrame(columns=('numWorkers', 'splitMethod', 'aggregateMethod', 'epochs', 'accuracies', 'time'))


    def run(self):
        trans = TSocket.TSocket(self.hostName, self.portFE)
        trans = TTransport.TFramedTransport(trans)
        proto = TBinaryProtocol.TBinaryProtocol(trans)
        client = FrontEnd.Client(proto)

        splitMethods = ["random", "class", "equal"]
        aggregateMethods = ["average", "weighted"]


        splitMethod = "random"
        aggregateMethod = "average"

        trans.open()
        i = 0
        # for splitMethod in splitMethods:
        #     for aggregateMethod in aggregateMethods:
        for i in range(3):
            result = client.trainNetwork(self.epochs, splitMethod=splitMethod, aggregateMethod=aggregateMethod)
            print(f"[Client] received: {result}, splitMethod: {splitMethod}, aggregateMethod: {aggregateMethod}")
            self.results.loc[i] = [result.numWorkers, splitMethod, aggregateMethod, self.epochs,
                              result.accuracies, result.time]
            i += 1
        trans.close()
        self.results.to_pickle("./../results/nodesThree.pkl")


def main():
    parser = argparse.ArgumentParser(description='Client for Distributed Neural Network Learning')
    parser.add_argument('--host', type=str, default="localhost",
                        help='host name (default: localhost)')
    parser.add_argument('--portFE', type=int, default=9090,
                        help='port number (default: 9090)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs (default: 5)')

    args = parser.parse_args()
    node = Client(args.host, args.portFE, args.epochs)
    node.run()


if __name__ == "__main__":
    main()