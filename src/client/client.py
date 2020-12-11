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


class Client:
    def __init__(self, hostName, portFE, epochs):
        self.hostName = hostName
        self.portFE = portFE
        self.epochs = epochs

    def run(self):
        trans = TSocket.TSocket(self.hostName, self.portFE)
        trans = TTransport.TBufferedTransport(trans)
        proto = TBinaryProtocol.TBinaryProtocol(trans)
        client = FrontEnd.Client(proto)

        trans.open()
        result = client.trainNetwork(self.epochs)
        print(f"[Client] received: {result}")
        trans.close()


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