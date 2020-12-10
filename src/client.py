import sys
sys.path.append("gen-py")

from project.FrontEnd import Client
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import argparse


class Experiment:
    def __init__(self, hostName, portFE, epochs):
        self.hostName = hostName
        self.portFE = portFE
        self.epochs = epochs

    def run(self):
        trans = TSocket.TSocket(self.hostName, self.portFE)
        trans = TTransport.TBufferedTransport(trans)
        proto = TBinaryProtocol.TBinaryProtocol(trans)
        client = Client(proto)

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
    node = Experiment(args.host, args.portFE, args.epochs)
    node.run()


if __name__ == "__main__":
    main()