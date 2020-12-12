"""
This file contains the code for the front-end server.

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

from project.FrontEnd import Processor
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TNonblockingServer
from serviceFE import FrontEndHandler

import argparse


class FENodeServer:
    def __init__(self, portFE):
        self.portFE = portFE

    def run(self):
        handler = FrontEndHandler()
        proc = Processor(handler)
        trans_svr = TSocket.TServerSocket(port=self.portFE)
        proto_fac = TBinaryProtocol.TBinaryProtocolFactory()
        server = TNonblockingServer.TNonblockingServer(proc, trans_svr, proto_fac)

        print(f"[Server] Started on port {self.portFE}")
        server.serve()


def main():
    parser = argparse.ArgumentParser(description='Front End Server for Distributed Neural Network Learning')
    parser.add_argument('--port', type=int, default=9090, metavar='N',
                        help='port number (default: 9090)')
    args = parser.parse_args()
    node = FENodeServer(args.port)
    node.run()


if __name__ == "__main__":
    main()