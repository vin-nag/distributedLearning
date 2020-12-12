"""
This file contains the code for the back-end server.

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

from project.BackEnd import Processor
from thrift.transport import TSocket, TSSLSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer, TNonblockingServer
from project.BackEnd import Client as ClientBE
from project.FrontEnd import Client as ClientFE
from backEnd.serviceBE import BackEndHandler
import argparse


class BENodeServer:
    def __init__(self, portBE, hostFE, portFE):
        self.portBE = portBE
        self.hostFE = hostFE
        self.portFE = portFE
        self.signalFrontEnd()

    def signalFrontEnd(self):
        sock = TSocket.TSocket(host=self.hostFE, port=self.portFE)
        transport = TTransport.TFramedTransport(sock)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        client = ClientFE(protocol)
        transport.open()
        print(f"[Back-end] Pinging Front-end with {self.hostFE}: {self.portBE}")
        result = client.registerNode(self.hostFE, self.portBE)
        if result:
            print("Succesful")
        else:
            print("Already in List")
        sock.close()
        transport.close()

    def run(self):
        handler = BackEndHandler()
        proc = Processor(handler)
        trans_svr = TSocket.TServerSocket(port=self.portBE)
        proto_fac = TBinaryProtocol.TBinaryProtocolFactory()
        server = TNonblockingServer.TNonblockingServer(proc, trans_svr, proto_fac)

        print(f"[Server] Started on port {self.portBE}")
        server.serve()


def main():
    parser = argparse.ArgumentParser(description='Back End Server for Distributed Neural Network Learning')
    parser.add_argument('--portBE', type=int, default=9091,
                        help='Back-end port number (default: 9091)')
    parser.add_argument('--hostFE', type=str, default="localhost",
                        help='Front-end host name (default: localhost)')
    parser.add_argument('--portFE', type=int, default=9090,
                        help='Front-end port number (default: 9090)')
    args = parser.parse_args()
    node = BENodeServer(args.portBE, args.hostFE, args.portFE)
    node.run()


if __name__ == "__main__":
    main()