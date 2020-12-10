import sys
sys.path.append("gen-py")

from project.BackEnd import Processor
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from serviceBE import BackEndHandler

import argparse


class BENode:
    def __init__(self, portBE):
        self.portBE = portBE

    def run(self):
        handler = BackEndHandler()
        proc = Processor(handler)
        trans_svr = TSocket.TServerSocket(port=self.portBE)
        trans_fac = TTransport.TBufferedTransportFactory()
        proto_fac = TBinaryProtocol.TBinaryProtocolFactory()
        server = TServer.TSimpleServer(proc, trans_svr, trans_fac, proto_fac)

        print(f"[Server] Started on port {self.portBE}")
        server.serve()


def main():
    parser = argparse.ArgumentParser(description='Back End Server for Distributed Neural Network Learning')
    parser.add_argument('--port', type=int, default=9091,
                        help='port number (default: 9091)')
    args = parser.parse_args()
    node = BENode(args.port)
    node.run()


if __name__ == "__main__":
    main()