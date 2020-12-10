import sys
sys.path.append("gen-py")

from project.FrontEnd import Processor
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from serviceFE import FrontEndHandler

import argparse


class FENode:
    def __init__(self, portFE):
        self.portFE = portFE

    def run(self):
        handler = FrontEndHandler()
        proc = Processor(handler)
        trans_svr = TSocket.TServerSocket(port=self.portFE)
        trans_fac = TTransport.TBufferedTransportFactory()
        proto_fac = TBinaryProtocol.TBinaryProtocolFactory()
        server = TServer.TSimpleServer(proc, trans_svr, trans_fac, proto_fac)

        print(f"[Server] Started on port {self.portFE}")
        server.serve()



def main():
    parser = argparse.ArgumentParser(description='Front End Server for Distributed Neural Network Learning')
    parser.add_argument('--port', type=int, default=9090, metavar='N',
                        help='port number (default: 9090)')
    args = parser.parse_args()
    node = FENode(args.port)
    node.run()


if __name__ == "__main__":
    main()