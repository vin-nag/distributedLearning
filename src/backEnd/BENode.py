from thrift.transport import TSocket, TSSLSocket
from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from project.BackEnd import Client as ClientBE


class BENode:
    def __init__(self, hostName, portNum):
        self.hostName = hostName
        self.portNum = portNum
        self.BENodeTransport, self.BENodeClient = None, None
        self.establishConnection()

    def establishConnection(self):
        sock = TSocket.TSocket(host=self.hostName, port=self.portNum)
        self.BENodeTransport = TTransport.TFramedTransport(sock)
        protocol = TBinaryProtocol.TBinaryProtocol(self.BENodeTransport)
        self.BENodeClient = ClientBE(protocol)

    def getTransport(self):
        return self.BENodeTransport

    def getClient(self):
        return self.BENodeClient

    def getInfo(self):
        return self.hostName + str(self.portNum)

