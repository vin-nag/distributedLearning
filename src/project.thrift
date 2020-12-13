
struct ResultFE {
    1: required list<double> accuracies;
    2: required double time;
    3: required i16 numWorkers;
}

struct ResultBE {
    1: required double accuracy;
}

service FrontEnd {
    ResultFE trainNetwork (1: i16 epochs 2: string splitMethod, 3: string aggregateMethod );
    bool registerNode (1: string hostVal, 2: i16 portNum);
}

service BackEnd {
    double trainNetworkBE (1: string stateDictFile, 2: list<i32> indices, 3: string outputFile);
}
