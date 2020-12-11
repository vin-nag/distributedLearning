
struct ResultFE {
    1: required list<double> accuracy;
    3: required double time;
}

struct ResultBE {
    1: required double accuracy;
}

service FrontEnd {
    oneway void trainNetwork (1: i16 epochs );
    bool registerNode (1: string hostVal, 2: i16 portNum);
}

service BackEnd {
    bool trainNetworkBE (1: string stateDictFile, 2: list<i32> indices, 3: string outputFile);
}
