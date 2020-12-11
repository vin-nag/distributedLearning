
struct ResultFE {
    1: required list<double> trainAccuracy;
    2: required list<double> testAccuracy;
    3: required list<double> times;
}

struct ResultBE {
    1: required double accuracy;
}

service FrontEnd {
    ResultFE trainNetwork (1: i16 epochs );
    bool registerNode (1: string hostVal, 2: i16 portNum);
}

service BackEnd {
    double trainNetworkBE (1: string stateDictFile, 2: list<i32> indices, 3: string outputFile);
}
