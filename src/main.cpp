#include <iostream>
#include "NeuronNetwork\Network.hpp"

int main(){

    int const nbOutput {1};
    int const intendedOutput{23};
    int const inputSize {7};
    std::vector<double>input = {20.0, 15.2, 3.3, 4.4, 5.5, 6.6, 7.7};

    NS::Network firstBrain = NS::Network(inputSize, nbOutput, 0, NS::functionHeavisideNeuron, NS::functionHeavisideNeuron);
    std::cout << firstBrain << std::endl;
    std::cout << "------------------------------" << std::endl;
    firstBrain.activateNetwork(input);
    std::cout << firstBrain << std::endl;
    return 0;
}