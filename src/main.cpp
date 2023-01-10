#include <iostream>
#include "NeuronNetwork\Network.hpp"

int main(){

    int const nbOutput {1};
    int const intendedOutput{23};
    int const inputSize {7};
    std::vector<double>input = {-20.0, -15.2, -3.3, -4.4, -5.5, -6.6, -7.7};
    
    NS::Layer sliceOfBrain = NS::Layer(inputSize,3,NS::functionSigmoidNeuron);
    sliceOfBrain.activateLayer(input);
    sliceOfBrain.showLayer();

/*
    NS::Network firstBrain = NS::Network(inputSize, nbOutput, 2, NS::functionSigmoidNeuron, NS::functionSigmoidNeuron);
    firstBrain.activateNetwork(input);
    std::cout << firstBrain << std::endl;
*/    
    return 0;
}