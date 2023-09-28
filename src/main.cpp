#include <iostream>
#include "NeuronNetwork\Network.hpp"

int main(){

    int const intendedOutput{23};
    std::vector<double>input = {-20.0, -15.2, -3.3, -4.4, -5.5, -6.6, -7.7};

    std::cout << "sliceOfBrain" << std::endl;
    NS::Layer sliceOfBrain = NS::Layer(input.size(),input.size(),NS::functionSigmoidNeuron);
    sliceOfBrain.activateLayer(input);
    sliceOfBrain.showLayer();

    std::cout << "smolBrain" << std::endl;
    NS::Network smolBrain = NS::Network(NS::MSELoss);
    smolBrain.addLayer(input.size(),input.size(),NS::functionSigmoidNeuron);
    smolBrain.addLayer(smolBrain.lastLayerSize(),30,NS::functionSigmoidNeuron);
    smolBrain.addLayer(smolBrain.lastLayerSize(),30,NS::functionSigmoidNeuron);
    smolBrain.addLayer(smolBrain.lastLayerSize(),2,NS::functionSigmoidNeuron);
    smolBrain.activateNetwork(input);
    std::cout << smolBrain << std::endl;

    return 0;
}