#pragma once
#include <math.h>
#include <vector>

namespace NS {

    enum lossFunction {
        crossEntropyLoss,
        MSELoss
    };

    std::vector<double> MSEFunction(const std::vector<std::vector<double>> outputs
            , const std::vector<std::vector<double>> intendedOutputs);

    enum activationType {
        functionSigmoidNeuron,
        functionReLuNeuron
    };
    std::vector<double> functionSigmoid(const std::vector<double> input);
    std::vector<double> functionReLu(const std::vector<double> input);

    std::vector<double> derivateSigmoid(const std::vector<double> output);
    std::vector<double> derivateReLu(const std::vector<double> output);

    void NetworkTesting();

}