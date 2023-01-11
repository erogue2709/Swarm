#pragma once
#include <math.h>
#include <vector>

namespace NS {

    enum lossFunction {
        crossEntropyLoss,
        MSE
    };

    enum activationType {
        functionSigmoidNeuron,
        functionReLuNeuron
    };
    std::vector<double> activeFunction(const activationType function, const std::vector<double> input);
    std::vector<double> functionSigmoid(const std::vector<double> input);
    std::vector<double> functionReLu(const std::vector<double> input);

    std::vector<double> derivateFunction(const activationType function, const std::vector<double> output);
    std::vector<double> derivateSigmoid(const std::vector<double> output);
    std::vector<double> derivateReLu(const std::vector<double> output);

}