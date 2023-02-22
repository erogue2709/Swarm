#include "ActiveNeuron.hpp"

namespace NS {

    std::vector<double> MSEFunction(const std::vector<std::vector<double>> outputs
            , const std::vector<std::vector<double>> intendedOutputs){
        //MSE for batch gradiant descent
        //change some std::vector<double> to double and sum results of fucntion
        //call / wrap into other function to convert to stochastic gradient descent??
        /*
        std::vector<double> temp;
        
        for(int k=0; k < outputs.size(); k++){
            temp[k] = (outputs[k]-intendedOutputs[k])*(outputs[k]-intendedOutputs[k]);
        }*/
        return outputs[0];
    }

    std::vector<double> activeFunction(const activationType function, const std::vector<double> input){
        switch (function)
        {
        case functionSigmoidNeuron:
            return functionSigmoid(input);
        case functionReLuNeuron:
            return functionReLu(input);
            break;
        }
        return input;
    }

    std::vector<double> functionSigmoid(const std::vector<double> input){
        std::vector<double> temp(input.size());
        for(int k=0; k < input.size(); k++){
            temp[k] = input[k]/(1.0+fabs(input[k]));
        }
        return temp;
    }

    std::vector<double> functionReLu(const std::vector<double> input){
        std::vector<double> temp(input.size());
        for(int k=0; k < input.size(); k++){
            temp[k] = (input[k]>=0)?input[k]:0.0;
        }
        return temp;
    }

    std::vector<double> derivateFunction(const activationType function, const std::vector<double> output){
        switch (function)
        {
        case functionSigmoidNeuron:
            return derivateSigmoid(output);
        case functionReLuNeuron:
            return derivateReLu(output);
            break;
        }
        return output;
    }

    std::vector<double> derivateSigmoid(const std::vector<double> output){
        std::vector<double> temp(output.size());
        for(int k=0; k < output.size(); k++){
            temp[k] = output[k]-(1-output[k]);
        }
        return temp;
    }

    std::vector<double> derivateReLu(const std::vector<double> output){
        std::vector<double> temp(output.size());
        for(int k=0; k < output.size(); k++){
            temp[k] = (output[k]>=0)?1.0:0.0;
        }
        return temp;
    }
}