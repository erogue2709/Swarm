#include "ActiveNeuron.hpp"

namespace NS {

    std::vector<double> MSEFunction(const std::vector<std::vector<double>> outputs, const std::vector<std::vector<double>> intendedOutputs){
                
        int intermediateSubtraction;
        std::vector<double> meanSqrtSum (outputs[0].size(), 0.0);

        for(int x=0; x < outputs.size(); x++){
            for(int y=0; y < outputs[x].size(); y++){
                intermediateSubtraction = outputs[x][y]-intendedOutputs[x][y];
                sqrtSum[y] += intermediateSubtraction*intermediateSubtraction;
            }
        }

        for(int k=0; k < meanSqrtSum[0].size(); k++){
            sqrtSum[k] /= outputs.size();
        }

        return meanSqrtSum;
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