#include "ActiveNeuron.hpp"

namespace NS {

    std::vector<double> MSEFunction(const std::vector<std::vector<double>> outputs
            , const std::vector<std::vector<double>> intendedOutputs){
        
        double sqrtSum = 0.0; //normalize to avoid overflow?
        std::vector<double> meanSqrtSum;
        
        for(int k=0; k < outputs.size(); k++){
            //Nope... need to loop every outputs for every neurons (1 in mse' case tho :/)
            sqrtSum += (outputs[k][0]-intendedOutputs[k][0])*(outputs[k][0]-intendedOutputs[k][0]);
        }
        meanSqrtSum.push_back(sqrtSum/outputs.size());

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