#pragma once
#include "..\NeuronLayer\Layer.hpp"
#include <vector>

namespace NS{

    class Network
    {
        friend std::ostream& operator<<(std::ostream& out, const Network& Network);

        private:
            const unsigned int m_inputSize;
            const unsigned int m_outputSize;
            const unsigned int m_numberOfHiddenLayers;
            const double m_learningRate = 0.015; 
            NS::lossFunction m_lossFunction = MSELoss;
            std::vector<Layer*> m_networkLayers;
        
        public:
            Network(unsigned int t_inputSize
            , unsigned int t_numberOfHiddenLayers
            , activationType t_hiddenLayers
            , unsigned int t_outputSize
            , activationType t_outputLayer
            );
            //change constructor to work only with t_inputSize
            //make addLayer function
            //is vurtial function for lossFunction better/faster

            std::vector<double> activateNetwork(const std::vector<double> t_input) const;

            void batchGDTrainning(std::vector<std::vector<double>> t_input
                    , std::vector<std::vector<double>> t_intendedOutput) const;
            
            void miniBatchGDTrainning(std::vector<std::vector<double>> t_input
                    , std::vector<std::vector<double>> t_intendedOutput) const;

            void stochasticGDTrainning(std::vector<std::vector<double>> t_input
                    , std::vector<std::vector<double>> t_intendedOutput) const;
    };
}