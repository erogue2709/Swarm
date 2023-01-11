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
            NS::lossFunction m_lossFunction = MSE;
            std::vector<Layer*> m_networkLayers;
        
        public:
            Network(unsigned int t_inputSize
            , unsigned int t_numberOfHiddenLayers
            , activationType t_hiddenLayers
            , unsigned int t_outputSize
            , activationType t_outputLayer
            );

            void activateNetwork(const std::vector<double> t_input) const;
            void trainNetwork(std::vector<double> t_input, std::vector<double> t_output) const;
    };
}