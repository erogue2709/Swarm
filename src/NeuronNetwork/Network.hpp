#pragma once
#include "..\NeuronLayer\Layer.hpp"
#include <vector>

namespace NS{

    enum lossFunction {
         crossEntropyLoss
    };

    class Network
    {
        friend std::ostream& operator<<(std::ostream& out, const Network& Network);

        private:
            const unsigned int m_inputSize, m_numberOfOutputs, m_numberOfHiddenLayers;
            std::vector<activationType> m_activationTypes;
            std::vector<Layer*> m_networkLayers;
            NS::lossFunction m_lossFunction = crossEntropyLoss;
        
        public:
            Network(unsigned int t_inputSize, unsigned int t_numberOfOutputs
                , unsigned int t_numberOfHiddenLayers, activationType t_hiddenLayers
                , activationType t_outputLayer);

            void activateNetwork(std::vector<double> t_input) const;
            void trainNetwork(std::vector<double> t_input, std::vector<double> t_output) const;
    };

}
