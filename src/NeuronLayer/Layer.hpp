#pragma once
#include "..\Neuron\ActiveNeuron.hpp"
#include <vector>
#include <functional>

namespace NS{
    
    class Layer
    {
        friend std::ostream& operator<<(std::ostream& out, const Layer& layer);

        private:
            const unsigned int m_numberOfInputs, m_numberOfOutputs;
            activationType m_activationType;
            std::vector<Neuron*> m_layerOfNeurons;
            std::vector<double> m_output;

        public:
            Layer(unsigned int t_numberOfInputs, unsigned int t_numberOfOutputs
                , activationType t_activationType);

            const unsigned int getNumberOfInputs() const { return m_numberOfInputs; }
            const unsigned int getNumberOfOutputs() const { return m_numberOfOutputs; }
            activationType getActivationType() const { return m_activationType; }
            std::vector<Neuron*> getLayerOfNeurons() const { return m_layerOfNeurons; }
            std::vector<double> getLayerOutputs() const { return m_output; };

            void activateLayer(std::vector<double> t_input);
            void updateLayerOutput();
            void showLayer() const;
        
    };
}