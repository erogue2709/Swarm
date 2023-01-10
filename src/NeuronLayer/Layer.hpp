#pragma once
#include "..\Neuron\ActiveNeuron.hpp"
#include <vector>
#include <random>
#include <iostream>

namespace NS{
    
    class Layer
    {
        friend std::ostream& operator<<(std::ostream& out, const Layer& layer);

        private:

            const unsigned int m_numberOfInputs;
            const unsigned int m_numberNeurons;
            const activationType m_activationFunction;

            std::vector<std::vector<double>> m_weights;
            std::vector<double> m_bias;
            std::vector<double> m_outputs;

        public:

            Layer(const unsigned int t_numberOfInputs
                , const unsigned int t_numberOfNeurons
                , const activationType t_activationFunction
                );

            const unsigned int getNumberOfInputs() const { return m_numberOfInputs; }
            const unsigned int getNumberOfNeurons() const { return m_numberNeurons; }
            activationType getActivationFunction() const { return m_activationFunction; }

            std::vector<std::vector<double>> getWeights() const { return m_weights; }
            std::vector<double> getBias() const { return m_bias; };
            std::vector<double> getOutputs() const { return m_outputs; };

            void setWeights(const std::vector<std::vector<double>> t_Weights);
            void setBias(const std::vector<double> t_bias);

            std::vector<double> activateLayer(const std::vector<double> t_input);
            void showLayer() const;
        
    };
}