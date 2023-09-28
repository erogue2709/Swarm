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
            const unsigned int m_numberOfNeurons;
            std::vector<std::vector<double>> m_weights;
            std::vector<double> m_bias;
            std::vector<double> m_outputs;
            std::vector<double> (*activateFunctionLayer)(const std::vector<double>);
            std::vector<double> (*activateDerivateLayer)(const std::vector<double>);


        public:

            Layer(const unsigned int t_numberOfInputs, const unsigned int t_numberOfNeurons, const activationType t_activationFunction );

            const unsigned int getNumberOfInputs() const { return m_numberOfInputs; }
            const unsigned int getNumberOfNeurons() const { return m_numberOfNeurons; }
            std::vector<std::vector<double>> getWeights() const { return m_weights; }
            std::vector<double> getBias() const { return m_bias; };
            std::vector<double> getOutputs() const { return m_outputs; };

            void setWeights(const std::vector<std::vector<double>> t_Weights);
            void setBias(const std::vector<double> t_bias);

            std::vector<double> activateLayer(const std::vector<double> t_input);
            std::vector<double> gradiantWeightsLayer(const std::vector<double> t_error);

            void initFunction(const activationType t_activationFunction)
            {
                switch (t_activationFunction)
                {
                case functionSigmoidNeuron:
                    activateFunctionLayer = &functionSigmoid;
                    activateDerivateLayer = &derivateSigmoid;
                    break;
                case functionReLuNeuron:
                    activateFunctionLayer = &functionReLu;
                    activateDerivateLayer = &derivateReLu;
                    break;
                }
            }

            void initWeights()
            {
                std::random_device seed;
                std::mt19937 generator (seed());
                std::uniform_real_distribution<double> distribution(-1.0,1.0);
                m_weights.resize(m_numberOfNeurons, std::vector<double>(m_numberOfInputs));
                for(int x = 0; x < m_weights.size(); x++){
                    for(int y = 0; y < m_weights[x].size(); y++){
                        m_weights[x][y] = distribution(generator);
                    }
                }
            }

            void showLayer() const
            {
                for( auto x : getOutputs() ){
                        std::cout << x << std::endl;
                }
                std::cout << std::endl;
                return;
            }
        
    };
}