#include "Layer.hpp"
#include "..\Neuron\ActiveNeuron.hpp"

namespace NS {
    std::ostream& operator<<(std::ostream& out, const NS::Layer& layer)
    {
        layer.showLayer();
        return out;
    }

    void setWeights(const std::vector<std::vector<double>> t_Weights){
        for(int k = 0; k < m_weights.size(); k++){
            m_weights[k] = t_Weights[k];
        }
    }

    Layer::Layer(const unsigned int t_numberOfInputs , const unsigned int t_numberOfNeurons , const activationType t_activationFunction )
        : m_numberOfInputs  {t_numberOfInputs}
        , m_numberOfNeurons {t_numberOfNeurons}
    {
        m_bias.resize(m_numberOfNeurons, 1.0);
        m_outputs.resize(m_numberOfNeurons, 0.0);
        
        initFunction(t_activationFunction);
        initWeights();
    }
    
    std::vector<double> Layer::activateLayer(const std::vector<double> t_input){
        std::vector<double> temp(m_weights.size());
        for(int k = 0; k < m_weights.size(); k++){
            temp[k] = std::inner_product(m_weights[k].begin(), m_weights[k].end(), t_input.begin(), 0.0)+m_bias[k];
        }
        m_outputs = activateFunctionLayer(temp);
        return m_outputs;
    }gradiantLayer

    std::vector<double> Layer::gradiantWeightsLayer(const std::vector<double> t_inputDerivate, const std::vector<double> t_inputDerivate, const int t_learningRate){

        for(int k = 0; k < m_weights.size(); k++){
            for(int l = 0; l < m_weights.size(); l++){
                m_weights[k][l] -= t_learningRate*derivateLossFunction(m_outputs[k])*t_inputDerivate[k];
            }
        }
        m_outputs = activateFunctionLayer(temp);
        return m_outputs;
    }gradiantLayer
}