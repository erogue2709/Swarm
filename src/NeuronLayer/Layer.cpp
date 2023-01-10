#include "Layer.hpp"
#include "..\Neuron\ActiveNeuron.hpp"

namespace NS {

    Layer::Layer(const unsigned int t_numberOfInputs
                , const unsigned int t_numberOfNeurons
                , const activationType t_activationFunction
                )
        : m_numberOfInputs  {t_numberOfInputs}
        , m_numberNeurons {t_numberOfNeurons}
        , m_activationFunction  {t_activationFunction}{

        m_bias.resize(m_numberNeurons, 1.0);
        m_outputs.resize(m_numberNeurons, 0.0);
        
        auto seed = std::default_random_engine{std::random_device()()};
        std::default_random_engine generator (seed);
        std::uniform_real_distribution<double> distribution(-10.0,10.0);
        m_weights.resize(t_numberOfNeurons, std::vector<double>(t_numberOfInputs));
        for(int x = 0; x < m_weights.size(); x++){
            for(int y = 0; y < m_weights[x].size(); y++){
                m_weights[x][y] = distribution(generator);
            }
        }
    }
    
    std::vector<double> Layer::activateLayer(const std::vector<double> t_input){
        std::vector<double> temp(m_weights.size());
        for(int k = 0; k < m_weights.size(); k++){
            temp[k] = std::inner_product(m_weights[k].begin(), m_weights[k].end(), t_input.begin(), 0.0)+m_bias[k];
            std::cout << temp[k] << " ";
        }
        std::cout << std::ends;
        m_outputs = NS::activeFunction(m_activationFunction, temp);
        return m_outputs;
    }

    std::ostream& operator<<(std::ostream& out, const NS::Layer& layer){
        layer.showLayer();
        return out;
    }

    void Layer::showLayer() const{
        std::cout << std::endl;
        std::cout << "ActivationType = "  << getActivationFunction() << std::endl;
        std::cout << "NumberOfInputs = "  << getNumberOfInputs() << std::endl;
        std::cout << "NumberOfNeurons = " << getNumberOfNeurons() << std::endl;
        std::cout << "Neurons' m_outputs :" << std::endl;
        
        for( auto x : getOutputs() ){
                std::cout << x << std::endl;
        }
        std::cout << std::endl;
        return;
    }
}