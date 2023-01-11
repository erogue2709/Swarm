#include "..\NeuronLayer\Layer.hpp"
#include "Network.hpp"
#include <iostream>

namespace NS{

    Network::Network(unsigned int t_inputSize
                , unsigned int t_numberOfHiddenLayers
                , activationType t_hiddenLayers
                , unsigned int t_outputSize
                , activationType t_outputLayer
                )
            : m_inputSize {t_inputSize}
            , m_outputSize {t_outputSize}
            , m_numberOfHiddenLayers {t_numberOfHiddenLayers}
    {
        for(int i=0; i<m_numberOfHiddenLayers; i++){
            m_networkLayers.push_back(new NS::Layer(m_inputSize, m_inputSize, t_hiddenLayers));
        }
        m_networkLayers.push_back(new NS::Layer(m_inputSize, t_outputSize, t_outputLayer));
    }

    void Network::activateNetwork(const std::vector<double> t_input) const {
        m_networkLayers[0]->activateLayer(t_input);
        for( int i=1; i < m_networkLayers.size() ; i++ ){
            m_networkLayers[i]->activateLayer(m_networkLayers[i-1]->getOutputs());
        }
    }

    void Network::trainNetwork(std::vector<double> t_input, std::vector<double> t_output) const{
        //de la merde, reduit le scoop t'as pas besoin de tout ca pouir du swarm sigmo+meilleur traitement/mapping des données
    }

    std::ostream& operator<<(std::ostream& out, const NS::Network& Network){
        for( auto k : Network.m_networkLayers ){
            k->showLayer();
            std::cout << " " << std::endl;
        }
        return out;
    }
}