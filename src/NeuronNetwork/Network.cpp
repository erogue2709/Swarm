#include "..\NeuronLayer\Layer.hpp"
#include "Network.hpp"
#include <iostream>

namespace NS{

    std::ostream& operator<<(std::ostream& out, const NS::Network& Network){
        for( auto k : Network.m_networkLayers ){
            k->showLayer();
        }
        return out;
    }

    Network::Network(unsigned int t_inputSize, unsigned int t_numberOfOutputs
            , unsigned int t_numberOfHiddenLayers, activationType t_hiddenLayers
            , activationType t_outputLayer)
        : m_inputSize            {t_inputSize}
        , m_numberOfOutputs      {t_numberOfOutputs}
        , m_numberOfHiddenLayers {t_numberOfHiddenLayers}
    {
        m_activationTypes = {t_outputLayer};
        Layer* temp = new NS::Layer(m_inputSize, m_numberOfOutputs, t_outputLayer);
        m_networkLayers = {temp};

        for(int i=0; i<m_numberOfHiddenLayers; i++){
            m_activationTypes.insert(m_activationTypes.begin(), t_hiddenLayers);
            temp = new NS::Layer(m_inputSize, m_inputSize, t_hiddenLayers);
            m_networkLayers.insert(m_networkLayers.begin(), temp);
        }
    }
    void Network::activateNetwork(std::vector<double> t_input) const {
        m_networkLayers[0]->activateLayer(t_input);
        for( int i=0; i<m_numberOfHiddenLayers; i++ ){
            //m_networkLayers[i+1]->activateLayer(m_networkLayers[i]->getOutputs());
        }
    }

    void Network::trainNetwork(std::vector<double> t_input, std::vector<double> t_output) const{
        //de la merde, reduit le scoop t'as pas besoin de tout ca pouir du swarm sigmo+meilleur traitement/mapping des données
    }
}