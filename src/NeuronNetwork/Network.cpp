#include "..\NeuronLayer\Layer.hpp"
#include "Network.hpp"
#include <iostream>

namespace NS{

    Network::Network(){};

    void Network::addLayer(const unsigned int t_numberOfInputs
                , const unsigned int t_numberOfNeurons
                , const activationType t_activationFunction)
    {
        if(t_numberOfInputs == 0 || t_numberOfNeurons == 0){
            std::cout << "Incorrect value, unable to add layer" << std::endl;
            return;
        }
        m_networkLayers.push_back(new NS::Layer(t_numberOfInputs, t_numberOfNeurons, t_activationFunction));
    }

    unsigned int Network::lastLayerSize(){
        if(!m_networkLayers.empty()){
            return m_networkLayers.back()->getNumberOfNeurons();
        }
        return 0;
    }

    std::vector<double> Network::activateNetwork(const std::vector<double> t_input) const {
        if(!m_networkLayers.empty()){
            if(m_networkLayers[0]->getNumberOfInputs() == t_input.size()){
                m_networkLayers[0]->activateLayer(t_input);
                for( int i=1; i < m_networkLayers.size() ; i++ ){
                    m_networkLayers[i]->activateLayer(m_networkLayers[i-1]->getOutputs());
                }
            }else{
                std::cout << "Incorrect input size, unable to activate network" << std::endl;
            }
            return m_networkLayers.back()->getOutputs();
        }
        std::cout << "Empty network, no activatiob possible" << std::endl;
        return {};
    }

    void Network::batchGDTrainning(std::vector<std::vector<double>> t_inputs
            , std::vector<std::vector<double>> t_intendedOutputs) const{
                
        std::vector<std::vector<double>> outputs;
        while(!trainningDone()){
            for(int k=0; k < t_inputs.size(); k++ ){
                outputs.push_back(activateNetwork(t_inputs[k]));
            }
        }
    }

    bool Network::trainningDone(){
        return true;
    }

    std::ostream& operator<<(std::ostream& out, const NS::Network& Network){
        for( auto k : Network.m_networkLayers ){
            k->showLayer();
            std::cout << " " << std::endl;
        }
        return out;
    }
}