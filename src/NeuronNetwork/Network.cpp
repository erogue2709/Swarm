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

    std::vector<double> Network::activateNetwork(const std::vector<double> t_input) const {
        m_networkLayers[0]->activateLayer(t_input);
        for( int i=1; i < m_networkLayers.size() ; i++ ){
            m_networkLayers[i]->activateLayer(m_networkLayers[i-1]->getOutputs());
        }
        return m_networkLayers.back()->getOutputs();
    }

    void Network::batchGDTrainning(std::vector<std::vector<double>> t_inputs
            , std::vector<std::vector<double>> t_intendedOutputs) const{
        //batch trainning
        bool trainningDone = false;
        std::vector<std::vector<double>> outputs;
        while(!trainningDone){
            trainningDone == true;
            for(int k=0; k < t_inputs.size(); k++ ){
                outputs.push_back(activateNetwork(t_inputs[k]));
            }
            for(auto k : NS::MSEFunction(outputs, t_intendedOutputs)){
                trainningDone = (trainningDone && k <= 0.05)?true:false;
            }
            if(!trainningDone){

            }
        }
    }

    std::ostream& operator<<(std::ostream& out, const NS::Network& Network){
        for( auto k : Network.m_networkLayers ){
            k->showLayer();
            std::cout << " " << std::endl;
        }
        return out;
    }
}