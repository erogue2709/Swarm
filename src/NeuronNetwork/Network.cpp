#include "..\NeuronLayer\Layer.hpp"
#include "Network.hpp"
#include <iostream>
#include <chrono>

namespace NS{

    Network::Network(NS::lossFunction m_lossFunction)
    {
        switch (m_lossFunction)
        {
        case crossEntropyLoss:
            //TODO need update later
            calculateError =  &NS::MSEFunction;
            break;
        case MSELoss:
            calculateError =  &MSEFunction;
            break;
        }
    };

    void Network::addLayer(const unsigned int t_numberOfInputs
                , const unsigned int t_numberOfOutputs
                , const activationType t_activationFunction)
    {
        if(t_numberOfInputs == 0 || t_numberOfOutputs == 0){
            std::cout << "Incorrect value, unable to add layer" << std::endl;
            return;
        }
        m_networkLayers.push_back(new NS::Layer(t_numberOfInputs, t_numberOfOutputs, t_activationFunction));
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
        std::cout << "Empty network, no activation possible" << std::endl;
        return {};
    }

    void Network::batchGDTrainning(std::vector<std::vector<double>> t_inputs
            , std::vector<std::vector<double>> t_targetOutputs) const{
        
        std::vector<std::vector<double>> error;
        std::vector<std::vector<double>> trainningOutputs;
        std::vector<std::vector<double>> outputGrag;

        std::chrono::time_point beginningTime = std::chrono::steady_clock::now();

        while( std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginningTime) < std::chrono::seconds(m_trainningTime)  ){
            for(int k=0; k < t_inputs.size(); k++ ){
                trainningOutputs.push_back(activateNetwork(t_inputs[k]));
            }
            error = calculateError(trainningOutputs, t_targetOutputs);

            outputGrag = t_targetOutputs;
            for(int layerIdx = m_networkLayers.size(); layerIdx != 0; layerIdx--){
                m_networkLayers[layerIdx]->gradiantWeightsLayer(outputGrag, m_learningRate)
            }
            // for m_networkLayers do gradient descent
        }
    }

    //need function to validate network (mse -> 1 output for regression else crossentropy for classification)

    std::ostream& operator<<(std::ostream& out, const NS::Network& Network){
        for( auto k : Network.m_networkLayers ){
            k->showLayer();
            std::cout << " " << std::endl;
        }
        return out;
    }
}