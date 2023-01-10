#include "..\Neuron\Neuron.hpp"
#include "Layer.hpp"
#include <vector>
#include <functional>
#include <iostream>


namespace NS {

    std::ostream& operator<<(std::ostream& out, const NS::Layer& layer){
        layer.showLayer();
        return out;
    }

    Layer::Layer(unsigned int t_numberOfInputs, unsigned int t_numberOfOutputs
            , activationType t_activationType)
        : m_numberOfInputs  {t_numberOfInputs}
        , m_numberOfOutputs {t_numberOfOutputs}
        , m_activationType  {t_activationType}
    {
        m_output.resize(m_numberOfOutputs);
        m_layerOfNeurons.resize(m_numberOfOutputs);
        switch (m_activationType)
        {
        case functionSigmoidNeuron:
            for(int i=0; i < m_numberOfOutputs; i++){
                SigmoidNeuron* newNeuron = new SigmoidNeuron(m_numberOfInputs);
                m_layerOfNeurons[i] = newNeuron;
            }
            break;
        case functionReLuNeuron:
            for(int i=0; i < m_numberOfOutputs; i++){
                //free maybe? one day? learn destructor lmao
                ReLuNeuron* newNeuron = new ReLuNeuron(m_numberOfInputs);
                m_layerOfNeurons[i] = newNeuron;
            }
            break;
        }
    }
    
    void Layer::activateLayer(std::vector<double> t_input){
        for(auto k : m_layerOfNeurons){
            k->activationFunction(t_input);
        }
        updateLayerOutput();
    }

    void Layer::updateLayerOutput() {
        for( int i=0; i<m_numberOfOutputs; i++ ){
            m_output[i] = m_layerOfNeurons[i]->getOutput();
        }
    }

    void Layer::showLayer() const{
        std::cout << "ActivationType = "  << getActivationType() << std::endl;
        std::cout << "NumberOfInputs = "  << getNumberOfInputs() << std::endl;
        std::cout << "NumberOfOutputs = " << getNumberOfOutputs() << std::endl;
        std::cout << "Neurons' Outputs :" << std::endl;
        
        for( auto k : getLayerOfNeurons() ){
            std::cout << '\t' << k->getOutput() << std::endl;
        }
        return;
    }
}