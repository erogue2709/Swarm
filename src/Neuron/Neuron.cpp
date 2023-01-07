#include "Neuron.hpp"
#include <iostream>
#include <random>
#include <vector>

namespace NS {

    std::ostream& operator<<(std::ostream& out, const NS::Neuron& neuron){
        out << "Bias = " << neuron.getBias() << std::endl;
        out << "Weights (" << neuron.getNumberOfWeights() << "):" << std::endl;
        
        for( auto const weight: neuron.getWeights() ){
            out << '\t' << weight << std::endl;
        }
        out << "Output = " << neuron.getOutput() << std::endl;
        return out;
    }

    void Neuron::initWeight(){
        
        auto seed = std::default_random_engine{std::random_device()()};
        std::default_random_engine generator (seed);
        std::uniform_real_distribution<double> distribution(-10.0,10.0);

        for( int i=0 ; i<m_numberOfWeights; i++ ){
            m_weights[i] = distribution(generator);
        }
    }

    double Neuron::preActivation(std::vector<double> t_input){

        double preActivationOutput = m_bias;
        for(int i=0; i < m_numberOfWeights; i++){
            preActivationOutput += t_input[i] * m_weights[i];
        }
        return preActivationOutput;
    }
    
}
