#pragma once
#include "Neuron.hpp"

namespace NS {

    enum activationType {
        functionHeavisideNeuron,
        functionReLuNeuron
    };

    class HeavisideNeuron : public NS::Neuron{

        public:
            using Neuron::Neuron;

            virtual void activationFunction(std::vector<double> t_input) override{
                double preActivationOutput = preActivation(t_input);
                setOutput((preActivationOutput>=0)?1.0:-1.0);
            }
    };

    class ReLuNeuron : public NS::Neuron{

        public:
            using Neuron::Neuron;

            virtual void activationFunction(std::vector<double> t_input) override{
                double preActivationOutput = preActivation(t_input);
                setOutput((preActivationOutput>=0)?preActivationOutput:0.0);
            }
    };
}