#pragma once
#include <math.h>
#include "Neuron.hpp"

namespace NS {

    enum activationType {
        functionSigmoidNeuron,
        functionReLuNeuron
    };

    class SigmoidNeuron : public NS::Neuron{

        public:
            using Neuron::Neuron;

            virtual void activationFunction(std::vector<double> t_input) override{
                double pA = preActivation(t_input);

                //fast Sigmoid from https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
                setOutput( pA/(1.0+fabs(pA)) );
            }

            virtual double derivativeFunction() override{
                return (getOutput()*(1-getOutput()));
            }
    };

    class ReLuNeuron : public NS::Neuron{

        public:
            using Neuron::Neuron;

            virtual void activationFunction(std::vector<double> t_input) override{
                double preActivationOutput = preActivation(t_input);
                setOutput((preActivationOutput>=0)?preActivationOutput:0.0);
            }

            virtual double derivativeFunction() override{
                return ((getOutput()>=0)?1.0:0.0);
            }
    };
}