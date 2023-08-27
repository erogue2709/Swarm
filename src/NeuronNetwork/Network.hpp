#pragma once
#include "..\NeuronLayer\Layer.hpp"
#include <vector>

namespace NS{

    class Network
    {
        friend std::ostream& operator<<(std::ostream& out, const Network& Network);

        private:
            const double m_learningRate = 0.015; 
            int m_trainningTime = 30; //do batchTrainning for m_trainningTime seconds
            NS::lossFunction m_lossFunction = MSELoss;
            std::vector<Layer*> m_networkLayers;
        
        public:
            Network();
            //is virtual function for lossFunction better/faster

            void addLayer(const unsigned int t_numberOfInputs
                , const unsigned int t_numberOfNeurons
                , const activationType t_activationFunction
                );

            unsigned int lastLayerSize();

            bool trainningDone();

            std::vector<double> activateNetwork(const std::vector<double> t_input) const;

            void batchGDTrainning(std::vector<std::vector<double>> t_input
                    , std::vector<std::vector<double>> t_intendedOutput) const;
            
            void miniBatchGDTrainning(std::vector<std::vector<double>> t_input
                    , std::vector<std::vector<double>> t_intendedOutput) const;

            void stochasticGDTrainning(std::vector<std::vector<double>> t_input
                    , std::vector<std::vector<double>> t_intendedOutput) const;

            std::vector<double> calculateError(
                std::vector<std::vector<double>> t_trainningOutputs
            , std::vector<std::vector<double>> t_intendedOutputs);
    };
}