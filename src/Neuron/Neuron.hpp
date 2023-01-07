#pragma once
#include <ostream>
#include <vector>

namespace NS{
    
    class Neuron
    {
        friend std::ostream& operator<<(std::ostream& out, const Neuron& neuron);

        protected:
            std::vector<double> m_weights;
            double preActivation(std::vector<double> t_input);

        private:
            void initWeight();

            const unsigned int m_numberOfWeights;
            double m_bias{1.0};
            double m_output{0.0};

        public:
            Neuron(unsigned int t_numberOfInputs): m_numberOfWeights{t_numberOfInputs}
            {
                m_weights.resize(m_numberOfWeights);
                initWeight();
            }
            
            void setBias(double t_bias){ m_bias = t_bias; }
            void setOutput(double t_output) { m_output = t_output; }

            int getNumberOfWeights() const { return m_numberOfWeights; }
            double getBias() const { return m_bias; }
            std::vector<double> getWeights() const { return m_weights; }
            double getOutput() const { return m_output; }

            
            virtual void activationFunction(std::vector<double> t_input) =0;
    };
}