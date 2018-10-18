// main.cpp

#include <vector>
#include <iostream>
#include <math.h>

using namespace std;


// LINEAR MODULE
class LinearLayer
{
public:
    LinearLayer();
    float Forward(float x0) const;
    vector<float> Backward(float x0, float dedl) const;

    vector<float>& GetMutableWeights();
private:
    float m_bias;
    vector<float> m_weights;
};

LinearLayer::LinearLayer()
{
    // w_0 and w_1
    m_weights.push_back(-0.5);
    m_weights.push_back(2.5);

    // bias input is always 1.0
    m_bias = 1.0;
}

float LinearLayer::Forward(float x0) const
{
    // Linear combination of input, weight, and bias
    return (m_weights.at(0) * x0) + (m_weights.at(1) * m_bias);
}

vector<float> LinearLayer::Backward(float x0, float dedl) const
{
    // Linear derivative
    float dldw0 = x0;
    float dldw1 = m_bias;

    // Chain rule
    float dedw0 = dedl * dldw0;
    float dedw1 = dedl * dldw1;

    // Return gradient
    return {dedw0, dedw1};
}

vector<float>& LinearLayer::GetMutableWeights()
{
    return m_weights;
}

// END LINEAR MODULE

// ERROR MODULE
class SquaredError
{
public:
    SquaredError() {};
    float Forward(float output, float target) const;
    float Backward(float output, float input) const;
private:

};

float SquaredError::Forward(float output, float target) const
{
    float difference = target - output;
    return difference * difference; // square the difference
}

float SquaredError::Backward(float output, float input) const
{
    return -2.0 * (output - input);
}
// END ERROR MODULE

float CalcAverage(const vector<float>& vals)
{
    float sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        sum += vals.at(i);
    }
    return sum / ((float)vals.size());
}

vector<float> CalcAverageGrad(const vector<vector<float>>& grads)
{
    // Sum up each column
    vector<float> avgs(grads.at(0).size());
    for (int i = 0; i < grads.size(); ++i)
    {
        for (size_t j = 0; j < grads.at(i).size(); ++j)
        {
            avgs.at(j) += grads.at(i).at(j);
        }
    }

    // Divide by num rows for avg
    for (size_t i = 0; i < avgs.size(); ++i)
    {
        avgs.at(i) /= ((float)grads.size());
    }
    
    return avgs;
}

int main(int argc, char const *argv[])
{
    float learningRate = 0.01;
    vector<vector<float>> allData = {
        {2.0, 3.0},
        {4.0, 5.0}
    };

    // Define Modules
    LinearLayer linearLayer;
    SquaredError errorModule;

    // Number of times to iterate over dataset
    size_t numEpochs = 10;
    for (size_t epoch = 0; epoch < numEpochs; ++epoch)
    {
        cout << "----" << "START EPOCH " << epoch << "----" << endl;

        // Data struct to hold on to gradients for each training example
        // which we are going to average at the end
        vector<vector<float>> gradients;

        // Hold onto error for each training example, so we can average over
        // dataset
        vector<float> errors;

        // Iterate over dataset
        for (const vector<float>& dataPoint : allData)
        {
            cout << "--" << "START ITER " << "--" << endl;
            float x = dataPoint.at(0);
            float y = dataPoint.at(1);

            cout << "x = " << x << " y = " << y << endl;

            float prediction = linearLayer.Forward(x);
            cout << "prediction = " << prediction << endl;

            float error = errorModule.Forward(y, prediction);
            cout << "error = " << error << endl;

            // Derivative of error with respect to linear layer
            float dedl = errorModule.Backward(y, prediction);
            cout << "dedl = " << error << endl;

            // Compute gradient of error function wrt weights
            vector<float> gradient = linearLayer.Backward(x, dedl);
            cout << "gradient = " 
                 << "{" << gradient.at(0) << "," << gradient.at(1)
                 << "}" << endl;

            // Save off errors / gradients
            errors.push_back(error);
            gradients.push_back(gradient);

            cout << "--" << "END ITER " << "--" << endl;
        }

        // Compute average error
        float avgError = CalcAverage(errors);
        cout << "avgError = " << avgError << endl;

        // Compute average gradient
        vector<float> avgGrad = CalcAverageGrad(gradients);
        cout << "average gradient = " 
                 << "{" << avgGrad.at(0) << "," << avgGrad.at(1)
                 << "}" << endl;

        // Update weights given learning rate and average gradient
        vector<float>& weights = linearLayer.GetMutableWeights();
        weights.at(0) -= learningRate * avgGrad.at(0);
        weights.at(1) -= learningRate * avgGrad.at(1);

        cout << "new weights = " 
                 << "{" << weights.at(0) << "," << weights.at(1)
                 << "}" << endl;

        cout << "----" << "END EPOCH " << epoch << "----" << endl;
    }
  
    return 0;
}