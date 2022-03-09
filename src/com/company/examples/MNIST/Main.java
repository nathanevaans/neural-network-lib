package com.company.examples.MNIST;

import com.company.examples.MNIST.util.FileReader;
import com.company.examples.MNIST.util.Sample;

import com.company.lib.NetworkType;
import com.company.lib.NeuralNetwork;

import java.util.Collections;
import java.util.List;

public class Main {

    private static final List<Sample> trainingData = FileReader.loadImageData("train");
    private static final List<Sample> testData = FileReader.loadImageData("t10k");

    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(NetworkType.CONV, 0.4);

        // TODO: find why error in getMatrixElement and fix it
        neuralNetwork.addConvolutionalLayer(5, 5, 5);
        neuralNetwork.addConvolutionalLayer(3, 2, 2);
        neuralNetwork.addDenseLayer(15);
        neuralNetwork.addDenseLayer(10);

        train(neuralNetwork, 1);
        test(neuralNetwork);
    }

    private static void train(NeuralNetwork neuralNetwork, int epochs) {
        double error = 0;
        for (int i = 0, sampleCount = 0; i < epochs; i++) {
            for (int j = 0; j < trainingData.size(); j++, sampleCount++) {
                Sample currentSample = trainingData.get(j);

                double[] output = neuralNetwork.forward(currentSample.getData(), 1, 28, 28);

                error = (error * sampleCount + mse(output, currentSample.getExpectedValue())) / (sampleCount + 1);
                double[] errorGradient = msePrime(output, currentSample.getExpectedValue());

                neuralNetwork.backward(errorGradient);
                neuralNetwork.gradientDescent();

                if ((sampleCount) % 100 == 0 && sampleCount != 0) {
                    System.out.println("epoch: " + (i + 1) + " | sample: " + j + " | error: " + error);
                }
            }
            Collections.shuffle(trainingData);
        }
    }

    private static void test(NeuralNetwork neuralNetwork) {
        double error;
        double averageError = 0;
        double[] output;
        int correct = 0;
        int sampleCount = 0;
        Collections.shuffle(testData);
        for (Sample digit : testData) {
            output = neuralNetwork.forward(digit.getData(), 1, 28, 28);
            error = mse(output, digit.getExpectedValue());
            averageError = (averageError * sampleCount + error) / (sampleCount + 1);
            sampleCount++;

            int label = outputToLabel(output);
            if (label == digit.getLabel()) correct++;

            System.out.println(digit);
            System.out.println("output: " + outputToLabel(output) + " | expected: " + digit.getLabel() + " | error: " + error);
        }
        System.out.println("AVERAGE ERROR: " + averageError);
        System.out.println(correct + "/" + testData.size() + " correct");
    }

    private static int outputToLabel(double[] output) {
        int index = 0;
        double value = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > value) {
                value = output[i];
                index = i;
            }
        }
        return index;
    }

    private static double mse(double[] networkOutput, double[] expected) {
        double output = 0;
        for (int i = 0; i < networkOutput.length; i++) {
            double difference = expected[i] - networkOutput[i];
            output += difference * difference;
        }
        return output / networkOutput.length;
    }

    private static double[] msePrime(double[] networkOutput, double[] expected) {
        double[] output = new double[networkOutput.length];
        for (int i = 0; i < networkOutput.length; i++) {
            output[i] = 2 * (networkOutput[i] - expected[i]) / networkOutput.length;
        }
        return output;
    }
}
