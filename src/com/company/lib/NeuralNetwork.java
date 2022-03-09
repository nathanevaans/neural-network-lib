package com.company.lib;

import com.company.lib.util.Reversed;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private List<Convolution> convolutionalLayers;
    private final List<Dense> denseLayers;

    private final NetworkType type;

    private Activation convActivation;
    private Activation denseActivation;

    private final double learningRate;

    public NeuralNetwork(NetworkType type, double learningRate) {
        this.type = type;
        denseLayers = new ArrayList<>();
        denseActivation = Activation.SIGMOID;

        this.learningRate = learningRate;

        if (type == NetworkType.CONV) {
            convolutionalLayers = new ArrayList<>();
            convActivation = Activation.RELU;
        }
    }

    public double[] forward(double[][][] input) {
        double[] flat = flatten(input);
        return forward(flat, input.length, input[0].length, input[0][0].length);
    }

    public double[] forward(double[] input, int inputZ, int inputY, int inputX) {
        for (Convolution c : convolutionalLayers) {
            input = c.forward(input, inputZ, inputY, inputX);
            input = convActivation.forward(input);

            inputZ = c.getOutputZ();
            inputY = c.getOutputY();
            inputX = c.getOutputX();
        }

        double[] output = new double[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return forward(output);
    }

    public double[] forward(double[] input) {
        for (Dense d : denseLayers) {
            input = d.forward(input);
            input = denseActivation.forward(input);
        }
        return input;
    }

    public void backward(double[] input) {
        double[] output = input;
        for (Dense d : Reversed.reversed(denseLayers)) {
            output = denseActivation.backward(output, d.getOutput());
            output = d.backward(output);
        }

        if (type == NetworkType.CONV) {
            for (Convolution c : Reversed.reversed(convolutionalLayers)) {
                output = convActivation.backward(output, c.getOutput());
                output = c.backward(output);
            }
        }
    }

    public void gradientDescent() {
        for (Dense d : denseLayers) {
            d.gradientDescent(learningRate);
        }

        if (type == NetworkType.CONV) {
            for (Convolution c : convolutionalLayers) {
                c.gradientDescent(learningRate);
            }
        }
    }

    public void addConvolutionalLayer(int outputCount, int kernelX, int kernelY) {
        convolutionalLayers.add(new Convolution(outputCount, kernelX, kernelY));
    }

    public void addDenseLayer(int outputSize) {
        denseLayers.add(new Dense(outputSize));
    }

    public void setConvolutionalActivation(Activation activation) {
        convActivation = activation;
    }

    public void setDenseActivation(Activation activation) {
        denseActivation = activation;
    }

    private double[] flatten(double[][][] input) {
        double[] output = new double[input.length * input[0].length * input[0][0].length];
        for (int i = 0, resultIndex = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                for (int k = 0; k < input[0][0].length; k++, resultIndex++) {
                    output[resultIndex] = input[i][j][k];
                }
            }
        }
        return output;
    }
}