package com.company.lib;

import com.company.lib.util.Maths;

public enum Activation {
    RELU, LEAKY_RELU, SIGMOID, TANH;

    public double[] forward(double[] input) {
        return switch (this) {
            case RELU -> relu(input);
            case LEAKY_RELU -> leakyRelu(input);
            case SIGMOID -> sigmoid(input);
            case TANH -> tanh(input);
        };
    }

    public double[] backward(double[] gradientOutput, double[] activationInput) {
        double[] prime;
        switch (this) {
            case RELU -> prime = reluPrime(activationInput);
            case LEAKY_RELU -> prime = leakyReluPrime(activationInput);
            case SIGMOID -> prime = sigmoidPrime(activationInput);
            case TANH -> prime = tanhPrime(activationInput);
            default -> throw new IllegalStateException("Unexpected value: " + this);
        }
        return Maths.hadamard(gradientOutput, prime);
    }

    // RELU
    private double[] relu(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }

    private double[] reluPrime(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] > 0 ? 1 : 0;
        }
        return output;
    }


    // LEAKY_RELU
    private double[] leakyRelu(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] < 0 ? 0.83333 * input[i] : input[i];
        }
        return output;
    }

    private double[] leakyReluPrime(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] < 0 ? 0.83333 : 1;
        }
        return output;
    }


    // SIGMOID
    private double[] sigmoid(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = 1 / (1 + Math.exp(-input[i]));
        }
        return output;
    }

    private double[] sigmoidPrime(double[] input) {
        double[] intermediate = sigmoid(input);
        for (int i = 0; i < intermediate.length; i++) {
            intermediate[i] *= (1 - intermediate[i]);
        }
        return intermediate;
    }

    // TANH
    private double[] tanh(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.tanh(input[i]);
        }
        return output;
    }

    private double[] tanhPrime(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = 1 - Math.pow(Math.tanh(input[i]), 2);
        }
        return output;
    }
}