package com.company.lib;

public interface Layer {
    default double[] forward(double[] input) {
        throw new UnsupportedOperationException("forward has not been implemented");
    }

    default double[] backward(double[] input) {
        throw new UnsupportedOperationException("backward has not been implemented");
    }

    default void gradientDescent(double learningRate) {
        throw new UnsupportedOperationException("gradient descent has not been implemented");
    }
}