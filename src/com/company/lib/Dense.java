package com.company.lib;

import com.company.lib.util.Maths;

public class Dense implements Layer {

    private double[] input;
    private double[] output;
    private int outputX;

    private double[] weights; // dim: inputX x outputX

    private double[] bias;

    private double[] dw;
    private double[] db; // db = dy
    private double[] dx;

    private boolean init;

    public Dense(int outputSize) {
        init = false;
        outputX = outputSize;
    }

    private void initLayer() {
        init = true;
        weights = new double[input.length * outputX];
        double min = -5;
        double max = 5;
        double intermediate;
        for (int i = 0; i < weights.length; i++) {
            intermediate = Math.random() / Math.nextDown(1.0);
            weights[i] = min * (1 - intermediate) + max * intermediate;

        }

        bias = new double[outputX];
        for (int i = 0; i < bias.length; i++) {
            intermediate = Math.random() / Math.nextDown(1.0);
            bias[i] = min * (1 - intermediate) + max * intermediate;
        }
    }

    @Override
    public double[] forward(double[] input) {
        this.input = input;
        if (!init) initLayer();

        double[] intermediate = Maths.multiply(input, weights, outputX);
        output = Maths.add(intermediate, bias);
        return output.clone();
    }

    @Override
    public double[] backward(double[] input) {
        // db = input
        db = input;

        // dw = this.input^T . input
        // no manual transpose used because the multiply function handles it as it is implied
        // by trying to multiply 2 vectors, the first argument is 'transposed'
        dw = Maths.multiply(this.input, input);

        // dx = input . weights^T
        dx = Maths.multiply(input, Maths.transpose(weights, this.input.length, outputX), this.input.length);
        return dx.clone();
    }

    @Override
    public void gradientDescent(double learningRate) {
        // w = w - lr * dw
        weights = Maths.add(weights, Maths.multiply(-learningRate, dw));
        // b = b - lr * db
        bias = Maths.add(bias, Maths.multiply(-learningRate, db));
    }

    public double[] getInput() {
        return input;
    }

    public double[] getOutput() {
        return output;
    }
}