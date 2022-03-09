package com.company.lib;

import com.company.lib.util.Maths;

public class Convolution {

    private double[] input;
    private int inputZ, inputY, inputX;
    private int inputMLength;

    private double[] output;
    private final int outputZ;
    private int outputY;
    private int outputX;
    private int outputMLength;

    // [inputZ][outputZ][kernelY][kernelX]
    private double[] kernels;
    private final int kernelY;
    private final int kernelX;
    private final int kernelMLength;

    // [outputZ, outputY, outputX]
    private double[] bias;


    private double[] dk;
    private double[] db; // db = dy
    private double[] dx;

    private boolean init;

    public Convolution(int outputCount, int kernelX, int kernelY) {
        this.outputZ = outputCount;
        this.kernelX = kernelX;
        this.kernelY = kernelY;
        kernelMLength = kernelY * kernelX;
    }

    private void initLayer() {
        init = true;

        kernels = new double[inputZ * outputZ * kernelY * kernelX];
        double min = -5;
        double max = 5;
        double intermediate;
        for (int i = 0; i < kernels.length; i++) {
            intermediate = Math.random() / Math.nextDown(1.0);
            kernels[i] = min * (1 - intermediate) + max * intermediate;
        }

        outputX = inputX - kernelX + 1;
        outputY = inputY - kernelY + 1;
        outputMLength = outputY * outputX;
        bias = new double[outputZ * outputY * outputX];
        for (int i = 0; i < bias.length; i++) {
            intermediate = Math.random() / Math.nextDown(1.0);
            bias[i] = min * (1 - intermediate) + max * intermediate;
        }
    }

    public double[] forward(double[] input, int inputZ, int inputY, int inputX) {
        this.inputZ = inputZ;
        this.inputY = inputY;
        this.inputX = inputX;
        inputMLength = inputY * inputX;
        this.input = input;
        if (!init) initLayer();
        output = new double[outputZ * outputY * outputX];

        double[] intermediate;
        for (int i = 0; i < outputZ; i++) {
            for (int j = 0; j < inputZ; j++) {
                intermediate = Maths.crossCorrelationValid(
                        Maths.getMatrixElementAt(input, inputY, inputX, j),
                        inputX,
                        Maths.getMatrixElementAt(kernels, inputZ, kernelY, kernelX, j, i),
                        kernelX,
                        kernelMLength,
                        outputX,
                        outputMLength
                );
                intermediate = Maths.add(Maths.getMatrixElementAt(output, outputY, outputX, i), intermediate);
                System.arraycopy(intermediate, 0, output, (i * intermediate.length), intermediate.length);
            }
        }
        output = Maths.add(output, bias);
        return output.clone();
    }

    public double[] backward(double[] input) {
        db = input;

        // dk = this.input * input, valid cross-correlation
        // dk[j][i] = x[j] * dy[i]
        dk = new double[inputZ * outputZ * kernelY * kernelX];
        double[] intermediate;
        for (int i = 0; i < outputZ; i++) {
            for (int j = 0; j < inputZ; j++) {
                intermediate = Maths.crossCorrelationValid(
                        Maths.getMatrixElementAt(this.input, inputY, inputX, j),
                        inputX,
                        Maths.getMatrixElementAt(input, outputY, outputX, i),
                        outputX,
                        outputMLength,
                        kernelX,
                        kernelMLength
                );
                System.arraycopy(intermediate, 0, dk, kernelMLength * (j * inputZ + i), kernelMLength);
            }
        }

        // dx = input * k, full convolution
        // dx[i] = sum {j = [0, outputSize)} dy[j] * k[i][j]
        dx = new double[inputZ * inputY * inputX];
        for (int i = 0; i < inputZ; i++) {
            for (int j = 0; j < outputZ; j++) {
                intermediate = Maths.convolutionFull(
                        Maths.getMatrixElementAt(input, outputY, outputX, j),
                        outputY,
                        outputX,
                        Maths.getMatrixElementAt(kernels, inputZ, kernelY, kernelX, i, j),
                        kernelY,
                        kernelX,
                        kernelMLength,
                        inputX,
                        inputMLength);
                intermediate = Maths.add(Maths.getMatrixElementAt(dx, inputY, inputX, i), intermediate);
                System.arraycopy(intermediate, 0, dx, (i * inputY * inputX), inputMLength);
            }
        }
        return dx.clone();
    }

    public void gradientDescent(double learningRate) {
        // k = k - lr * dk
        kernels = Maths.add(kernels, Maths.multiply(-learningRate, dk));
        // b = b - lr * db
        bias = Maths.add(bias, Maths.multiply(-learningRate, db));
    }

    public double[] getInput() {
        return input;
    }

    public double[] getOutput() {
        return output;
    }

    public int getOutputX() {
        return outputX;
    }

    public int getOutputY() {
        return outputY;
    }

    public int getOutputZ() {
        return outputZ;
    }
}