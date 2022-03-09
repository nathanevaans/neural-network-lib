package com.company.examples;

import com.company.examples.util.Pair;
import com.company.lib.NetworkType;
import com.company.lib.NeuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class XOR {
    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(NetworkType.DENSE, 0.3);
        neuralNetwork.addDenseLayer(3);
        neuralNetwork.addDenseLayer(2);
        neuralNetwork.addDenseLayer(3);
        neuralNetwork.addDenseLayer(1);

        ArrayList<Pair<double[], double[]>> trainingData = new ArrayList<>() {{
            add(new Pair<>(new double[]{0, 0}, new double[]{0}));
            add(new Pair<>(new double[]{0, 1}, new double[]{1}));
            add(new Pair<>(new double[]{1, 0}, new double[]{1}));
            add(new Pair<>(new double[]{1, 1}, new double[]{0}));
        }};

        for (int i = 0; i < 10000; i++) {
            for (Pair<double[], double[]> trainingDatum : trainingData) {
                double[] y = neuralNetwork.forward(trainingDatum.first());
                double error = mse(y, trainingDatum.second());
                System.out.println("error: " + error);
                double[] dy = msePrime(y, trainingDatum.second());
                neuralNetwork.backward(dy);
                neuralNetwork.gradientDescent();
            }
            Collections.shuffle(trainingData);
        }

        trainingData.forEach(p -> System.out.println(Arrays.toString(p.first()) + " -> " + Arrays.toString(neuralNetwork.forward(p.second()))));
    }

    private static double mse(double[] networkOutput, double[] expected) {
        double output = 0;
        for (int i = 0; i < networkOutput.length; i++) {
            double difference = expected[i] - networkOutput[i];
            output += difference * difference;
        }
        return output;
    }

    private static double[] msePrime(double[] networkOutput, double[] expected) {
        double[] output = new double[networkOutput.length];
        for (int i = 0; i < networkOutput.length; i++) {
            output[i] = 2 * (networkOutput[i] - expected[i]) / networkOutput.length;
        }
        return output;
    }
}