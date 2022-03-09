package com.company.examples.MNIST.util;

import static java.lang.Math.min;

public class Sample {

    private static final double[][] EXPECTED_VALUES = {
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    };

    private final double[] data;
    private final int label;

    public Sample(double[] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    public double[] getExpectedValue() {
        return EXPECTED_VALUES[label];
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        String line = " ---------------------------- ";
        sb.append(line);
        int cnt = 0;
        for (int r = 0; r < 28; r++) {
            sb.append("\n|");
            for (int c = 0; c < 28; c++) {
                sb.append(toChar(data[cnt++]));
            }
            sb.append("|");
        }
        sb.append("\n").append(line).append("\n");
        return sb.toString();
    }

    private char toChar(double val) {
        return " .:-=+*#%@".charAt(min((int) (val * 10), 9));
    }
}
