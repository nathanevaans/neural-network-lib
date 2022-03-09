package com.company.lib.util;

public class Maths {

    public static double[] multiply(double[] v0, double[] v1) {
        double[] output = new double[v0.length * v1.length];
        for (int i = 0; i < v0.length; i++) {
            for (int j = 0; j < v1.length; j++) {
                output[(i * v1.length) + j] = v0[i] * v1[j];
            }
        }
        return output;
    }

    public static double[] multiply(double[] v, double[] m, int mX) {
        double[] output = new double[mX];
        for (int i = 0; i < mX; i++) {
            for (int j = 0, k = i; j < v.length; j++, k += mX) {
                output[i] += v[j] * m[k];
            }
        }
        return output;
    }

    public static double[] multiply(double s, double[] v_m) {
        double[] output = new double[v_m.length];
        for (int i = 0; i < v_m.length; i++) {
            output[i] = v_m[i] * s;
        }
        return output;
    }

    public static double[] add(double[] v_m0, double[] v_m1) {
        double[] output = new double[v_m0.length];
        for (int i = 0; i < v_m0.length; i++) {
            output[i] = v_m0[i] + v_m1[i];
        }
        return output;
    }

    public static double[] transpose(double[] m, int mY, int mX) {
        double[] output = new double[m.length];
        for (int i = 0; i < mX; i++) {
            for (int j = 0; j < mY; j++) {
                output[(i * mY) + j] = m[(j * mX) + i];
            }
        }
        return output;
    }

    public static double[] getMatrixElementAt(double[] t, int tmY, int tmX, int x) {
        double[] element = new double[tmY * tmX];
        System.arraycopy(t, (x * element.length), element, 0, element.length);
        return element;
    }

    public static double[] getMatrixElementAt(double[] t, int tY, int tmY, int tmX, int y, int x) {
        double[] element = new double[tmY * tmX];
        System.arraycopy(t, element.length * (y * tY + x), element, 0, element.length);
        return element;
    }

    public static double[] hadamard(double[] m0, double[] m1) {
        double[] output = new double[m0.length];
        for (int i = 0; i < m0.length; i++) {
            output[i] = m0[i] * m1[i];
        }
        return output;
    }

    public static double[] crossCorrelationValid(double[] image, int imageX, double[] kernel, int kernelX, int kernelLength, int outputX, int outputLength) {
        double[] result = new double[outputLength];
        for (int i = 0, y = 0, x = 0; i < outputLength; i++) {
            for (int j = 0, tY = y, tX = x; j < kernelLength; j++) {
                result[i] += kernel[j] * image[(tY * imageX) + tX];
                tX++;
                if ((j + 1) % kernelX == 0) {
                    tX = x;
                    tY++;
                }
            }
            x++;
            if ((i + 1) % outputX == 0) {
                y++;
                x = 0;
            }
        }
        return result;
    }

    public static double[] convolutionFull(double[] image, int imageY, int imageX, double[] kernel, int kernelY, int kernelX, int kernelLength, int outputX, int outputLength) {
        return crossCorrelationFull(image, imageY, imageX, rotate180(kernel, kernelY, kernelX), kernelY, kernelX, kernelLength, outputX, outputLength);
    }

    private static double[] crossCorrelationFull(double[] image, int imageY, int imageX, double[] kernel, int kernelY, int kernelX, int kernelLength, int outputX, int outputLength) {
        int yPadding = kernelY - 1;
        int xPadding = kernelX - 1;
        image = pad(image, imageY, imageX, yPadding, xPadding);
        return crossCorrelationValid(image, imageX + xPadding + xPadding, kernel, kernelX, kernelLength,  outputX, outputLength);
    }

    private static double[] pad(double[] image, int imageY, int imageX, int yPadding, int xPadding) {
        double[] result = new double[(imageY + yPadding + yPadding) * (imageX + xPadding + xPadding)];
        int resultX = imageX + xPadding + xPadding;
        int srcIndex = 0;
        int distIndex = (resultX * yPadding) + xPadding;
        for (int i = 0; i < imageY; i++) {
            System.arraycopy(image, srcIndex, result, distIndex, imageX);
            srcIndex += imageX;
            distIndex += resultX;
        }
        return result;
    }

    private static double[] rotate180(double[] m, int mY, int mX) {
        double[] output = new double[m.length];
        for (int i = 0; i < mY; i++) {
            for (int j = 0; j < mX; j++) {
                output[(i * mX) + j] = m[((mY - i - 1) * mX) + mX - j - 1];
            }
        }
        return output;
    }
}