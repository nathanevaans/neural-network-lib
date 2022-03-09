package com.company.examples.MNIST.util;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.lang.String.format;

public class FileReader {

    private final static int LABEL_FILE_MAGIC_NUMBER = 2049;
    private final static int IMAGE_FILE_MAGIC_NUMBER = 2051;

    public static List<Sample> loadImageData(String filePrefix) {
        List<Sample> images = null;
        ClassLoader loader = FileReader.class.getClassLoader();
        String imageFileName = "com/company/examples/MNIST/resources/" + filePrefix + "-images-idx3-ubyte";
        String labelFileName = "com/company/examples/MNIST/resources/" + filePrefix + "-labels-idx1-ubyte";
        try (
                DataInputStream imageIS = new DataInputStream(loader.getResourceAsStream(imageFileName));
                DataInputStream labelIS = new DataInputStream(loader.getResourceAsStream(labelFileName));
        ) {
            if (imageIS.readInt() != IMAGE_FILE_MAGIC_NUMBER)
                throw new IOException("Unknown file format for " + imageFileName);

            if (labelIS.readInt() != LABEL_FILE_MAGIC_NUMBER)
                throw new IOException("Unknown file format for " + labelFileName);

            int imageCount = imageIS.readInt();
            int labelCount = labelIS.readInt();

            if (imageCount != labelCount)
                throw new IOException(format("File %s and %s contains data for different number of images", imageFileName, labelFileName));

            images = new ArrayList<>(imageCount);
            int rows = imageIS.readInt();
            int cols = imageIS.readInt();
            byte[] data = new byte[rows * cols];

            for (int i = 0; i < imageCount; i++) {
                double[] image = new double[rows * cols];
                imageIS.read(data, 0, data.length);
                for (int j = 0; j < image.length; j++) {
                    image[j] = (data[j] & 255) / 255.0;
                }
                images.add(new Sample(image, labelIS.readByte()));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return images;
    }
}
