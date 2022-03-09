# JAVA NEURAL NETWORK LIBRARY

XOR example:

```java
// dense neural netowork with learning rate 0.3
NeuralNetwork nn = new NeuralNetwork(NetworkType.DENSE, 0.3);

// first hidden layer with 3 nodes
nn.addDenseLayer(3);
// second hidden layer with 2 nodes
nn.addDenseLayer(2);
// third hidden layer with 3 nodes
nn.addDenseLayer(3);
// output layer with 1 node
nn.addDenseLayer(1);

// training data
ArrayList<Pair<double[], double[]>> trainingData = new ArrayList<>() {{
  add(new Pair<>(new double[]{0, 0}, new double[]{0}));
  add(new Pair<>(new double[]{0, 1}, new double[]{1}));
  add(new Pair<>(new double[]{1, 0}, new double[]{1}));
  add(new Pair<>(new double[]{1, 1}, new double[]{0}));
}};

// training loop
for (int i = 0; i < 10000; i++) {
  for (Pair<double[], double[]> trainingDatum : trainingData) {
    // datum through network
    double[] y = neuralNetwork.forward(trainingDatum.first());
    // network error using mean squared error
    double error = mse(y, trainingDatum.second());
    System.out.println("error: " + error);
    // error gradient 
    double[] dy = msePrime(y, trainingDatum.second());
    // backpropagation
    neuralNetwork.backward(dy);
    // gradient descent
    neuralNetwork.gradientDescent();
  }
  // reorder training data
  Collections.shuffle(trainingData);
}

// test by passing each data item through the trained network and see the output
trainingData.forEach(p -> System.out.println(Arrays.toString(p.first()) + " -> " + Arrays.toString(neuralNetwork.forward(p.second()))));
```

Convolutional Neural Network example (MNIST):

```java
// convolutional neural network with learning rate 0.4
NeuralNetwork nn = new NeuralNetwork(NetworkType.CONV, 0.4);

// first hidden convolutional layer with 5 5x5 filters
nn.addConvolutionalLayer(5, 5, 5);
// second hidden convolutional layer with 3 2x2 filters
nn.addConvolutionalLayer(3, 2, 2);
// first hidden dense layer with 15 nodes
nn.addDenseLayer(15);
// output layer with 10 nodes
nn.addDenseLayer(10);

// training looop
double error = 0;
for (int i = 0, sampleCount = 0; i < epochs; i++) {
  for (int j = 0; j < trainingData.size(); j++, sampleCount++) {
    Sample currentSample = trainingData.get(j);
    
    // datum through network
    double[] output = neuralNetwork.forward(currentSample.getData(), 1, 28, 28);

    // calculating the cummulative error from the network error using mean squared error
    error = (error * sampleCount + mse(output, currentSample.getExpectedValue())) / (sampleCount + 1);
    double[] errorGradient = msePrime(output, currentSample.getExpectedValue());

    // backpropagation
    neuralNetwork.backward(errorGradient);
    // gradient descent
    neuralNetwork.gradientDescent();

    if ((sampleCount) % 100 == 0 && sampleCount != 0) {
      System.out.println("epoch: " + (i + 1) + " | sample: " + j + " | error: " + error);
    }
  }
  // reorder training data
  Collections.shuffle(trainingData);
}
```
