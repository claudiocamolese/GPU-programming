# MNIST CNN with C and CUDA


This repo was directly copied from the [Programming Massively Parallel Processors](https://github.com/katsudon16/programming_massively_parallel_processors) repo, specifically from chapter 16: Deep Learning.

The repo aims to implement basic neural network layers using C and CUDA kernels from scratch.


## MNIST Data

The MNIST data can be found from multiple sources, including [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). Download the dataset and store them in the `data` directory:

- Training files
```
- data/train-images-idx3-ubyte
- data/train-labels-idx1-ubyte
```

- Test files
```
- data/t10k-images-idx3-ubyte
- data/t10k-labels-idx1-ubyte
```

## Implementation

The main code can be found in `train_mnist.cu` file. 


### Image preprocessing

The initial 28x28 `uint8_t` images are:

1) converted into `float`
2) normalized (i.e., dividing by 255)
3) padded with 0s into 32x32 images


### Model architecture

The model architecture currently consists of the following layers:

- 2D convolutional layers: 16 filters, each `5 x 5`; produces output dimension of `[batch_size, 16, 28, 28]`
- Sigmoid layer; used *simply because* it's mentioned in the book
- Max pooling layer with kernel size `2 x 2`; produces output dimension of `[batch_size, 16, 14, 14]`
- Flatten layer; produces output dim of `[batch_size, 3136]`
- Linear layer with dimension `3136 x 10`; produces output dim of `[batch_size, 10]`
- Softmax layer

It then takes the logarithm of the softmax output and performs **negative log-likelihood** to compute the loss. Note that log softmax + NLL Loss is essentially **cross-entropy loss**.

Tons of improvements can certainly be made to the model architecture/training, including but not limited to:

- using a better dataset split method: use randomization or split by distribution assuming that the training set distribution reflects the test set distribution
- replacing sigmoid activation with ReLU
- adding batch normalization
- implementing early stopping
- using learning rate decay
- **using a more reliable evaluation metric** instead of accuracy.

These improvements are currently not a priority as this project's main goal is for C and CUDA programming practice.

### Tests

More info on the tests can be found in the test README file.


## Expected output

The following results were generated using NVIDIA GeForce GTX 1080.

```
[INFO] # Samples in training set: 60000
The train dataset is splitted into training (n=36000) and validation (n=24000).
Epoch 0:
Time taken per layer in the forward pass (ms):
>>> Layer 0 | total:     66.314 ms | average:      0.474 ms 
>>> Layer 1 | total:     24.063 ms | average:      0.172 ms 
>>> Layer 2 | total:     22.530 ms | average:      0.161 ms 
>>> Layer 3 | total:      0.000 ms | average:      0.000 ms 
>>> Layer 4 | total:     24.590 ms | average:      0.176 ms 
>>> Layer 5 | total:      3.372 ms | average:      0.024 ms 

Time taken per layer in the backward pass (ms):
>>> Layer 0 | total:   1947.216 ms | average:     13.909 ms 
>>> Layer 1 | total:     24.899 ms | average:      0.178 ms 
>>> Layer 2 | total:     20.287 ms | average:      0.145 ms 
>>> Layer 3 | total:      0.000 ms | average:      0.000 ms 
>>> Layer 4 | total:     45.685 ms | average:      0.326 ms 

Train loss: 610.436

Epoch 1:
Train loss: 149.599
Valid loss: 70.602 | accuracy: 76.425%

Epoch 2:
Train loss: 86.212

Epoch 3:
Train loss: 71.634
Valid loss: 42.728 | accuracy: 87.442%

Epoch 4:
Train loss: 63.749

Epoch 5:
Train loss: 60.004
Valid loss: 37.932 | accuracy: 88.367%

Epoch 6:
Train loss: 56.884

Epoch 7:
Train loss: 54.398
Valid loss: 35.864 | accuracy: 88.371%

Epoch 8:
Train loss: 52.558

Epoch 9:
Train loss: 51.004
Valid loss: 32.961 | accuracy: 89.721%

[INFO] # Samples in test set: 10000
Test accuracy: 90.520%
```

## Future improvements

The implementation of the CUDA kernels are still naive and may be slow. Further optimizations (or even moving the computation to CPU if using GPU is not necessary) will be done, e.g. optimizing `dW` computation, which currently takes >12 ms on average.

As mentioned above, optimizing the model architecture and training is currently not the priority.