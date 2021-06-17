

CMSC 426 – Computer Vision

Project 3: Classification with Neural Networks

Part 1 of this project involved implementing my own two-layer perceptron network and

training the network to use back-propagation to classify. Part 2 involved using Pytorch to

create and train a simple convolutional neural network to perform the same task

Alessandro Russo





**2.1 – Define the Network:**

To start I had to complete the forward function to return y\_hat. Y\_hat, as stated in the

project specifications, is equal to softmax(W2(σ(W1x + b1) + b2). Filling in the values in the

forward method was simple as I followed the variable assignment that was stated in lecture.

**2.2 – Initialize the Network:**

I filled out the initialization method and assigned W1 and W2 to random integers with some

standard deviation of 0.1. Following the diagram in the project specification, I set W1 shape

to 784 x 64 and W2’s shape was set to 64 x 10. B1 and b2 were initialized to zero with b1’s

shape being 1 x 64 and b2’s shape being 1 x 10. A side note is that I had to normalize all the

labels and image vectors to ensure no runtime errors. I also stabilized the softmax function.

**2.3 – Backpropagation**

This was the hardest part of the project that caused a lot of confusion. I was expected to

derive the following Jacobian matrices:

휕푎2 휕푎2 휕푎2 휕푓1 휕푎1 휕푎1

,

,

,

,

,

.

휕푏2 휕푊2 휕푓1 휕푎1 휕푏1 휕푊1

I used the chain rule to get the partial derivatives with respect to the vector variables W1

and W2 and b1 and b2.





**2.3 – Backpropagation**





**2.4 – Training**

I used, as required, a batch size of 256. The loop adjusted the start point and endpoint over

the whole epoch. Considerations were given to calculate the number of iterations to finish a

single epoch. The endpoint adjusted correctly for the final batch size. The weights and the

biases are adjusted at the end of each batch. Training was performed twice. Once on a

training set of 2000 and the other a training set of 50000 images. The accuracies can be seen

below.

Notice the accuracies increasing first drastically in the first 20 iterations, then converges

towards a value. The 2000 training set converges to around 0.92 and the 50000 training set

converges to 1.0.

**2.5 – Overfitting**

We can see from the training accuracy results that an accuracy of 0.92 was achieved for the

2000 images. However, the validation never reached more than 0.82 for the 2000 images

training set. This shows that the 2000 images caused an overfit for the neural network. With

the 50000 training set we do not as much overfitting comparing the validation results with

the training results, especially at the higher iterations. Comparing the orange line and blue

line for the 50000 training set, there is not a big gap between them.





**2.6 – How well did it work?**

As you can see the networks learned very quickly in the first 20 iterations. This can be seen

with the steep slope in the first part of the curve. After iteration 20 the accuracy for the

training reached in the 2000 training set images the neural network started to converge to a

value around 0.92 for the training set while the accuracy for 50000 training set images

reaches 1.00. Overall, both accuracies for training and validation sets are higher than 0.8

which is a good score

**2.7 – Where did It make mistakes?**

I created a dictionary to track where the neural network made mistakes. The dictionary is

keyed by each digit and the value is the number of misses for classifying this digit. We can see

that the highest misses are for the 8 followed by the 3. This could be because they look

similar. Another is the 5 and the 2 which have a similar amount of mistakes. Both 5 and 2 also

look similar if one is inverted.

{0: 5, 1: 14, 2: 31, 3: 39, 4: 17, 5: 34, 6: 15, 7: 14, 8: 63, 9: 19}

**2.8 – Visualize the Weights**

2000 training network running for the test data set gave an accuracy of 0.8078.

50000 training network running for the test data set gave an accuracy of 0.9116

Weights from the network were saved, loaded and the test images/labels were ran and pass

ed through my accuracy function to get these scores.

**3.1 – Describe the Network Architecture**

Fig 1

Shapes are the same in layers above





**3.2 – Train CNN (UNFINISHED)**

**3.3 – Overfitting?**

The 2000 training set reaches 0.97 accuracy score while the validation for the 2000 reaches 0.965.

The 5000 training set reaches 1.0 accuracy score while the validation for the 50000 reaches 1.0

also. We can conclude not much overfitting happened with the training data for the cnn.

**3.4 – How well does it work?**

Higher accuracies were reached using cnn compared with the np neural network and

there was less of an overfit. In addition, the runtime was much faster, and I checked this

by measuring the times with the time module even though it was very obvious. Lastly, I

could have improved the accuracy in the np networks by regularization or some form of

augmentation.





References:

<https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>

Convolution Diagram (fig 1) taken from

https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/

