import numpy as np
import time
import matplotlib.pyplot as plt

# train_images_np refers to x
train_images_np = np.load('./Project3_Data/MNIST_train_images.npy')
# train_images_np= train_images_np[0:1000, :]
train_size = train_images_np.shape[0]

# train_labels_np refers to y
train_labels = np.load('./Project3_Data/MNIST_train_labels.npy')
# print("train labels", train_labels.shape)
# train_labels= train_labels[0:1000]

# print("train labels", train_labels.shape)

val_images_np = np.load('./Project3_Data/MNIST_val_images.npy')
# print("VALIDATION IMAGES:", val_images_np.shape)
# val_images_np= val_images_np[0:1000, :]
val_size = val_images_np.shape[0]

val_labels = np.load('./Project3_Data/MNIST_val_labels.npy')
# print("VALIDATION LABELS:", val_labels_np.shape)

val_labels_np = np.zeros((val_size, 10))

for idx, i in zip(range(len(val_labels)), val_labels):
    # np.put(train_labels_np, [i * (idx + 1)], 1)
    raveled_index = np.ravel_multi_index(([idx], [i]), val_labels_np.shape)
    np.put(val_labels_np, raveled_index, 1)

# val_labels_np= val_labels_np[0:1000]

test_images_np = np.load('./Project3_Data/MNIST_test_images.npy')
# print("TEST IMAGES:", test_images_np.shape)
# test_images_np= test_images_np[0:1000, :]

test_labels = np.load('./Project3_Data/MNIST_test_labels.npy')
test_size = test_labels.shape[0]
# test_labels= test_labels[0:1000]

# print("train images", train_images.shape)
# replace 50000 with train_labels_np.shape[0]
train_labels_np = np.zeros((train_size, 10))

for idx, i in zip(range(len(train_labels)), train_labels):
    # np.put(train_labels_np, [i * (idx + 1)], 1)
    raveled_index = np.ravel_multi_index(([idx], [i]), train_labels_np.shape)
    np.put(train_labels_np, raveled_index, 1)

# print(np.nonzero(train_labels_np))
# print("#########")
# print(train_labels)
# print("#########")
# print(train_labels_np[0])
# print("#########")
# print(train_labels_np[49999])

test_labels_np = np.zeros((test_size, 10))

for idx, i in zip(range(len(test_labels)), test_labels):
    # np.put(test_labels_np, ([idx], [i]), 1)
    raveled_index = np.ravel_multi_index(([idx], [i]), test_labels_np.shape)
    np.put(test_labels_np, raveled_index, 1)

# train_images_np.reshape((28, 28))
train_images_np.astype("float64")
# norm = np.linalg.norm(train_images_np)
train_images_np = train_images_np / 255

# test_images_np.reshape((28, 28))
test_images_np.astype("float64")
# norm = np.linalg.norm(test_images_np)
test_images_np = test_images_np / 255

val_images_np.astype("float64")
val_images_np = val_images_np / 255

#print(train_images_np.dtype)


##Template MLP code
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def softmax_derivative(x):
    p = softmax(x)
    batchsize = x.shape[0]

    # print("wX + B shape: ", x.shape)
    # print("softmax shape: ", p.shape)

    deriv = np.diag(p)
    output_array = np.zeros((batchsize, 10, 10))

    # print("p.shape[0]:", p.shape[0])
    # print("DERIV LEN:", len(deriv))

    for k in range(p.shape[0]):
        for i in range(len(deriv)):
            for j in range(len(deriv)):
                if i == j:
                    raveled_index = np.ravel_multi_index(([k], [i], [j]), output_array.shape)
                    np.put(output_array, raveled_index, p[i] * (1 - p[i]))
                else:
                    raveled_index = np.ravel_multi_index(([k], [i], [j]), output_array.shape)
                    np.put(output_array, raveled_index, -p[i] * p[j])

    # print("J_DERIV: ", output_array.shape)
    return output_array.sum(axis=1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    p = sigmoid(x)
    deriv = p * (np.ones(x.shape) - p)
    return deriv


def CrossEntropy_derivative(y_hat, y):
    p = CrossEntropy(y_hat, y)
    return p - y


class MLP():

    def __init__(self):
        # Initialize all the parametres
        # Uncomment and complete the following lines
        self.W1 = np.random.normal(size=(784, 64), scale=0.1)
        self.b1 = np.zeros(shape=(1, 64))
        self.W2 = np.random.normal(size=(64, 10), scale=0.1)
        self.b2 = np.zeros(shape=(1, 10))
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0

    def forward(self, x):
        # Feed data through the network
        # Uncomment and complete the following lines
        # print("FORWARD FUNCTION ############")

        self.x = x

        batchsize = x.shape[0]

        self.W1x = np.dot(x, self.W1)
        # print("SELF.W1X shape: ", self.W1x.shape)

        b1_ones = np.ones((batchsize, 1))
        b1_reshaped = np.dot(b1_ones, self.b1)
        self.a1 = self.W1x + b1_reshaped

        # print("B1 RESHAPED shape: ", b1_reshaped.shape)
        # print("SELF.A1 shape: ", self.a1.shape)

        self.f1 = sigmoid(self.a1)

        # print("F1 shape: ", self.f1.shape)

        self.W2x = np.dot(self.f1, self.W2)  # x is f1 in regards to this part

        # print("SELF.w2x shape: ", self.W2x.shape)

        self.a2 = self.W2x + self.b2

        # print("SELF.a2 shape: ", self.a2.shape)

        self.y_hat = softmax(self.a2)

        # print("SELF.y_hat shape: ", self.y_hat.shape)

        return self.y_hat

    def predict(self, x):
        b1_ones = np.ones((x.shape[0], 1))
        b1_reshaped = np.dot(b1_ones, self.b1)

        W1x = np.dot(x, self.W1)

        a1 = W1x + b1_reshaped

        # print("B1 RESHAPED shape: ", b1_reshaped.shape)
        # print("SELF.A1 shape: ", self.a1.shape)

        f1 = sigmoid(a1)

        # print("F1 shape: ", self.f1.shape)

        W2x = np.dot(f1, self.W2)  # x is f1 in regards to this part

        # print("SELF.w2x shape: ", self.W2x.shape)

        a2 = W2x + self.b2

        # print("SELF.a2 shape: ", self.a2.shape)

        y_hat = softmax(a2)

        # print("SELF.y_hat shape: ", self.y_hat.shape)

        return y_hat

    def update_grad(self, y):
        # print("UPDATE GRAD FUNCTION ############")

        # print("W2x:", self.W2x.shape)

        batchsize = y.shape[0]

        b2_ones = np.ones((batchsize, 1))
        b2_reshaped = np.dot(b2_ones, self.b2)

        b1_ones = np.ones((batchsize, 1))
        b1_reshaped = np.dot(b1_ones, self.b1)

        # print("B2:", b2_reshaped.shape)

        # RECOMPUTE self.W2x
        #
        softmax_derivative_a2_value = softmax_derivative(self.W2x + b2_reshaped)

        # dA2db2 = softmax_derivative_a2_value
        dA2db2 = np.ones(b2_reshaped.shape)

        # print("DA2DB2: ", dA2db2.shape)

        # dA2dW2 = np.dot(softmax_derivative_a2_value.T, self.f1)
        dA2dW2 = self.f1

        # print("DA2DW2 shape: ", dA2dW2.shape)

        # dA2dF1 = np.dot(softmax_derivative_a2_value, self.W2.T)
        dA2dF1 = self.W2

        sigmoid_derivative_a1_value = sigmoid_derivative(self.W1x + self.b1)
        dF1dA1 = sigmoid_derivative(self.a1)
        # dA1db1 = sigmoid_derivative_a1_value
        dA1db1 = np.ones(b1_reshaped.shape)

        # dA1dW1 = np.dot(sigmoid_derivative_a1_value.T, self.x)
        # DA1DW1 not being used
        dA1dW1 = self.x

        dLdA2 = self.y_hat - y

        # print("DLDA2: ", dLdA2.shape)
        # print("SELF.YHAT: ", self.y_hat)
        # print("Y: ", y)
        # print("SELF.YHAT SHAPE: ", self.y_hat.shape)
        # print("Y SHAPE: ", y.shape)

        # dLdW2 = np.dot(dLdA2, dA2dW2)
        dLdW2 = np.dot(dA2dW2.T, dLdA2)

        # print("DLDW2 SHAPE: ", dLdW2.shape)

        # dLdb2 = np.dot(dLdA2, dA2db2)
        dLdb2 = dLdA2 * dA2db2

        # print("DLDB2 SHAPE: ", dLdb2.shape)
        # print("DLDA2 SHAPE: ", dLdA2.shape)
        # print("DF1DA1 SHAPE: ", dF1dA1.shape)
        # print("A1 shape:", self.a1.shape)

        # dLdF1 = np.dot(np.dot(dLdA2, sigmoid_derivative(self.a1)), dF1dA1)
        dLdF1 = np.dot(dLdA2, dA2dF1.T)

        sigmoid_derivative_a1_p = sigmoid_derivative(self.a1)

        # print("SIGMOID DERIV A1:", sigmoid_derivative_a1_p.shape)
        # print("W2 shape:", self.W2.shape)
        # print("X shape:", self.x.shape)

        dA2dW1 = np.dot(np.dot(sigmoid_derivative_a1_p, self.W2).T, self.x)
        # print("DA")
        # dA2dW1 = dA2dF1 * dF1dA1 * dA1dW1
        ## CHANGING
        # dA2dW1 = dA2dW2

        # print("dF1Da1", dF1dA1.shape)
        # print("dlDa2", dLdA2.shape)
        # print("da2Df1", dA2dF1.shape)

        dLdA1 = np.dot(dLdA2, dA2dF1.T) * dF1dA1

        # print("DLDA2 shape:", dLdA2.shape)
        # print("DA2DW1 shape:", dA2dW1.shape)
        # print("SOFTMAX DERIV A2 values:", np.nonzero(softmax_derivative(self.a2)))
        # print("SIGMOID DERIV A1 values:", np.nonzero(sigmoid_derivative_a1_p))
        # print("W2 values:", np.nonzero(self.W2))
        # print("X values:", np.nonzero(self.x))

        # print("DLDA2 shape:", dLdA2.shape)
        # print("W1 shape:", self.W1.shape)
        # print("W2 shape:", self.W2.shape)
        # print("x shape:", self.x.shape)
        # print("sigmoid deriv w1x + b shape:", sigmoid_derivative_a1_p.shape)

        # dLdW1 = np.dot((np.dot(softmax_derivative(self.a2) * dLdA2, self.W2.T) * sigmoid_derivative_a1_p).T, self.x).T
        dLdW1 = np.dot(self.x.T, np.dot(dLdA2, self.W2.T) * sigmoid_derivative_a1_p)

        # print("DLDW1 values:", np.nonzero(dLdW1))
        # print("DA2dF1 shape:", dA2dF1.shape)

        # multiply by 1 - use the shape of I to reshape B if needed
        dF1db1 = sigmoid_derivative_a1_p * 1

        # print("DF1DB1 shape:", dF1db1.shape)

        dLdb1 = np.dot(dLdA2, dA2dF1.T) * dF1db1

        self.W2_grad = self.W2_grad + dLdW2

        self.b2_grad = self.b2_grad + np.dot(np.ones((1, batchsize)), dLdb2)

        self.W1_grad = self.W1_grad + dLdW1
        self.b1_grad = self.b1_grad + np.dot(np.ones((1, batchsize)), dLdb1)

        # print("b1 grad values:", self.b1_grad)
        # print("b2 grad values:", self.b2_grad)
        # print("FINAL W1 Values:", self.W1_grad[np.nonzero(self.W1_grad)])
        # print("FINAL W2 Values:", self.W2_grad)

    def update_params(self, learning_rate):
        self.W2 = self.W2 - learning_rate * self.W2_grad
        self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(-1)
        self.W1 = self.W1 - learning_rate * self.W1_grad
        self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)


def accuracy(y, y_hat):
    maxp = np.max(y_hat, axis=1)

    for i in range(len(maxp)):
        y_hat[i][y_hat[i] != maxp[i]] = 0
        y_hat[i][y_hat[i] == maxp[i]] = 1

    # print(y_hat)
    # print("y sum", np.sum(y))
    # print("y_hat sum", np.sum(y_hat))
    # misses = np.sum(np.abs(np.bitwise_xor(y , y_hat)))
    misses = 0
    for i in range(y.shape[0]):
        if not all(np.equal(y[i], y_hat[i])):
            misses = misses + 1

    # print("misses", misses)
    output_size = y.shape[0]
    # print("output size", output_size)
    acc = (output_size - misses) / output_size
    return acc


## Init the MLP
from sklearn.metrics import accuracy_score

myNet_2000 = MLP()

epoch_size = 2000
t1 = time.perf_counter_ns()

# learning_rate=1e-3
learning_rate = 0.001
n_epochs = 100
counter = 0

# print("TRAIN IMAGES NP SHAPE: ", train_images_np.shape)

# accuracy_list = np.zeros(2000)
accuracy_train_list_2000 = []
accuracy_val_list_2000 = []

## Training code
for iter in range(n_epochs):
    batchsize = 256

    num_of_batches = epoch_size // batchsize
    if epoch_size % batchsize != 0:
        num_of_batches += 1

    start_point = 0
    end_point = start_point + batchsize

    y_hat_pred = None

    for i in range(num_of_batches):

        # Code to train network goes here
        if i == num_of_batches - 1:
            end_point = epoch_size

        myNet_2000.reset_grad()
        y_hat = myNet_2000.forward(train_images_np[start_point:end_point, :])

        res = accuracy(train_labels_np[start_point:end_point], y_hat)
        # print(res)

        if y_hat_pred is None:
            # print("yhat is none")
            y_hat_pred = y_hat
        else:
            y_hat_pred = np.vstack((y_hat_pred, y_hat))

        myNet_2000.update_grad(train_labels_np[start_point:end_point])
        myNet_2000.update_params(learning_rate)
        start_point += batchsize
        end_point = start_point + batchsize

    # Code to compute validation loss/accuracy goes here
    # yhat_pred = myNet_2000.predict(train_images_np[0:2000, :])
    # print("yhat pred shape", y_hat_pred.shape)

    val_yhat_pred = myNet_2000.predict(val_images_np)
    val_acc = accuracy(val_labels_np, val_yhat_pred)
    acc = accuracy_score(train_labels_np[0:epoch_size], y_hat_pred)

    # acc = accuracy(train_labels_np[0:epoch_size], y_hat_pred)
    accuracy_train_list_2000.append(acc)
    accuracy_val_list_2000.append(val_acc)
    # print(counter)
    # counter = counter + 1

print(accuracy_train_list_2000)
print(accuracy_val_list_2000)
t2 = time.perf_counter_ns()
print("Time taken for np 2000 training network", t2 - t1)

d = {i: 0 for i in range(10)}

for i in range(epoch_size):
    if not all(np.equal(train_labels_np[i], y_hat_pred[i])):
        d[train_labels[i]] += 1

print("Mistakes:", d)

num_iterations = [i + 1 for i in range(len(accuracy_train_list_2000))]

plt.plot(num_iterations, accuracy_train_list_2000, label = "train")
plt.plot(num_iterations, accuracy_val_list_2000, label = "validation")
plt.legend()
plt.xlabel("Num Iterations")
plt.ylabel("Accuracy")
plt.title("2000 Images")

np.save("w1_2000.npy", myNet_2000.W1)
np.save("w2_2000.npy", myNet_2000.W2)
np.save("b1_2000.npy", myNet_2000.b1)
np.save("b2_2000.npy", myNet_2000.b2)

myNet_2000.W1 = np.load("w1_2000.npy")
myNet_2000.W2 = np.load("w2_2000.npy")
myNet_2000.b1 = np.load("b1_2000.npy")
myNet_2000.b2 = np.load("b2_2000.npy")

pred_test = myNet_2000.predict(test_images_np)
acc_test = accuracy(test_labels_np, pred_test)
print("2000 trainset against test_images accuracy:", acc_test)

myNet_50000 = MLP()

epoch_size = 50000

# learning_rate=1e-3
learning_rate = 0.001
n_epochs = 100

t3 = time.perf_counter_ns()

# print("TRAIN IMAGES NP SHAPE: ", train_images_np.shape)

# accuracy_list = np.zeros(2000)
accuracy_train_list_50000 = []
accuracy_val_list_50000 = []
counter = 0

## Training code
for iter in range(n_epochs):
    batchsize = 256

    num_of_batches = epoch_size // batchsize
    if epoch_size % batchsize != 0:
        num_of_batches += 1

    start_point = 0
    end_point = start_point + batchsize

    y_hat_pred = None

    for i in range(num_of_batches):

        # Code to train network goes here
        if i == num_of_batches - 1:
            end_point = epoch_size

        myNet_50000.reset_grad()
        y_hat = myNet_50000.forward(train_images_np[start_point:end_point, :])

        res = accuracy(train_labels_np[start_point:end_point], y_hat)
        # print(res)

        if y_hat_pred is None:
            # print("yhat is none")
            y_hat_pred = y_hat
        else:
            y_hat_pred = np.vstack((y_hat_pred, y_hat))

        myNet_50000.update_grad(train_labels_np[start_point:end_point])
        myNet_50000.update_params(learning_rate)
        start_point += batchsize
        end_point = start_point + batchsize

    # Code to compute validation loss/accuracy goes here
    # yhat_pred = myNet_2000.predict(train_images_np[0:2000, :])
    # print("yhat pred shape", y_hat_pred.shape)

    val_yhat_pred = myNet_50000.predict(val_images_np)
    val_acc = accuracy(val_labels_np, val_yhat_pred)
    acc = accuracy_score(train_labels_np[0:epoch_size], y_hat_pred)

    # acc = accuracy(train_labels_np[0:epoch_size], y_hat_pred)
    accuracy_train_list_50000.append(acc)
    accuracy_val_list_50000.append(val_acc)

    #print(counter)
    counter = counter + 1

print(accuracy_train_list_50000)
print(accuracy_val_list_50000)
t4 = time.perf_counter_ns()
print("Time taken for np 50000 training network", t4 - t3)

np.save("w1_50000.npy", myNet_50000.W1)
np.save("w2_50000.npy", myNet_50000.W2)
np.save("b1_50000.npy", myNet_50000.b1)
np.save("b2_50000.npy", myNet_50000.b2)

myNet_50000.W1 = np.load("w1_50000.npy")
myNet_50000.W2 = np.load("w2_50000.npy")
myNet_50000.b1 = np.load("b1_50000.npy")
myNet_50000.b2 = np.load("b2_50000.npy")

pred_test = myNet_50000.predict(test_images_np)
acc_test = accuracy(test_labels_np, pred_test)
print("50000 trainset against test_images accuracy:", acc_test)

num_iterations = [i + 1 for i in range(len(accuracy_train_list_50000))]

plt.plot(num_iterations, accuracy_train_list_50000, label = "train")
plt.plot(num_iterations, accuracy_val_list_50000, label = "validation")
plt.legend()
plt.xlabel("Num Iterations")
plt.ylabel("Accuracy")
plt.title("50000 Images")

## Template for ConvNet Code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNet(nn.Module):
    # From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # one of the layers of the neural network
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.view(-1, 1, 28, 28))))
        x = self.pool(F.relu(self.conv2(x)))
        # conv1d -> maxpooling -> conv2d -> another max pooling
        # reshape data again
        x = x.view(-1, 16 * 4 * 4)
        # 0 for all neg values and linear for all pos values __|/
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Your training and testing code goes here
from torch.optim import SGD

myNet_torch = ConvNet()

epoch_size = 2000

# learning_rate=1e-3
learning_rate = 0.01
n_epochs = 100

t5 = time.perf_counter_ns()

# print("TRAIN IMAGES NP SHAPE: ", train_images_np.shape)

# accuracy_list = np.zeros(2000)
accuracy_train_list_conv_2000 = []
accuracy_val_list_conv_2000 = []

criterion = nn.CrossEntropyLoss()
optimizer = SGD(myNet_torch.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(n_epochs):  # loop over the dataset multiple times

    batchsize = 256

    num_of_batches = epoch_size // batchsize
    if epoch_size % batchsize != 0:
        num_of_batches += 1

    start_point = 0
    end_point = start_point + batchsize
    # y_hat_pred = None

    correct = 0
    total = 0

    for i in range(num_of_batches):

        running_loss = 0.0

        # Code to train network goes here
        if i == num_of_batches - 1:
            end_point = epoch_size

        # get the inputs; data is a list of [inputs, labels]
        inputs = train_images_np[start_point:end_point, :]
        inputs = torch.Tensor(inputs)

        labels = train_labels[start_point:end_point]
        labels = torch.Tensor(labels)
        labels = labels.type(torch.LongTensor)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = myNet_torch(inputs)

        # print(type(outputs))
        # print(type(labels))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        start_point += batchsize
        end_point = start_point + batchsize

    # print('Finished Training')
    # Compute accuracy

    inputs = train_images_np[0:epoch_size, :]
    inputs = torch.Tensor(inputs)

    labels = train_labels[0:epoch_size]
    labels = torch.Tensor(labels)
    labels = labels.type(torch.LongTensor)

    outputs = myNet_torch(inputs)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    acc = correct / total
    print(acc)

    accuracy_train_list_conv_2000.append(acc)

    inputs = val_images_np[0:epoch_size, :]
    inputs = torch.Tensor(inputs)

    labels = val_labels[0:epoch_size]
    labels = torch.Tensor(labels)
    labels = labels.type(torch.LongTensor)

    outputs = myNet_torch(inputs)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    acc = correct / total

    accuracy_val_list_conv_2000.append(acc)

print(accuracy_train_list_conv_2000)
print(accuracy_val_list_conv_2000)
t6 = time.perf_counter_ns()
print("Time taken for cnn 2000 training network", t6 - t5)

num_iterations = [i + 1 for i in range(len(accuracy_train_list_conv_2000))]

plt.plot(num_iterations, accuracy_train_list_conv_2000, label = "train")
plt.plot(num_iterations, accuracy_val_list_conv_2000, label = "validation")
plt.legend()
plt.xlabel("Num Iterations")
plt.ylabel("Accuracy")
plt.title("2000 Images: ConvNet")

torch.save(myNet_torch.state_dict(), 'model_weights2000_cnn.pth')

myNet_torch.load_state_dict(torch.load('model_weights2000_cnn.pth'))
myNet_torch.eval()


myNet_torch50000 = ConvNet()

epoch_size = 50000

# learning_rate=1e-3
learning_rate = 0.01
n_epochs = 100
t7 = time.perf_counter_ns()

# print("TRAIN IMAGES NP SHAPE: ", train_images_np.shape)

# accuracy_list = np.zeros(2000)
accuracy_train_list_conv_50000 = []
accuracy_val_list_conv_50000 = []

criterion = nn.CrossEntropyLoss()
optimizer = SGD(myNet_torch50000.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(n_epochs):  # loop over the dataset multiple times

    batchsize = 256

    num_of_batches = epoch_size // batchsize
    if epoch_size % batchsize != 0:
        num_of_batches += 1

    start_point = 0
    end_point = start_point + batchsize
    # y_hat_pred = None

    correct = 0
    total = 0

    for i in range(num_of_batches):

        running_loss = 0.0

        # Code to train network goes here
        if i == num_of_batches - 1:
            end_point = epoch_size

        # get the inputs; data is a list of [inputs, labels]
        inputs = train_images_np[start_point:end_point, :]
        inputs = torch.Tensor(inputs)

        labels = train_labels[start_point:end_point]
        labels = torch.Tensor(labels)
        labels = labels.type(torch.LongTensor)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = myNet_torch50000(inputs)

        # print(type(outputs))
        # print(type(labels))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        start_point += batchsize
        end_point = start_point + batchsize

    # print('Finished Training')
    # Compute accuracy

    inputs = train_images_np[0:epoch_size, :]
    inputs = torch.Tensor(inputs)

    labels = train_labels[0:epoch_size]
    labels = torch.Tensor(labels)
    labels = labels.type(torch.LongTensor)

    outputs = myNet_torch50000(inputs)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    acc = correct / total
    print(acc)

    accuracy_train_list_conv_50000.append(acc)

    inputs = val_images_np[0:epoch_size, :]
    inputs = torch.Tensor(inputs)

    labels = val_labels[0:epoch_size]
    labels = torch.Tensor(labels)
    labels = labels.type(torch.LongTensor)

    outputs = myNet_torch50000(inputs)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    acc = correct / total

    accuracy_val_list_conv_50000.append(acc)

print(accuracy_train_list_conv_50000)
print(accuracy_val_list_conv_50000)
t8 = time.perf_counter_ns()
print("Time taken for cnn 50000 training network", t8 - t7)

num_iterations = [i + 1 for i in range(len(accuracy_train_list_conv_50000))]

plt.plot(num_iterations, accuracy_train_list_conv_50000, label = "train")
plt.plot(num_iterations, accuracy_val_list_conv_50000, label = "validation")
plt.legend()
plt.xlabel("Num Iterations")
plt.ylabel("Accuracy")
plt.title("50000 Images: ConvNet")

torch.save(myNet_torch50000.state_dict(), 'model_weights50000_cnn.pth')
myNet_torch50000.load_state_dict(torch.load('model_weights50000_cnn.pth'))
myNet_torch50000.eval()