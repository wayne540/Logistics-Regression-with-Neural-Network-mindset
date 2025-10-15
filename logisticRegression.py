import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from h5py import File
import random
import tensorflow


def load_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # get file exact file names
    training_set = os.path.join(BASE_DIR, "train_catvnoncat.h5")
    testing_set = os.path.join(BASE_DIR, "test_catvnoncat.h5")

    # load training and testing data
    train_data = File(training_set, "r")
    train_set_x_orig = np.array(train_data["train_set_x"][:])
    train_set_y_orig = np.array(train_data["train_set_y"][:])

    test_data = File(testing_set, "r")
    test_set_x_orig = np.array(test_data["test_set_x"][:])
    test_set_y_orig = np.array(test_data["test_set_y"][:])

    classes = np.array(train_data["list_classes"][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

train_set_y_orig = train_set_y_orig.reshape(1, -1)
test_set_y_orig = test_set_y_orig.reshape(1, -1)

# display a random image
index = random.randint(0, 209)
plt.imshow(train_set_x_orig[index])
label = train_set_y_orig[0, index]
print (f"Lebel for: {label} ({'cat' if label == 1 else 'non cat'})")
# plt.show()

m_train = train_set_y_orig.shape[1]
m_test = test_set_y_orig.shape[1]
num_px = train_set_x_orig.shape[1]

print (f"Number of training examples: {m_train}")
print (f"Number of testing examples: {m_test}")
print(f"Image size is {num_px} by {num_px} pixels")
print (f"shape of train_set_y_orig is: {train_set_y_orig.shape}")
print (f"shape of test_set_y_orig is: {test_set_y_orig.shape}")
print (f"shape of train_set_x_orig is: {train_set_x_orig.shape}")
print (f"shape of test_set_x_orig is: {test_set_x_orig.shape}")

# flatten and standardize the data
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x_flatten = train_set_x_flatten/255
test_set_x_flatten = test_set_x_flatten/255

print (f"shape of train_set_x_flatten is {train_set_x_flatten.shape}")
print (f"Shape of test_set_x_flatten is {test_set_x_flatten.shape} ")

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def innit_params(dim):
    b = 0
    return np.zeros((dim, 1)), b

def propagation(w, b, X, Y):
    m = X.shape[1]
    epsilon = 1e-8
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
    
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.dot(A - Y)

    return {"dw" : dw, "db" : db}, np.squeeze(cost)

def optimazation(w, b, X, Y, epochs, lr, print_cost = False):
    costs = []
    for i in range (epochs):
        cost, grads = propagation(w, b, X, Y)
        
        w -= lr * grads["dw"] 
        b -= lr * grads["db"]

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"cost after {epochs} epochs is {cost}")

    return {"w" : w, "b" : b}, grads, cost
