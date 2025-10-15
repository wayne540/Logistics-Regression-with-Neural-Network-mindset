# Logistics-Regression-with-Neural-Network-mindset
Before diving “deep” into Neural Network, it is essential to get a good intuition. What better way than to understand the logic of how the Logistic Regression Algorithm can be modeled as a simple Neural Network that actually learns from data.
The classic application of Logistics regression is binary classification. Logistic regression is useful if we are working with a dataset where the classes are more or less “linearly separable.” Neural networks are somewhat related to logistic regression. Basically, we can think of logistic regression as a one-layer neural network.

![Sigmoid Function](https://raw.githubusercontent.com/wayne540/Logistics-Regression-with-Neural-Network-mindset/main/images/image1.png)

It is quite common to use the Logistic sigmoid function as an Activation function. It is more advisable as you get into a deep neural network to use it ONLY for the Output layers. Logistic regression takes an input, passes it through a sigmoid function, then returns an output of probability between 0 and 1. This sigmoid function is responsible for classifying the input.

![Sigmoid Function](https://raw.githubusercontent.com/wayne540/Logistics-Regression-with-Neural-Network-mindset/main/images/image2.png)

Ok, I think some visible codes will reveal the notion much better.
To get a good intuition through practical, I will be using a problem set from Andrew Ng Deep Learning AI. We will be building a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat. Before I go forward, you will need to have some basic understanding of the following or else it might seem like a foreign language.
## Prerequisites
•	Image processing

•	[Python](https://www.python.org/doc/)

•	[NumPy](https://numpy.org/doc/)

•	Linear algebra

•	Calculus

I will try to give a step by step instructions for the coding of the problem set. The full code can also be found on my Github, the link will be attached. First, we want to import all the necessary packages.

![Sigmoid Function](https://raw.githubusercontent.com/wayne540/Logistics-Regression-with-Neural-Network-mindset/main/images/image3.png)

Next is to preprocess the data. I will be using an h5 file containing images of cats and other images that are non-cats. [Cat Dataset on Kaggle](https://www.kaggle.com/datasets/crawford/cat-dataset)
 to the Cat dataset. We are working with 209 training and 50 test examples, and a pixels value of 64 x 64 x 3, which is not much to let our model reach its full potential to learn, but it will do for the sake of intuition. The number I just stated are the values that represent the images of the dataset. The images will be the input we feed into our logistics regression. An image is a 3-dimensional matrix that holds pixel intensity values of Red, Green, and Blue channel. In order to feed it into our network, we will convert this image (3d-matrix ) to a 1d-matrix, vector.
A single train image is of dimension [64, 64, 3] where 64, is the width, 64 is the height and 3 is the number of channels, when flattened to a 1-dim image, we get [1, 12288]. The image below should help you understand better.

![Sigmoid Function](https://raw.githubusercontent.com/wayne540/Logistics-Regression-with-Neural-Network-mindset/main/images/image4.png)

Moving forward, we know a flattened vector will have a dimension [1, 12288], note that we have 209 training examples and 50 test examples. Thus we will end up having:

•	An array of dimension [12288,209] to hold our train images.

•	An array of dimension[12288,50] to hold our test images.

•	An array of dimension [1,209] to hold our train labels.

•	An array of dimension [1,50]to hold our test labels.

After flattening our image, we Standardize. One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you subtract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel). A quick recap of the first segment : Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, …). Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1). “Standardize” the data.

![Sigmoid Function](https://raw.githubusercontent.com/wayne540/Logistics-Regression-with-Neural-Network-mindset/main/images/image5.png)

Our data has been preprocessed. Let’s get to building our logistics regression model that will classify cats. Remember the idea here is to understand the mindset of Neural Networks using Logistics Regression. LR is pretty much a one-layer neural network

![Sigmoid Function](https://raw.githubusercontent.com/wayne540/Logistics-Regression-with-Neural-Network-mindset/main/images/image6.png)

Let me explains some basic concepts in regard to the above image.
Neuron- a neuron(where we have w x +b, within the circle in the above image) in deep learning, is a biologically inspired representation of a neuron inside the human brain. Similar to a neuron in the human brain, an artificial neuron accepts inputs from some other neurons and fires a value to the next set of artificial neurons. Inside a single neuron, two computations are performed. Weighted and Activation.
Weighted sum- Every input value x to an artificial neuron has a weight w attached to it, which tells about the relative importance of that input with other inputs. Each weight is multiplied with its corresponding input and gets summed up to produce a single value z.
Activation- After computing the weighted sum z, an activation function a=g(z) is applied to this weighted sum z. An activation function is a simple mathematical transformation of an input value to an output value by introducing a non-linearity. This is necessary because real-world inputs are non-linear and we need our neural network to learn this non-linearity somehow.
Logistics Regression Concept: These are the steps we will be carrying out to build the model algorithm
###  Mathematical Expression for the Algorithm

Initialize the weights `w` and biases `b` to random values (say 0 or using random distribution).

For each training sample in the dataset:

- Calculate the output value `( a^{(i)} )` for an input sample `(x^{(i)})`.
- First: find out the weighted sum `( z^{(i)})`.
- Second: compute the activation value `( a^{(i)} = y'^{(i)} = g(z^{(i)}))` for the weighted sum `( z^{(i)})`.
- As we know the true label for this input training sample \( y^{(i)} \), we use that to find the loss `( L(a^{(i)}, y^{(i)}))`.

Calculate the cost function `( J )`, which is the sum of all losses divided by the number of training examples `(m)`, i.e.,

`J = (1/m) * Σ L(a(i), y(i))`

To minimize the cost function, compute the gradients for parameters `dJ/dw`  and `dJ/db` using the chain rule of calculus.

Use gradient descent to update the parameters **w** and **b**.

Perform the above procedure until the cost function becomes minimum.

---

####  The mathematical expressions for our algorithm:

1. **Weighted Sum of the ith training example:**

`z(i) = wᵀx(i) + b`

Here “\( .T \)” stands for Transpose.

2. **Activation of the ith training example (using sigmoid):**

`y′(i) = a(i) = σ(z(i)) = 1 / (1 + e^(−z(i)))`

3. **Loss function of the ith training example:**

`L(a(i), y(i)) = −y(i)log(a(i)) − (1 − y(i))log(1 − a(i))`

4. **Cost function for all training examples:**

`J = (1/m) * Σ L(a(i), y(i))`

`J = −(1/m) * Σ [ y(i)log(a(i)) + (1 − y(i))log(1 − a(i)) ]`

5. ** Gradient Descent w.r.t cost function, weights and bias **

`dJ/dw = (1/m) * X(A − Y)ᵀ`

`dJ/db = (1/m) * Σ (a(i) − y(i))`

6. ** Parameters update rule **
   
`w = w − α * (dJ/dw)`

`b = b − α * (dJ/db)`
Note, we will be implementing vectorization, meaning instead of looping through each 209 training example, which will slow thing down, we will combine them, using numpy library. The below image should give a better understanding how dimensions of the numpy array should like.

![Sigmoid Function](https://raw.githubusercontent.com/wayne540/Logistics-Regression-with-Neural-Network-mindset/main/images/image7.png)



