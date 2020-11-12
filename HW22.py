# Code fpr the solution of homework task 2.2 to check for linear separability

# imported packages
import numpy as np
import random

# functions and classes


def signum(x):
    if x > 0:
        y = 1
    else:
        y = -1
    return y


def derivative_tanh(x):
    output = (1 - np.tanh(x)**2)
    return output


def return_energy(y_model, y_true):
    sum_energy = y_model - y_true
    energy = 0.5 * sum_energy
    return energy


def return_deriv_energy(y_model, y_true):
    output = 2*(y_true - y_model)*(-1)
    return output


class Perceptron:

    def __init__(self, dim=4):
        self.dim = dim
        self.input = np.zeros((dim, 1))
        self.output = None
        self.w = np.random.uniform(-0.2, 0.2, size=(4, 1))
        self.bias = random.uniform(-1, 1)
        self.d_w = np.zeros((4,1))
        self.d_bias = 0
        self.loss = 0
        self.d_loss = 0
        self.b = 0

    def calc_output(self, input):
        self.input = input
        b = 0.5*(np.dot(self.w.T, self.input) - self.bias)
        self.b = b
        self.output = int(np.tanh(b))

    def calc_derivative(self):
        self.d_w = (1 + np.tanh(self.b)) * 0.5 * self.input
        self.d_bias = (1 + np.tanh(self.b)) * (-0.5)

    def calc_loss(self, y_real):
        self.loss = 0.5*(y_real - self.output)**2
        self.d_loss = (y_real - self.output)*(-1)

    def backprob(self, learning_rate=0.02):
        self.calc_derivative()
        delta_w = self.d_loss*self.d_w
        delta_bias = self.d_loss*self.d_bias
        self.w = self.w - delta_w*learning_rate
        self.bias = self.bias - delta_bias*learning_rate

    def update_neuron(self, x_sample, y_sample):
        self.calc_output(x_sample)
        self.calc_loss(y_sample)
        self.backprob()

# create the input arrays for the training
a1 = np.array([[-1],[-1],[-1],[-1]])
a2 = np.array([[1],[-1],[-1],[-1]])
a3 = np.array([[-1],[1],[-1],[-1]])
a4 = np.array([[-1],[-1],[1],[-1]])
a5 = np.array([[-1],[-1],[-1],[1]])
a6 = np.array([[1],[-1],[-1],[-1]])
a7 = np.array([[1],[-1],[1],[-1]])
a8 = np.array([[1],[-1],[-1],[1]])
a9 = np.array([[-1],[1],[1],[-1]])
a10 = np.array([[-1],[1],[-1],[1]])
a11 = np.array([[-1],[-1],[1],[1]])
a12 = np.array([[1],[1],[1],[-1]])
a13 = np.array([[1],[1],[-1],[-1]])
a14 = np.array([[1],[-1],[1],[1]])
a15 = np.array([[-1],[1],[1],[1]])
a16 = np.array([[1],[1],[1],[1]])

x_train = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16]

# Training process

# define the output for the training
y_train_A = [1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1]
y_train_B = [-1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1]
y_train_C = [-1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1]
y_train_D = [1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1]
y_train_E = [1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1]
y_train_F = [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1]

y_train_list = [y_train_A, y_train_B, y_train_C, y_train_D, y_train_E, y_train_F]
# define the number of iterations for the training process
n_iterations = 10**5
n_rounds = 10

for train_sets in y_train_list:

    print('the training set ' + str(train_sets) + ' was chosen!')
    y_chosen = train_sets
    list_n_correct_patterns = []
    for k in range(n_rounds):

        my_perceptron = Perceptron(4)

        for i in range(n_iterations):
            random_index = random.randint(0, 15)
            x_sample = x_train[random_index]
            y_sample = y_chosen[random_index]
            my_perceptron.update_neuron(x_sample, y_sample)

        # store the number of correct predicted patterns
        n_correct_patterns = 0

        for idx, element in enumerate(x_train):

            my_perceptron.calc_output(element)
            model_output = my_perceptron.output
            model_output = signum(model_output)
            if model_output == y_chosen[idx]:
                n_correct_patterns += 1

        # print('{} patterns are correct'.format(n_correct_patterns))
        list_n_correct_patterns.append(n_correct_patterns)
        for number in list_n_correct_patterns:
            if number == 16:
                print('the dataset is linear separable!')
                break




