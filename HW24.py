# Code for homework task 2.4 to train a 2 layer neural network
# Imported packages
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


def signum(x):
    if x > 0:
        output = 1
    else:
        output = -1
    return output


def derivative_tanh(x):
    output = (1 - np.tanh(x)**2)
    return output


def return_energy(y_model, y_true):
    sum_energy = y_model - y_true
    energy = 0.5 * sum_energy
    return energy


def return_deriv_energy(y_model, y_true):
    output = 2*(y_true - y_model)
    return output


class NetworkLayer:

    def __init__(self, number_neurons, input_dim=4):
        self.number_neurons = number_neurons
        self.weights = np.random.uniform(-1, 1, size=(number_neurons, input_dim))
        self.biases = np.random.uniform(-1, 1, size=(number_neurons, 1))
        self.output = np.zeros((input_dim, 1))
        self.d_weights = np.zeros((input_dim, number_neurons))
        self.d_biases = 0
        self.input = np.zeros((input_dim, 1))
        self.d_input = np.zeros((input_dim, 1))
        self.d_activation = None

    def calc_output(self, input):
        self.input = input
        b = (np.dot(self.weights, input) - self.biases)
        self.output = np.copy(np.tanh(b))
        return np.tanh(b)

    def calc_derivative(self):
        self.d_activation = np.copy(1 - np.tanh(self.output) ** 2)
        self.d_weights = np.copy(self.input)
        self.d_biases = -1
        self.d_input = np.copy(self.weights)

    def backprob(self, d_next_layer, learning_rate=0.02):

        self.calc_derivative()
        self.weights = self.weights - learning_rate * np.dot((self.d_activation * d_next_layer), self.d_weights.T)
        self.biases = self.biases - learning_rate * self.d_biases * self.d_activation * d_next_layer
        self.d_input = np.dot(self.d_input.T, (self.d_activation * d_next_layer))
        return self.d_input


class NeuralNetwork:

    def __init__(self, n_hidden_layers=2, neurons_per_layer=[10, 10], input_dim=2):
        self.layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                self.layers.append(NetworkLayer(neurons_per_layer[i], input_dim=input_dim))
            else:
                self.layers.append(NetworkLayer(neurons_per_layer[i], input_dim=neurons_per_layer[i-1]))

        self.layers.append(NetworkLayer(1, input_dim=neurons_per_layer[-1]))
        self.input = np.zeros((input_dim, 1))
        self.output = 0
        self.loss = 0
        self.d_loss = 0
        self.learning_rate = 0.02

    def calc_output(self, input_vector):
        self.input = np.copy(input_vector)
        temp_output = None
        for i in range(len(self.layers)):
            if i == 0:
                temp_output = np.copy(self.layers[i].calc_output(input_vector))
            else:
                temp_output = np.copy(self.layers[i].calc_output(temp_output))
        last_output = float(temp_output)
        self.output = np.copy(last_output)
        return last_output

    def calc_loss(self, y_real):
        self.loss = 0.5*(y_real - self.output)**2
        self.d_loss = (y_real - self.output)*(-1)

    def backprob(self, y_true):
        self.calc_loss(y_true)
        d_next_layer = None
        for i in range(len(self.layers)-1, -1, -1):
            # print(i)
            if i == len(self.layers)-1:
                d_next_layer = self.layers[i].backprob(np.array(self.d_loss))
            else:
                d_next_layer = self.layers[i].backprob(d_next_layer)

    def calc_val_loss(self, y_true):
        temp_loss = float(abs(signum(self.output) - y_true))
        return temp_loss


# import the training and validation data
training_file = 'training_set.csv'
validation_file = 'validation_set.csv'

# store the data sets in a dataframe
training_df = pd.read_csv('./' + training_file, header=None)
validation_df = pd.read_csv('./' + validation_file, header=None)

# separate input and output data
x_train = training_df.iloc[:, :-1].values
y_train = training_df.iloc[:, -1].values.reshape(-1,1)
x_val = validation_df.iloc[:, :-1].values
y_val = validation_df.iloc[:, -1].values.reshape(-1,1)

# Prepare the neural network
test_NN = NeuralNetwork()
n_epochs = 600
n_iterations = len(x_train)
val_error_list = []
order_list = [i for i in range(len(x_train))]
random.shuffle(order_list)
prev_loss = 1
curr_loss = 0

for i in range(n_epochs):
    print(i)
    random.shuffle(order_list)
    loss_list = []
    temp_NN = test_NN
    for number in order_list:
        n_random = np.random.randint(n_iterations)
        x_test = x_train[number, :].reshape(-1, 1)
        y_test = y_train[number, :].reshape(-1, 1)

        test_NN.calc_output(x_test)
        loss_list.append(test_NN.loss)
        test_NN.backprob(y_test)

    curr_loss = float((sum(loss_list))/len(x_train))
    print('the loss of the model is {}'.format(float((sum(loss_list))/len(x_train))))
    sum_error = 0
    pred_list = []
    prev_val_error = 1
    for j in range(len(x_val)):

        x_val_sample = x_val[j, :].reshape(-1, 1)
        y_val_sample = y_val[j, :].reshape(-1, 1)
        test_NN.calc_output(x_val_sample)
        curr_val_loss = test_NN.calc_val_loss(y_val_sample)
        pred_list.append(test_NN.output)
        sum_error += curr_val_loss

    val_set_error = sum_error/(2*len(x_val))
    val_error_list.append(float(val_set_error))
    print('val error is: {}'.format(val_set_error))
    if val_set_error < 0.12:
        print('good model found')
        break

plt.plot(val_error_list)
plt.show()
#%%
# create function to extract the weights and store them in a csv


def get_weights_and_bias(layer):
    weights = layer.weights
    df_weights = pd.DataFrame(weights)
    bias = layer.biases
    df_biases = pd.DataFrame(bias)
    return df_weights, df_biases


first_layer_weights, first_layer_bias = get_weights_and_bias(test_NN.layers[0])
second_layer_weights, second_layer_bias = get_weights_and_bias(test_NN.layers[1])
third_layer_weights, third_layer_bias = get_weights_and_bias(test_NN.layers[2])

#%%
# change the third layer weights matrix for the correct shape
third_layer_weights = third_layer_weights.T

# store them as csv_files
first_layer_weights.to_csv('w1.csv', index=False, header=False)
second_layer_weights.to_csv('w2.csv', index=False, header=False)
third_layer_weights.to_csv('w3.csv', index=False, header=False)

# store the bias as csv_files
first_layer_bias.to_csv('t1.csv', index=False, header=False)
second_layer_bias.to_csv('t2.csv', index=False, header=False)
third_layer_bias.to_csv('t3.csv', index=False, header=False)
