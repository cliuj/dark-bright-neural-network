import math

debug = False



def sigmoid_squash(s):
    sigmoid = 1.0 / (1.0 + math.exp(-s))
    return sigmoid


def sigma(inputs, weights):
    sigma = 0
    for _input, weight in zip(inputs, weights):
        sigma += _input.val * weight.val

    return sigma


def print_val(layer):
    i = 0
    for neuron in layer:
        print("neuron {} val: {}".format(i, neuron.val))
        i+=1

def print_weights(weights):
    i = 0
    for weight in weights:
        print("weight {}: {}".format(i, weight.val))
        i+=1

def print_error_terms(layer):
    i = 0
    for neuron in layer:
        print("neuron {} error_term: {}".format(i, neuron.error_term))
        i+=1


class InputNeuron:
    def __init__(self, val):
        self.val = val

class Weight:
    def __init__(self, val):
        self.val = val

    def change_weight(self, learning_rate, error_term, val):
        weight_change = self.val + (learning_rate * error_term * val)
        self.val = weight_change

class HiddenNeuron:
    def __init__(self):
        self.val = 0;

    def calc_forward(self, inputs, weights):
        self.val = sigmoid_squash(sigma(inputs, weights))

    def calc_backward(self, output_layer, weights):

        sigma = 0
        for output_neuron, weight in zip(output_layer, weights):
            sigma += output_neuron.error_term * weight.val

        self.error_term = sigma * (self.val * (1 - self.val))

class OutputNeuron:
    def __init__(self):
        self.val = 0

    def calc_forward(self, inputs, weights):
        self.val = sigmoid_squash(sigma(inputs, weights))

    def calc_error_term(self, target):
        self.error_term = self.val * (1 - self.val) * (target - self.val)


class ANN:
    def __init__(self, test_set):
        self.test_set = test_set
        self.input_layer = test_set.input_layer
        self.hidden_layer = test_set.hidden_layer
        self.output_layer = test_set.output_layer
    

    def forward_propagation(self):

        # input -> hidden layer output:
        self.hidden_layer[0].calc_forward(self.input_layer, self.test_set.i_h_weights1)
        self.hidden_layer[1].calc_forward(self.input_layer, self.test_set.i_h_weights2)


        if debug:
            print("forward propagation hidden layer:")
            print_val(self.hidden_layer)

        # hidden -> output layer output:
        self.test_set.output_layer[0].calc_forward(self.hidden_layer, self.test_set.h_o_weights1)
        self.test_set.output_layer[1].calc_forward(self.hidden_layer, self.test_set.h_o_weights2)

        print("forward propagation output layer:")
        print_val(self.output_layer)

    def backward_propagation(self):

        # output -> hidden layer:
        self.output_layer[0].calc_error_term(self.test_set.target1)
        self.output_layer[1].calc_error_term(self.test_set.target2)

        if debug:
            print_error_terms(self.output_layer)

        # calculate the error terms for the hidden layer
        backward_o_h_weights1 = [self.test_set.h_o_weights1[0], self.test_set.h_o_weights2[0]]
        backward_o_h_weights2 = [self.test_set.h_o_weights1[1], self.test_set.h_o_weights2[1]]

        self.hidden_layer[0].calc_backward(self.output_layer, backward_o_h_weights1)
        self.hidden_layer[1].calc_backward(self.output_layer, backward_o_h_weights2)

        if debug:
            print_error_terms(self.hidden_layer)


        # calculate the hidden -> output weight changes
        for weight in self.test_set.h_o_weights1:
            weight.change_weight(self.test_set.learning_rate, self.output_layer[0].error_term, self.output_layer[0].val)

        for weight in self.test_set.h_o_weights2:
            weight.change_weight(self.test_set.learning_rate, self.output_layer[1].error_term, self.output_layer[1].val)

        # printout the new hidden -> output weight values
        if debug:
            print_weights(self.test_set.h_o_weights1)
            print_weights(self.test_set.h_o_weights2)

        # calculate the input -> hidden weight changes
        for weight in self.test_set.i_h_weights1:
            weight.change_weight(self.test_set.learning_rate, self.hidden_layer[0].error_term, self.hidden_layer[0].val)
        
        for weight in self.test_set.i_h_weights2:
            weight.change_weight(self.test_set.learning_rate, self.hidden_layer[1].error_term, self.hidden_layer[1].val)

        # printout the new input -> hidden weight values
        if debug:
            print_weights(self.test_set.i_h_weights1)
            print_weights(self.test_set.i_h_weights2)

class TestSet:
    def __init__(self):
        # inputs
        self.input_layer = [InputNeuron(0), InputNeuron(1), InputNeuron(1), InputNeuron(0)]
        
        # inputs -> hidden weights
        self.i_h_weights1 = [Weight(0.2), Weight(0.3), Weight(0.5), Weight(0.1)]
        self.i_h_weights2 = [Weight(1.1), Weight(0.4), Weight(0.1), Weight(0.6)]

        # hidden layer
        self.hidden_layer = [HiddenNeuron(), HiddenNeuron()]

        # hidden -> output weights
        self.h_o_weights1 = [Weight(0.3), Weight(0.4)]
        self.h_o_weights2 = [Weight(0.1), Weight(0.7)]

        # output layer
        self.output_layer = [OutputNeuron(), OutputNeuron()]

        # output targets
        self.target1 = 1
        self.target2 = 0

        # learning rate
        self.learning_rate = 1.2


if __name__ == "__main__":
    ann = ANN(TestSet())
    i = 0
    for x in range(0, 10):
        print("iteration: {}".format(i))
        ann.forward_propagation()
        ann.backward_propagation()
        i+=1
        print()
