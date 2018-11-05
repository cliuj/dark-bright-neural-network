import math

def sigmoid_squash(s):
    sigmoid = 1.0 / (1.0 + math.exp(-s))
    return sigmoid


def sigma(inputs, weights):
    sigma = 0
    for _input, weight in zip(inputs, weights):
        sigma += _input.val * weight.val

    return sigma


class InputNeuron:
    def __init__(self, val):
        self.val = val


class Weight:
    def __init__(self, val):
        self.val = val


class HiddenNeuron:
    def __init__(self):
        self.val = 0;
        self.backward_val = 0;

    def calc_forward(self, inputs, weights):
        self.val = sigmoid_squash(sigma(inputs, weights))

    def calc_backward(self, output_layer, weights):

        # sigma(weights + output_error_terms)
        sigma = 0
        for output_neuron, weight in zip(output_layer, weights):
            sigma += output_neuron.error_term * weight.val
            print("output_neuron error term: {}, weight: {}".format(output_neuron.error_term, weight.val))    
        print("sigma: {}".format(sigma))
        self.h = sigma * (self.val * (1 - self.val))
        print("E: {}, 1-E: {}".format(self.val, 1 - self.val))
        print("H: {}".format(self.h))

class OutputNeuron:
    def __init__(self):
        self.val = 0
        self.backward_val = 0

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

        i = 1
        for hidden_neuron in self.hidden_layer:
            print("hidden {}: {}".format(i, hidden_neuron.val))

        # hidden -> output layer output:
        self.test_set.output_layer[0].calc_forward(self.hidden_layer, self.test_set.h_o_weights1)
        self.test_set.output_layer[1].calc_forward(self.hidden_layer, self.test_set.h_o_weights2)


        i = 1 
        for output_neuron in self.output_layer:
            print("output {}: {}".format(i, output_neuron.val))
            i+=1

    def backward_propagation(self):

        # output -> hidden layer:
        self.output_layer[0].calc_error_term(self.test_set.target1)
        self.output_layer[1].calc_error_term(self.test_set.target2)

        i = 1
        for output_neuron in self.output_layer:
            print("output {} error term: {}".format(i, output_neuron.error_term))
            i += 1

        backward_o_h_weights1 = [self.test_set.h_o_weights1[0], self.test_set.h_o_weights2[0]]
        backward_o_h_weights2 = [self.test_set.h_o_weights1[1], self.test_set.h_o_weights2[1]]

        self.hidden_layer[0].calc_backward(self.output_layer, backward_o_h_weights1)
        self.hidden_layer[1].calc_backward(self.output_layer, backward_o_h_weights2)

        i = 1
        for hidden_neuron in self.hidden_layer:
            print("hidden {} error term: {}".format(i, hidden_neuron.h))
            i += 1

        

class TestSet:
    def __init__(self):
        # inputs
        # self.input_layer = [InputNeuron(0), InputNeuron(1), InputNeuron(1), InputNeuron(0)]
        self.input_layer = [InputNeuron(10), InputNeuron(30), InputNeuron(20)]
        
        # inputs -> hidden weights
        # self.i_h_weights1 = [Weight(0.2), Weight(0.3), Weight(0.5), Weight(0.1)]
        # self.i_h_weights2 = [Weight(1.1), Weight(0.4), Weight(0.1), Weight(0.6)]
        self.i_h_weights1 = [Weight(0.2), Weight(-0.1), Weight(0.4)]
        self.i_h_weights2 = [Weight(0.7), Weight(-1.2), Weight(1.2)]

        # hidden layer
        self.hidden_layer = [HiddenNeuron(), HiddenNeuron()]

        # hidden -> output weights
        # self.h_o_weights1 = [Weight(0.3), Weight(0.4)]
        # self.h_o_weights2 = [Weight(0.1), Weight(0.7)]
        self.h_o_weights1 = [Weight(1.1), Weight(0.1)]
        self.h_o_weights2 = [Weight(3.1), Weight(1.17)]

        # output layer
        self.output_layer = [OutputNeuron(), OutputNeuron()]

        # output targets
        self.target1 = 1
        self.target2 = 0

ann = ANN(TestSet())
ann.forward_propagation()
ann.backward_propagation()
