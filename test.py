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

    def calc_backward(self, output_layer, weight):
        self.error_term = weight.val * (self.val * (1 - self.val))

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
        self.input_layer = test_set.input_layers[0]
        self.hidden_layer = test_set.hidden_layer
        self.output_layer = test_set.output_layer
    
    def change_input_layer(self, input_index):
        self.input_layer = self.test_set.input_layers[input_index]

    def forward_propagation(self):

        # input -> hidden layer output:
        self.hidden_layer[0].calc_forward(self.input_layer, self.test_set.i_h_weights1)
        self.hidden_layer[1].calc_forward(self.input_layer, self.test_set.i_h_weights2)
        self.hidden_layer[2].calc_forward(self.input_layer, self.test_set.i_h_weights3)


        if debug:
            print("forward propagation hidden layer:")
            print_val(self.hidden_layer)

        # hidden -> output layer output:
        self.test_set.output_layer[0].calc_forward(self.hidden_layer, self.test_set.h_o_weights1)

        print("forward propagation output layer:")
        print_val(self.output_layer)

    def backward_propagation(self):

        # output -> hidden layer:
        self.output_layer[0].calc_error_term(self.test_set.target1)

        if debug:
            print_error_terms(self.output_layer)

        # calculate the error terms for the hidden layer
        backward_o_h_weights1 = [self.test_set.h_o_weights1[0], self.test_set.h_o_weights1[1], self.test_set.h_o_weights1[2]]

        self.hidden_layer[0].calc_backward(self.output_layer, backward_o_h_weights1[0])
        self.hidden_layer[1].calc_backward(self.output_layer, backward_o_h_weights1[1])
        self.hidden_layer[2].calc_backward(self.output_layer, backward_o_h_weights1[2])

        if debug:
            print_error_terms(self.hidden_layer)


        # calculate the hidden -> output weight changes
        for weight in self.test_set.h_o_weights1:
            weight.change_weight(self.test_set.learning_rate, self.output_layer[0].error_term, self.output_layer[0].val)

        # printout the new hidden -> output weight values
        if debug:
            print_weights(self.test_set.h_o_weights1)
            print_weights(self.test_set.h_o_weights2)

        # calculate the input -> hidden weight changes
        for weight in self.test_set.i_h_weights1:
            weight.change_weight(self.test_set.learning_rate, self.hidden_layer[0].error_term, self.hidden_layer[0].val)
        
        for weight in self.test_set.i_h_weights2:
            weight.change_weight(self.test_set.learning_rate, self.hidden_layer[1].error_term, self.hidden_layer[1].val)

        for weight in self.test_set.i_h_weights3:
            weight.change_weight(self.test_set.learning_rate, self.hidden_layer[2].error_term, self.hidden_layer[2].val)

        # printout the new input -> hidden weight values
        if debug:
            print_weights(self.test_set.i_h_weights1)
            print_weights(self.test_set.i_h_weights2)

    def manage_target(self):
        sum = 0;
        for _input in self.input_layer:
            sum += _input.val 
        if sum > 1:
            self.test_set.target1 = 1
        else:
            self.test_set.target1 = 0

    def get_prediction(self):
        if self.output_layer[0].val > 0.5:
            return "bright"
        else:
            return "dark"

    def save_weights(self):
        f = open("output_weights.txt", 'w')
        for i in range(0, 4):
            f.write("{} ".format(self.test_set.i_h_weights1[i].val))
        f.write("\n")
        
        for i in range(0, 4):
            f.write("{} ".format(self.test_set.i_h_weights2[i].val))
        f.write("\n")

        for i in range(0, 4):
            f.write("{} ".format(self.test_set.i_h_weights3[i].val))
        f.write("\n")
        
        for i in range(0, 3):
            
            f.write("{} ".format(self.test_set.h_o_weights1[i].val))
        f.write("\n")


class TestSet:
    def __init__(self):
        # inputs
        # self.input_layer = [InputNeuron(0), InputNeuron(1), InputNeuron(1), InputNeuron(0)]
        self.input_data = self.parse_file("input_data.txt")
        self.input_layers = self.generate_input_layers(self.input_data)
        
        
        #for input_layer in self.input_layers:
        #    print_val(input_layer)
        
        
        
        self.weights = self.parse_file("initial_weights.txt")
        #self.weights = self.parse_file("output_weights.txt")
        self.generate_weights(self.weights)
        
        if debug:
            print_weights(self.i_h_weights1)
            print_weights(self.i_h_weights2)
            print_weights(self.h_o_weights1)
        
        
        
        # inputs -> hidden weights
        # hidden layer
        self.hidden_layer = [HiddenNeuron(), HiddenNeuron(), HiddenNeuron()]

        # hidden -> output weights

        # output layer
        self.output_layer = [OutputNeuron()]
        
        # output targets
        self.target1 = 0 

        # learning rate
        self.learning_rate = 1

    def parse_file(self, file):
        input_data = [line.rstrip('\n') for line in open(file)]

        #print(input_data)
        #print((input_data[0])[0])

        return input_data

    def generate_input_layers(self, input_data):
        input_layers = []
        for data in input_data:
            input_layer = []
            for bit in list(data):
                input_layer.append(InputNeuron(int(bit)))
            input_layers.append(input_layer)

        return input_layers
    
    def generate_weights(self, weights):
        weight_layers = []
        self.i_h_weights1 = []
        self.i_h_weights2 = []
        self.i_h_weights3 = []
        self.h_o_weights1 = []

        for weight_section in weights:
            # split the floats
            weight_layers.append(weight_section.split())
        
        
        # add input to hidden layer weights
        for i in range(0,4):
            self.i_h_weights1.append(Weight(float((weight_layers[0])[i])))
            self.i_h_weights2.append(Weight(float((weight_layers[1])[i])))
            self.i_h_weights3.append(Weight(float((weight_layers[2])[i])))
        
        # add hidden to output layer weights
        for i in range(0,3):
            self.h_o_weights1.append(Weight(float((weight_layers[2])[i])))

ann = ANN(TestSet())
i = 0
for x in range(0, 1000):
    for layer in range(0, 16):
        #print("iteration: {}".format(i))
        ann.change_input_layer(layer)
        ann.manage_target()
        ann.forward_propagation()
        ann.backward_propagation()
        print(ann.get_prediction());
        i+=1
    print()
ann.save_weights()
