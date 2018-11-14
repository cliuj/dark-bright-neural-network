import math

debug = True

# performs sigmoid operation with s as its exponent
def sigmoid(s):
    sigmod = 1.0/(1.0 + math.exp(-s))
    return sigmoid


# returns an array of every line in the file
def parse_file(to_parse_file):
    return [line.rstrip('\n') for line in open(to_parse_file)]






class InputNeuron:
    def __init__(self, val):
        self.val = val

class Weight:
    def __init__(self, val):
        self.val = val

class HiddenNeuron:
    def __init__(self):
        self.val = 0

class OutputNeuron:
    def __init__(self):
        self.val = 0




class ANN:
    def __init__(self):
        pass

    def generate_test_set(self):
        self.input_layers = self.__retrieve_inputs("inputs_data.txt")
        self.input_to_hidden_weights_1 = self.__retrieve_initial_weights("initial_weights.txt")



    # stores the inputs from "initial_weights.txt" into input_layers[]
    # on private because this function should not be called after the
    # first call
    def __retrieve_inputs(self, inputs_file):
        self.input_data = parse_file(inputs_file)
        
        if debug:
            print("input_data: ", self.input_data)

        input_layers = []
        for inputs in self.input_data:
            input_layer = []
            for bit in list(inputs):
                input_layer.append(InputNeuron(int(bit)))
            input_layers.append(input_layer)

        return input_layers

    # retrives the weights stored in "initial_weights.txt"
    def __retrieve_initial_weights(self, weights_file):
        weights_data = parse_file(weights_file)

        if debug:
            print("weights: ", weights_data)

        input_to_hidden1_weights = []
        input_to_hidden2_weights = []
        
        total_weights = []
        for w in range(0, 4):
            total_weights.append(weights_data[w].split())

        if debug:
            print("weights:", total_weights)

        for w in range(0, 4):
            input_to_hidden1_weights.append(Weight(float((total_weights[w])[0])))
            input_to_hidden2_weights.append(Weight(float((total_weights[w])[1])))

        if debug:
            print("input to hidden neuron 1 weights:")
            for w in range(0, 4):
                print(input_to_hidden1_weights[w].val)
            print()
            print("input to hidden neuron 2 weights:")
            for w in range(0, 4):
                print(input_to_hidden2_weights[w].val)



    def print_input_layer(self):
        for input_layer in self.input_layers:
            for input_neuron in input_layer:
                print(input_neuron.val, end="", flush=True)
            print()


ann = ANN()
ann.generate_test_set()

ann.print_input_layer()


