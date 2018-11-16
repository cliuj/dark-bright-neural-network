import math

debug = False

learning_rate = 0.1
target = 1
not_target = 0


def parse_file(to_parse_file):
    return [line.rstrip('\n') for line in open(to_parse_file)]


# credit to arronasterling on SO for this blank line skipper
def nonblank_lines(_file):
    for l in _file:
        line = l.rstrip()
        if line:
            yield line

def retrieve_inputs(inputs_file):
    parsed_file = parse_file(inputs_file)

    input_layers = []
    for input_layer in parsed_file:
        neurons = []
        for bit in list(input_layer):
            neurons.append(int(bit))
        input_layers.append(neurons)

    return input_layers


def load_inputs_from(input_layer):
    input_neuron['A'] = input_layer[0]
    input_neuron['B'] = input_layer[1]
    input_neuron['C'] = input_layer[2]
    input_neuron['D'] = input_layer[3]

def write_new_weights_to(new_weights_file, old_weights):
    write_weights_to = open(new_weights_file, 'w')
    
    for key, value in old_weights.items():
        write_weights_to.write("{}\n".format(value))

def load_weights_from(new_weights_file):
    parsed_file = parse_file(new_weights_file)
    line = 0
    for key, value in weight.items():
        weight[key] = float(parsed_file[line])
        line+=1




input_neuron = {
    'A': 0,
    'B': 0,
    'C': 0,
    'D': 0,
}

weight = {
    'A to E': 0.2, 
    'B to E': 0.1,
    'C to E': 0.4,
    'D to E': 0.5,
    
    'A to F': 0.7, 
    'B to F': 0.2,
    'C to F': 0.2,
    'D to F': 0.1,

    'E to G': 0.4,
    'F to G': 0.1,
    'E to H': 0.2,
    'F to H': 0.1
}

old_weight = {
    'A to E': 0, 
    'B to E': 0,
    'C to E': 0,
    'D to E': 0,
    
    'A to F': 0, 
    'B to F': 0,
    'C to F': 0,
    'D to F': 0,

    'E to G': 0,
    'F to G': 0,
    'E to H': 0,
    'F to H': 0
}

hidden_neuron = {
    'E': 0,
    'F': 0,
}

output_neuron = {
    'G':0,
    'H':0
}


def forward_propagation():
    # calculate the hidden layer neurons:
    hidden_neuron['E'] = calc_E_output()
    hidden_neuron['F'] = calc_F_output()
    
    if debug:
        print('Hidden neuron E: {}, F {}'.format(hidden_neuron['E'], hidden_neuron['F']))
    
    # calculate the output lyaer neurons:
    output_neuron['G'] = calc_G_output()
    output_neuron['H'] = calc_H_output()

    if debug:
        print('Output neuron G: {}, H: {}'.format(output_neuron['G'], output_neuron['H']))

def calc_E_output():
    sigma = (input_neuron['A'] * weight['A to E']) + (input_neuron['B'] * weight['B to E']) + (input_neuron['C'] * weight['C to E']) + (input_neuron['D'] * weight['D to E'])
    return sigmoid(sigma)

def calc_F_output():
    sigma = (input_neuron['A'] * weight['A to F']) + (input_neuron['B'] * weight['B to F']) + (input_neuron['C'] * weight['C to F']) + (input_neuron['D'] * weight['D to F'])
    return sigmoid(sigma)

def calc_G_output():
    sigma = (hidden_neuron['E'] * weight['E to G']) + (hidden_neuron['F'] * weight['F to G'])
    return sigmoid(sigma)

def calc_H_output():
    sigma = (hidden_neuron['E'] * weight['E to H']) + (hidden_neuron['F'] * weight['F to H'])
    return sigmoid(sigma)

def sigmoid(s):
    return 1.0/(1.0 + math.exp(-s))

def backward_propagation():
    # copy the current weights to old_weight dict
    for key, value in weight.items():
        old_weight[key] = value
    
    
    pixels_sum = 0
    for key, value in input_neuron.items():
        pixels_sum += value
    
    # switch the outputs' targets based on the number of bright pixels in the inputs
    if pixels_sum > 1:
        G_error_term = output_neuron['G'] * (1 - output_neuron['G']) * (target - output_neuron['G'])
        H_error_term = output_neuron['H'] * (1 - output_neuron['H']) * (not_target - output_neuron['H'])
    else:
        G_error_term = output_neuron['G'] * (1 - output_neuron['G']) * (not_target - output_neuron['G'])
        H_error_term = output_neuron['H'] * (1 - output_neuron['H']) * (target - output_neuron['H'])
    
    
    if debug:
        print('Error term G: {}, H: {}'.format(G_error_term, H_error_term))
    
    # calculate the hidden layer error term
    E_error_term = ((weight['E to G'] * G_error_term) + (weight['E to H'] * H_error_term)) * (hidden_neuron['E'] * (1 - hidden_neuron['E']))
    F_error_term = ((weight['F to G'] * G_error_term) + (weight['F to H'] * H_error_term)) * (hidden_neuron['F'] * (1 - hidden_neuron['F']))

    if debug:
        print('Error term E: {}, F: {}'.format(E_error_term, F_error_term))

    # change the weights between the input and hidden layers

    weight['A to E'] = old_weight['A to E'] + (learning_rate * E_error_term * input_neuron['A']) 
    weight['B to E'] = old_weight['B to E'] + (learning_rate * E_error_term * input_neuron['B'])
    weight['C to E'] = old_weight['C to E'] + (learning_rate * E_error_term * input_neuron['C'])
    weight['D to E'] = old_weight['D to E'] + (learning_rate * E_error_term * input_neuron['D'])

    
    weight['A to F'] = old_weight['A to F'] + (learning_rate * F_error_term * input_neuron['A'])
    weight['B to F'] = old_weight['B to F'] + (learning_rate * F_error_term * input_neuron['B'])
    weight['C to F'] = old_weight['C to F'] + (learning_rate * F_error_term * input_neuron['C'])
    weight['D to F'] = old_weight['D to F'] + (learning_rate * F_error_term * input_neuron['D'])

    if debug:
        print("New weights A to E: {}, B to E: {}, C to E: {}, D to E: {}".format(weight['A to E'], weight['B to E'], weight['C to E'], weight['D to E']))
        print("New weights A to F: {}, B to F: {}, C to F: {}, D to F: {}".format(weight['A to F'], weight['B to F'], weight['C to F'], weight['D to F']))

    # change the weights between the hidden and output layers

    weight['E to G'] = old_weight['E to G'] + (learning_rate * G_error_term * hidden_neuron['E'])
    weight['F to G'] = old_weight['F to G'] + (learning_rate * G_error_term * hidden_neuron['F'])
    weight['E to H'] = old_weight['E to H'] + (learning_rate * H_error_term * hidden_neuron['E'])
    weight['F to H'] = old_weight['F to H'] + (learning_rate * H_error_term * hidden_neuron['F'])
    



inputs = retrieve_inputs("inputs_data.txt")

def print_answer():
    if output_neuron['G'] > output_neuron['H']:
        print("bright")
    else:
        print("dark")

def train():
    for i in range(0, 1):
        for index in range(0, 16):
            load_inputs_from(inputs[index])
            forward_propagation()
            print(output_neuron, end = " ")
             
            backward_propagation()
            write_new_weights_to("new_weights.txt", weight)
    
    
    for i in range(0, 1000):
        for index in range(0, 16):
            load_inputs_from(inputs[index])
            load_weights_from("new_weights.txt")
            forward_propagation()
            print(output_neuron, end = " ")
            print_answer()
            backward_propagation()
            write_new_weights_to("new_weights.txt", weight)
        print()


# test to see if the neural network can answer correctly
def test(t):
    load_inputs_from(inputs[t])
    load_weights_from("trained_weights.txt")
    forward_propagation()
    print(output_neuron, end = " ")
    print("Input: ", end = " ")
    print(inputs[t], end = " ")
    print("Prediction: ", end = " ")
    print_answer()

# user input testing
while(True):
    t = input("Enter the index of the inputs: ")
    if t == "q":
        print("Exiting . . .")
        break
    elif int(t) > 15:
        print("Too large of a number")
    else:
        test(int(t))

