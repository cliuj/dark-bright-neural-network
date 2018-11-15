import math

debug = True

learning_rate = 0.1
targetG = 1
targetH = 0

input_neuron = {
    'A': 10,
    'B': 30,
    'C': 20,
    'D': 0,

}

weight = {
    'A to E': 0.2, 
    'B to E': -0.1,
    'C to E': 0.4,
    'D to E': 0,
    
    'A to F': 0.7, 
    'B to F': -1.2,
    'C to F': 1.2,
    'D to F': 0,

    'E to G': 1.1,
    'F to G': 0.1,
    'E to H': 3.1,
    'F to H': 1.17
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
    sigma = (input_neuron['A'] * weight['A to E']) + (input_neuron['B'] * weight['B to E']) + (input_neuron['C'] * weight['C to E'])# + (input_neuron['D'] * weight['D to E'])
    return sigmoid(sigma)

def calc_F_output():
    sigma = (input_neuron['A'] * weight['A to F']) + (input_neuron['B'] * weight['B to F']) + (input_neuron['C'] * weight['C to F'])# + (input_neuron['D'] * weight['D to F'])
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
    # calculate the error term 
    G_error_term = output_neuron['G'] * (1 - output_neuron['G']) * (targetG - output_neuron['G'])
    H_error_term = output_neuron['H'] * (1 - output_neuron['H']) * (targetH - output_neuron['H'])

    if debug:
        print('Error term G: {}, H: {}'.format(G_error_term, H_error_term))
    
    # calculate the hidden layer error term
    E_error_term = ((weight['E to G'] * G_error_term) + (weight['E to H'] * H_error_term)) * (hidden_neuron['E'] * (1 - hidden_neuron['E']))
    F_error_term = ((weight['F to G'] * G_error_term) + (weight['F to H'] * H_error_term)) * (hidden_neuron['F'] * (1 - hidden_neuron['F']))

    if debug:
        print('Error term E: {}, F: {}'.format(E_error_term, F_error_term))

    # change the weights between the input and hidden layers'
    weight['A to E'] = weight['A to E'] + learning_rate * E_error_term * input_neuron['A']
    weight['B to E'] = weight['B to E'] + learning_rate * E_error_term * input_neuron['B']
    weight['C to E'] = weight['C to E'] + learning_rate * E_error_term * input_neuron['C']
    weight['D to E'] = weight['D to E'] + learning_rate * E_error_term * input_neuron['D']

    
    weight['A to F'] = weight['A to F'] + learning_rate * F_error_term * input_neuron['A']
    weight['B to F'] = weight['B to F'] + learning_rate * F_error_term * input_neuron['B']
    weight['C to F'] = weight['C to F'] + learning_rate * F_error_term * input_neuron['C']
    weight['D to F'] = weight['D to F'] + learning_rate * F_error_term * input_neuron['D']
    

    if debug:
        print("New weights A to E: {}, B to E: {}, C to E: {}".format(weight['A to E'], weight['B to E'], weight['C to E']))
        print("New weights A to F: {}, B to F: {}, C to F: {}".format(weight['A to F'], weight['B to F'], weight['C to F']))


forward_propagation()
backward_propagation()
