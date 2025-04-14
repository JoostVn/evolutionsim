import numpy
from math import e, tanh
from matplotlib import pyplot as plt
import numpy as np

"""
TODO:
    - Improving network visualization (weights are not correct) / 2 color scale
    - Visualizing forward and backward pas
    - Dropping node class, computing forward pass of layer as:
        sigmoid(input_vector * weight_matrix + bias_vector)
        


https://www.youtube.com/watch?v=BBLJFYr7zB8
"""

class Activation:
    
    @staticmethod
    def sigmoid(x):
        return 1/(1+e**(-x))

    @staticmethod
    def tanh(x):
        return tanh(x)

    @staticmethod
    def plot(function, domain):
        x = np.arange(*domain, 0.1)
        f = np.vectorize(function)
        y = f(x)
        plt.figure(figsize=(8,5))
        plt.plot(x,y)
        plt.show()
    

class Node:
    
    def __init__(self, layer, activation_func=None):
        self.layer = layer
        self.activation_func = activation_func
        
    def random_init(self):
        domain = [-1, 1]
        self.weights = np.random.uniform(*domain, self.layer.input_dim)
        self.bias = np.random.uniform(*domain)
    
    def fire(self, input_values):
        input_sum = self.bias + (self.weights * input_values).sum()
        return self.activation_func(input_sum) 
            

class Layer:
    
    def __init__(self, size, input_dim, activation_func=None):
        self.size = size
        self.input_dim = input_dim
        self.activation_func = activation_func
        self.nodes = [Node(self, self.activation_func) for i in range(self.size)]
    
    def random_init(self):
        for node in self.nodes:
            node.random_init()
    
    def get_genome(self):
        """
        Returns all weights and biases of the layer's nodes.
        """
        biases = np.array([node.bias for node in self.nodes])
        weights = np.array([node.weights for node in self.nodes]).flatten()
        genome = np.concatenate((biases, weights))
        return genome
    
    def set_genome(self, genome):
        """
        Sets all weights and biases of the layer's nodes.
        """
        node_biases = genome[:self.size]
        node_weights = genome[self.size:].reshape(self.size, self.input_dim)
        for i, node in enumerate(self.nodes):
            node.bias = node_biases[i]
            node.weights = node_weights[i]
            
    def fire(self, input_values):
        fire_node = np.vectorize(lambda node: node.fire(input_values))
        output_values = fire_node(self.nodes)
        return output_values
        

class NeuralNetwork:
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        
    def random_init(self):
        for layer in self.layers:
            layer.random_init()
            
    def forward_pass(self, input_values):
        """
        Forward pass of the neural network based on an array of input values.
        """
        for layer in self.layers:
            output_values = layer.fire(input_values)
            input_values = output_values
        selection = np.argmax(output_values)
        return output_values, selection
    
    def compute_genome_len(self):
        """
        Returns the network genome size based on the layer dimension parameters,
        without having to intialize the network first.
        """
        prev_size = self.input_dim
        genome_len = 0
        for layer in self.layers:
            genome_len += (layer.size + (prev_size * layer.size))
            prev_size = layer.size
        return genome_len
    
    def get_genome(self):
        """
        returns a 1d array of all weights and biases in the network. 
        The genome array is structured as follows:
        [l1_biases, l1_weights, ..., ln_biases, ln_weights]
        """
        genome = np.array([])
        for layer in self.layers:
            genome = np.concatenate((genome, layer.get_genome()))
        return genome
    
    def set_genome(self, genome):
        """
        sets all weight and biases in the network. The genes array should be 
        structured as follows:
        [l1_biases, l1_weights, ..., ln_biases, ln_weights]
        """
        for layer in self.layers:
            gene_len = layer.size + layer.input_dim * layer.size
            layer.set_genome(genome[:gene_len])
            genome = genome[gene_len:]
        
    def network_to_coordinates(self):
        """
        Converts the network to sets of node coordinates and node connections
        for easy plotting.
        """
        # Node indices per layer (includes an input layer)
        node_indices = [list(range(self.input_dim))]
        node_indices += [list(range(layer.size)) for layer in self.layers]
       
        # Node coordinates per layer
        node_coords = []
        for x, layer in enumerate(node_indices):
            start = len(layer) / 2
            coords = np.array([(x, start-y) for y in layer])
            node_coords.append(coords)

        # Layer connection index pairs per layer with respect to previous layer
        # Syntanx: (cur_layer, prev layer). Input layer is skipped.
        connect_indices = []
        for i in range(len(self.layers)):
            layer_cur = node_indices[i+1]
            layer_prev = node_indices[i]
            connections = [(n_cur, n_prev) for n_cur in layer_cur 
                           for n_prev in layer_prev]
            connect_indices.append(connections)
        return node_indices, node_coords, connect_indices
                
    def visualize_structure(self, ax, node_size=800, font_size=12):
        """
        Visualizes the network structure with all weights and biases. First
        plots all nodes and biases, then fetches node coordinates from node 
        connection pairs and plots them based on weight positivety and magnitude.
        """
        node_indices, node_coords, connect_indices = self.network_to_coordinates()
        for layer, nodes in enumerate(node_coords):
            for node, coords in enumerate(nodes):
                
                # First layer node
                if layer == 0:  # 
                    ax.scatter(*coords, s=node_size, zorder=20, color='grey')
                
                # Other layer nodes, opacity = bias, white background node
                else:
                    bias = round(self.layers[layer-1].nodes[node].bias, 3)
                    color = 'green' if bias >= 0 else 'red'
                    ax.scatter(*coords, s=node_size, zorder=20, color='white')
                    ax.scatter(*coords, s=node_size, zorder=20, color=color, alpha=abs(bias))
                    ax.text(*coords-[0.05,0.5], bias, fontsize=font_size)
                    
        # Connections
        for layer, connections in enumerate(connect_indices):
            for n_cur, n_prev in connections:
                weight = self.layers[layer].nodes[n_cur].weights[n_prev]
                coords = (node_coords[layer+1][n_cur], node_coords[layer][n_prev])
                connection = np.vstack(coords).T
                color = 'green' if weight >= 0 else 'red'
                plt.plot(*connection, color=color, zorder=10, alpha=abs(weight))
        ax.set_xticks([])
        ax.set_yticks([])
    
    
    def visualize_forwardpass(self):
        pass
        

    
if __name__ == '__main__':
    
    # Network init
    n = NeuralNetwork(input_dim=10)
    n.add(Layer(size=4, input_dim=10, activation_func=Activation.sigmoid))
    n.add(Layer(size=4, input_dim=4, activation_func=Activation.sigmoid))
    n.random_init()
    
    fig, ax = plt.subplots(figsize=(12,8))
    n.visualize_structure(ax)
    plt.show()
    
    # Prediction
    values = np.random.uniform(-1, 1, 10)
    output, selection = n.forward_pass(values)
    print(f'\nOutpunt and selection:\n{output, selection}')
    
    # Testing get/set genes
    genome = n.get_genome()
    print(f'\nGenome:\n{genome}')
    
    # Genome length
    genome_len = n.compute_genome_len()
    print(f'\nGenome len:\n{genome_len}')
    

    







    
    
    