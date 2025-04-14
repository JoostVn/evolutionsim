import numpy
from math import e
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

class Layer:
    
    def __init__(self, size, input_dim):
        self.size = size
        self.input_dim = input_dim
        self.biases = np.zeros(size)
        self.weights = np.zeros((size, input_dim))
        self.sigmoid = np.vectorize(lambda x: 1/(1+e**(-x)))

    def __repr__(self):
        return f'\nBiases\n{self.biases}\n\nWeights\n{self.weights}\n'

    def __len__(self):
        return len(self.get_genome())

    def random_init(self):
        """
        Set all weights and biases to random uniform values.
        """
        domain = (-1, 1)
        self.biases = np.random.uniform(*domain, self.biases.shape)
        self.weights = np.random.uniform(*domain, self.weights.shape)
        
    def fire(self, input_values):
        """
        Pass input_values trough layer and return node activations.
        """
        Y = self.weights.dot(input_values) + self.biases
        activation = self.sigmoid(Y)
        return activation
    
    def get_genome(self):
        """
        Return genome as a flat array of (biases, weights).
        """
        return np.concatenate((self.biases, self.weights.flatten()))
        
    def set_genome(self, genome):
        """
        Set genome from a flat array of (biases, weights).
        """
        self.biases = genome[:self.size]
        self.weights = genome[self.size:].reshape(self.weights.shape)

    

class NeuralNetwork:
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.genome_size = input_dim
        self.layers = []

    def __len__(self):
        """
        Return the total lenght of the neural network genome.
        """
        return len(self.get_genome())

    def add(self, layer):
        """
        Add one layer to the end of the network. Also updates weights/biases.
        """
        self.layers.append(layer)
        
    def random_init(self):
        """
        Set all layer weights and biases to random uniform values.
        """
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

    def get_genome(self):
        """
        Retreives all weight and biases in the network. Genome stucture:
        (l1_biases, l1_weights, ..., ln_biases, ln_weights)
        """
        return np.concatenate([layer.get_genome() for layer in self.layers])

    def set_genome(self, genome):
        """
        sets all weight and biases in the network. Genome structure:
        (l1_biases, l1_weights, ..., ln_biases, ln_weights)
        """
        for layer in self.layers:
            layer.set_genome(genome[:len(layer)])
            genome = genome[len(layer):]



class NetworkPlotter:
    
    def __init__(self, network, scale=1):
        self.n = network
        self.nodes, self.edges = self.get_coordinates(scale)
        self.rgb = np.vectorize(self.color_picker)

    def color_picker(self, value):
        """
        Chooses a red or green color based on a given value.
        """
        color = 'red' if value < 0 else 'green'
        return to_rgb(color)

    def get_coordinates(self, scale):
        """
        Computes network node and connection coordinates for plotting.
        """
        # Dimensions of each layer
        layer_dims = [self.n.input_dim] + [l.size for l in self.n.layers]
        
        # Layer node coordinates including input layer
        layer_nodes = []
        for i, size in enumerate(layer_dims):
            x = np.full(size, i*scale)
            y = (np.arange(0, size, 1) - (size-1)/2) * scale
            layer_nodes.append(np.vstack((x,y)))
        
        # Node pair connections
        layer_edges = []
        for cur, prev in zip(layer_nodes[1:], layer_nodes[:-1]):
            layer_edges += [
                np.vstack((i, j)).T for i in cur.T for j in prev.T]
            
        # Stack and return all nodes and connections
        nodes = np.hstack(layer_nodes)
        edges = np.array(layer_edges)
        return nodes, edges
    
    def get_structure_elements(self):
        """
        Fetches flat vectors of weights and biases from the net.
        """
        weights = np.concatenate([l.weights.flatten() for l in self.n.layers])
        biases = np.concatenate([l.biases for l in self.n.layers])
        return weights, biases

    def get_forwardpass_elements(self):
        """
        Fetches flat vectors of edge and node values from a net forward pass.
        """
    
    
    def pytplot_structure(self, ax, node_size=800, font_size=12):
        """
        Matplotlib plot of network structure.
        """
        w, b = self.get_structure_elements()

        # Input layer
        input_layer = self.nodes.T[:self.n.input_dim].T
        ax.scatter(*input_layer, s=node_size, c='grey', zorder=20)
        
        # Hidden and output layers
        layers = self.nodes.T[self.n.input_dim:].T
        rgba_cols = np.vstack((self.rgb(b), abs(b))).T
        ax.scatter(*layers, s=node_size, c=rgba_cols, zorder=20)
        ax.scatter(*layers, s=node_size, c='white', zorder=10)
        
        # Node bias text
        offset = [0.08,0.35]
        annotate = np.vectorize(ax.text)
        annotate(*(layers.T - offset).T, b.round(2).astype(str), fontsize=font_size)
      
        # Layer edges
        rgba_cols = np.vstack((self.rgb(w), abs(w)/2)).T
        for connection, col in zip(self.edges, rgba_cols):
            ax.plot(*connection, color=col, zorder=5)

        ax.set_xticks([])
        ax.set_yticks([])
        
    def pytplot_forward_pass(self, ax, node_size=800, font_size=12):
        pass
    
    def pygame_forward_pass(self, screen, node_size):
        pass
    
        
if __name__ == '__main__':
    
    # Network init
    n = NeuralNetwork(input_dim=6)
    n.add(Layer(size=6, input_dim=6))
    n.add(Layer(size=5, input_dim=6))
    n.add(Layer(size=4, input_dim=5))
    n.random_init()
    
    # Forward pass
    input_values = np.random.uniform(-1,1,6)
    output_values, selection = n.forward_pass(input_values)
    print(f'{input_values}\n{output_values}\n{selection}')
    
    # Plotting
    plotter = NetworkPlotter(n)
    fig, ax = plt.subplots(figsize=(8,6))
    plotter.pytplot_structure(ax, node_size=400, font_size=10)
    plt.show()
    

    
        



        
