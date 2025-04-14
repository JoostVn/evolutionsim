import numpy
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.special import expit
import time

class Layer:
    
    def __init__(self, size, input_dim, activation_func=np.tanh):
        self.size = size
        self.input_dim = input_dim
        self.biases = np.zeros(size)
        self.weights = np.zeros((size, input_dim))
        self.activation_func = activation_func

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
        return self.activation_func(self.weights.dot(input_values) + self.biases)
    
    def fire_only_weights(self, input_values):
        """
        Debug mode: Only multiply input_values with edge weights.
        """
        return input_values * self.weights
        
    def fire_only_biases(self, input_edge_product):
        """
        Debug mode: Only add biases to input_edge_product.
        """
        return input_edge_product + self.biases
        
    def fire_only_activation(self, weight_bias_productsum):
        """
        Debug mode: Only apply activation function to weight_bias_productsum.
        """
        return self.activation_func(weight_bias_productsum)
    
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

    def forward_pass_debug(self, input_values):
        """
        Forward pass of the neural network based on an array of input values.
        Returns not only the network output, but the value of edges and 
        nodes for each intermediate step. Edge values are defined as 
        (edge weigth * prev node output) and node values are defined as
        the output of their activation function.
        """
        node_values = [input_values]
        edge_values = []
        for layer in self.layers:
            input_edge_product = layer.fire_only_weights(input_values)
            product_sum = input_edge_product.sum(axis=1)
            biases_sum = layer.fire_only_biases(product_sum)
            node_activations = layer.fire_only_activation(biases_sum)
            edge_values.append(input_edge_product)
            node_values.append(node_activations)
            input_values = node_activations
        output_values = node_activations
        selection = np.argmax(node_activations)
        return output_values, selection, node_values, edge_values

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
        self.node_coords, self.edge_coords = self.get_coordinates(scale)
        self.node_biases, self.edge_weights = self.get_weights_and_biases()
        self.color_red_green = np.vectorize(self._color_red_green)

    def _color_red_green(self, value):
        """
        Chooses a red or green color based on a given value.
        """
        return to_rgb('red' if value < 0 else 'green')
        
    def rgba_picker(self, values, alpha=True):
        """
        Returns the RGBA colors for an array of values. If alpha=False, only
        returns RGB colors.
        """
        rgb_cols = self.color_red_green(values)
        if alpha:
            return np.vstack((rgb_cols, abs(values))).T
        else:
            return rgb_cols
       
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
     
    def get_weights_and_biases(self):
        """
        Fetches flat vectors of weights and biases from the net.
        """
        weights = np.concatenate([l.weights.flatten() for l in self.n.layers])
        biases = np.concatenate([l.biases for l in self.n.layers])
        return biases, weights

    def get_text(self, coordinates, values, offset):
        """
        Returns text coordinates and strings given a list of node coordinates,
        node values and a text offset.
        """
        text_pos = (coordinates.T - np.array(offset)).T
        text_str = values.round(2).astype(str)
        return text_pos, text_str

    def pyplot_structure(self, ax, node_size=800, font_size=12, 
                         font_offset=[0.08,0.35]):
        """
        Matplotlib plot of network structure.
        """
        # Input + hidden + output layer nodes
        input_layer = self.node_coords.T[:self.n.input_dim].T
        ax.scatter(*input_layer, s=node_size, c='grey', zorder=20)
        layers = self.node_coords.T[self.n.input_dim:].T
        cols = self.rgba_picker(self.node_biases)
        ax.scatter(*layers, s=node_size, c='white', zorder=10)
        ax.scatter(*layers, s=node_size, c=cols, zorder=20)
        
        # Layer edges
        cols = self.rgba_picker(self.edge_weights)
        for connection, col in zip(self.edge_coords, cols):
            ax.plot(*connection, color=col, zorder=5)
        
        # Node bias text
        text_pos, text_str = self.get_text(layers, self.node_biases, font_offset)
        annotate = np.vectorize(ax.text)
        annotate(*text_pos, text_str, fontsize=font_size)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
    def pyplot_forward_pass(self, ax, node_values, edge_values, node_size=800, 
                            font_size=12, font_offset=[0.08,0.35]):
        """
        Matplotlib plot of edge values and node activations from a forward pass.
        """
        # Flattening value arrays
        edgevals = np.concatenate([e.flatten() for e in edge_values])
        nodevals = np.concatenate(node_values)
        
        # Node layers
        cols = self.rgba_picker(nodevals)
        ax.scatter(*self.node_coords, s=node_size, c='white', zorder=10)
        ax.scatter(*self.node_coords, s=node_size, c=cols, zorder=20)
        
        # Layer edges
        cols = self.rgba_picker(edgevals)
        for connection, col in zip(self.edge_coords, cols):
            ax.plot(*connection, color=col, zorder=5)
        
        # Node activation text
        text_pos, text_str = self.get_text(self.node_coords, nodevals, font_offset)
        annotate = np.vectorize(ax.text)
        annotate(*text_pos, text_str, fontsize=font_size)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
    
        
if __name__ == '__main__':
    
    # Network init
    n = NeuralNetwork(input_dim=10)
    n.add(Layer(size=8, input_dim=10))
    
    n.add(Layer(size=4, input_dim=8))
    n.random_init()
    
    # Forward pass
    input_values = np.random.uniform(-1,1,n.input_dim)
    output_values, selection, node_values, edge_values = n.forward_pass_debug(input_values)
    print(f'{input_values}\n{output_values}')
    
    
    # Plotting structure
    plotter = NetworkPlotter(n)
    fig, ax = plt.subplots(figsize=(8,8))
    plotter.pyplot_structure(
        ax, node_size=300, font_size=10, font_offset=[0.04, 0.48])
    plt.show()
    
    # Plotting forward pass
    plotter = NetworkPlotter(n)
    fig, ax = plt.subplots(figsize=(8,8))
    plotter.pyplot_forward_pass(
        ax, node_values, edge_values, node_size=300, font_size=10, 
        font_offset=[0.04, 0.48])
    plt.show()
    

        
