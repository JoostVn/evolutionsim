from NeuralNetwork import NeuralNetwork, Layer, NetworkPlotter
import numpy as np
from math import pi
import Tools
import pygame
from Color import Color, ColorGradient
import Sensors
        
        
class GeneralPilot:

    """
    Autopilot that can radially detect close objects, lineary detects 
    obstacles and detects the bot speed.
    """

    
    def __init__(self, bot, actions, radius, hidden_layer_dim):

        self.bot = bot
        self.actions = actions
        self.sensor_radius = radius
        sensor_cols = (Color.LCYAN, Color.GREY7)
        self.sensors = [
            Sensors.Speed(self.bot),
            Sensors.RadialArea(self.bot, (-1/12*pi, 1/12*pi), radius, sensor_cols),
            Sensors.RadialArea(self.bot, (1/12*pi, 6/12*pi), radius, sensor_cols),
            Sensors.RadialArea(self.bot, (6/12*pi, 12/12*pi), radius, sensor_cols),
            Sensors.RadialArea(self.bot, (12/12*pi, 18/12*pi), radius, sensor_cols),
            Sensors.RadialArea(self.bot, (18/12*pi, 23/12*pi), radius, sensor_cols),
            ]
        self.network, self.plotter = self.creat_nn(hidden_layer_dim)
    
    def creat_nn(self, hidden_layer_dim):
        """
        Creates a simple neural network with the sensors as input layer and
        actions as output layer. The hidden_layer_dim parameter determines the 
        number of nodes in the hidden layer.
        """ 
        nr_sensors = len(self.sensors)
        nr_actions = len(self.actions)
        network = NeuralNetwork(input_dim=nr_sensors)
        network.add(Layer(hidden_layer_dim, nr_sensors))
        network.add(Layer(nr_actions, hidden_layer_dim))
        plotter = NetworkPlotter(network)
        return network, plotter
    
    def steer(self, food_positions, food_distances):
        """
        Updates all sensor and interprets their values to trigger an action
        in self.bot.
        """
        sensor_values = np.array([sensor.read(
            object_positions = food_positions,
            object_distances = food_distances)
            for sensor in self.sensors])
        
        # Run values trough neural network and execute actions
        output_values, selection = self.network.forward_pass(sensor_values)
        for i, val in enumerate(output_values):
            if val > 0.5:
                self.actions[i]()
                

    def draw(self, screen, pan_offset, zoom):
        for sensor in self.sensors:
            sensor.draw(screen, pan_offset, zoom)
        
        
    
        
