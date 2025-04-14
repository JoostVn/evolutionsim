import numpy as np
from neural_network import NeuralNetwork, Layer
import pygame
from pygame import gfxdraw
from math import pi, sin, cos
import shapely.vectorized
from shapely.affinity import rotate, translate
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString, box
from pygame_plot import Color, ColorGradient
import time
import tools

class AutoPilot:

    def __init__(self, input_dim, hidden_layer_dim, output_dim):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.network = self.creat_nn(hidden_layer_dim)
        self.outputs = np.arange(self.output_dim)
    
    def creat_nn(self, hidden_layer_dim):
        """
        Creates a simple neural network with the sensors as input layer and
        actions as output layer. The hidden_layer_dim parameter determines the 
        number of nodes in the hidden layer.
        """ 
        network = NeuralNetwork(input_dim=self.input_dim)
        network.add(Layer(hidden_layer_dim, self.input_dim))
        network.add(Layer(self.output_dim, hidden_layer_dim))
        return network
    
    def action_direct(self, sensor_values):
        """
        Returns the direct network outputs.
        """
        output_values, selection = self.network.forward_pass(sensor_values)
        return output_values
    
    def action_single(self, sensor_values):
        """
        Runs given sensor values trough the network to get a single action 
        as output.
        """
        output_values, selection = self.network.forward_pass(sensor_values)
        return selection
                
    def action_cutoff(self, sensor_values, cutoff):
        """
        Runs given sensor values trough the network to get all actions for
        which the network output is greater than some cutoff.
        """
        output_values, selection = self.network.forward_pass(sensor_values)
        selections = np.where(output_values > cutoff)[0]
        return selections
    
    def debug_forward_pass(self, sensor_values):
        """
        Performs a network forward pass and returns the values for each node
        and edge for debug porpouses.
        """
        _, _, node_values, edge_values = self.network.forward_pass_debug(
            sensor_values)
        edge_values = np.concatenate([e.flatten() for e in edge_values])
        node_values = np.concatenate(node_values)
        return  node_values, edge_values


class NormalValueSensor:
    """
    Simple sensor that holds any number of max_values that are used to 
    standardize input_values for each method call to .read()
    """
    def __init__(self, max_values):
        self.nr_sensors = len(max_values)
        self.max_values = np.array(max_values)
        self.sensor_values = np.zeros(self.nr_sensors)

    def rotate(self, bot_pos, delta_angle):
        pass    
    
    def translate(self, delta_pos):
        pass    
    
    def read(self, **kwargs):
        """
        Returns a (0, max_values) standardized array based on input_values.
        """
        input_values = kwargs.get('input_values')
        self.sensor_values = np.array(input_values) / self.max_values
        return self.sensor_values

    def draw(self, screen, pan_offset, zoom):
        pass    



class RadialAreaSensor:
    
    """
    Sensor that scans multiple circular areas around a bot for items and 
    returns a value for each area based on the closest item.
    """
    
    def __init__(self, bot_pos, bot_angle, sensor_range, sensor_angles):
        self.nr_sensors = len(sensor_angles)
        self.sensor_range = sensor_range
        self.sensor_angles = np.array(sensor_angles)
        self.sensor_values = np.zeros(self.nr_sensors)
        self.color = np.array(Color.GREY5)
        self.gradient = ColorGradient(Color.GREY7, Color.CYAN3)
        self.polygon = self._create_polygon(bot_pos, bot_angle)

    def _create_polygon(self, bot_pos, bot_angle, resolution=None):
        """
        Creates a shapely MultiPolygon containing each sensor polygon.
        """
        resolution = pi/24
        
        sensor_polygons = []
        for (a1, a2) in self.sensor_angles:
            
            # Create extra sensor angles for nicer shape based on resolution
            if resolution is not None:
                outer_angles = np.append(np.arange(a1, a2, resolution), a2)
            else:
                outer_angles = np.append((a1, a2), a2)
            
            # Create sensor polygon points
            points = [(0,0)]
            for angle in outer_angles:
                point = np.multiply(self.sensor_range, (cos(angle), sin(angle)))
                points.append(point)
            sensor_polygons.append(Polygon(LineString(points)))
        polygon = MultiPolygon(sensor_polygons)
        polygon = translate(polygon, *bot_pos)
        polygon = rotate(
            polygon, bot_angle, origin=bot_pos, use_radians=True)
        return polygon
    
    def rotate(self, bot_pos, delta_angle):
        """
        Rotate the sensor polyons relative to bot_pos.
        """
        self.polygon = rotate(
            self.polygon, delta_angle, origin=tuple(bot_pos), use_radians=True)
        
    def translate(self, delta_pos):
        """
        Translate the sensor polyons.
        """
        self.polygon = translate(self.polygon, *delta_pos)
    
    def read(self, **kwargs):
        """
        Reads the standardized sensor value based on the closest food item in
        range. Value is between 1 (max closeness) and 0 (no food in range).
        """
        food_pos = kwargs.get('food_pos')
        food_distances = kwargs.get('food_distances')
        
        # Initialize closest food array and loop over food / sensor polygons
        closest = np.full(len(self.sensor_values), self.sensor_range)
        for food_i, food in enumerate(food_pos):
            for pol_i, pol in enumerate(self.polygon):
                
                # If sensor pol contains food, get dist and skip to next food
                if pol.contains(Point(food)):
                    closest[pol_i] = min(closest[pol_i], food_distances[food_i])
                    break
                
        # Invert and normalize sensor values array
        self.sensor_values = (self.sensor_range - closest) / self.sensor_range
        return self.sensor_values
    
        """
        food_pos = kwargs.get('food_pos')
        food_distances = kwargs.get('food_distances')
        self.sensor_values = np.zeros(self.nr_sensors)
        for i, pol in enumerate(self.polygon):
            in_area = shapely.vectorized.contains(pol, *food_pos.T)
            if in_area.any():
                min_dist = food_distances[in_area].min()
                self.sensor_values[i] =  1 - min_dist / self.sensor_range
        return self.sensor_values
        """

    def draw(self, screen, pan_offset, zoom):
        """
        Draw each sensor polygon to the screen with value dependend coloring.
        """
        for pol, val in zip(self.polygon, self.sensor_values):
            pol_points = np.array(pol.exterior.coords.xy).T
            pol_points_draw = (pol_points * zoom + pan_offset).astype(int)
            col = self.gradient.get_color(val)
            gfxdraw.aapolygon(screen, pol_points_draw, col)
            pygame.draw.polygon(screen, col, pol_points_draw)



class LinearSensor:
    
    """
    Sensor that scans lines originating from the bot for objects and
    returns a value based on the closest object.
    """
    def __init__(self, bot_pos, bot_angle, sensor_range, sensor_angles):
        self.nr_sensors = len(sensor_angles)
        self.sensor_range = sensor_range
        self.sensor_angles = np.array(sensor_angles)
        self.sensor_values = np.zeros(self.nr_sensors)
        self.color = Color.RED3
        self.gradient = ColorGradient(Color.GREY7, self.color)
        self.line = self._create_line(bot_pos, bot_angle)

    def _create_line(self, bot_pos, bot_angle):
        """
        Creates a shapely MultiLineString containing each sensor line.
        """
        sensor_lines = []
        for angle in self.sensor_angles:
            end = np.multiply(self.sensor_range, (cos(angle), sin(angle)))
            line = LineString(((0,0), end))
            sensor_lines.append(line)
        line = MultiLineString(sensor_lines)
        line = translate(line, *bot_pos)
        line = rotate(line, bot_angle, origin=bot_pos, use_radians=True)
        return line
    
    def rotate(self, bot_pos, delta_angle):
        """
        Rotate the sensor lines relative to bot_pos.
        """
        self.line = rotate(
            self.line, delta_angle, origin=tuple(bot_pos), use_radians=True)
        
    def translate(self, delta_pos):
        """
        Translate the sensor lines.
        """
        self.line = translate(self.line, *delta_pos) 
        
    def read(self, **kwargs):
        """
        Reads the standardized sensor values based on crossings between the
        sensor lines and barriers. When one or more barriers are detected,
        returns a value based on the closest barrier between 1 (closest) and
        0 (no barrier).
        """
        bot_pos = kwargs.get('bot_pos')
        barrier_polygons = kwargs.get('barrier_polygons')
        
        # Only consider inrange barrier polygons
        bot_point = Point(bot_pos)
        inrange_barriers = [
            pol for pol in barrier_polygons if 
            bot_point.distance(pol) <= self.sensor_range]
        
        # Loop over barriers and polygons to determine sensor values
        closest = np.full(len(self.sensor_values), self.sensor_range)
        for line_i, line in enumerate(self.line):
            for pol_i, pol in enumerate(inrange_barriers):
                
                # If sensor line crosses barrier pol, determine intersect dist
                if line.crosses(pol):
                    intersection = line.intersection(pol).boundary[0]
                    dist = bot_point.distance(intersection)
                    closest[line_i] = min(closest[line_i], dist)
            
        # Invert and normalize sensor values array
        self.sensor_values = (self.sensor_range - closest) / self.sensor_range
        return self.sensor_values
        
        """
        bot_pos = kwargs.get('bot_pos')
        barrier_polygons = kwargs.get('barrier_polygons')
        self.sensor_values = np.zeros(self.nr_sensors)
        for i, line in enumerate(self.line):
            for polygon in barrier_polygons:
                if line.crosses(polygon):
                    intersection = line.intersection(polygon).boundary[0]
                    dist = Point(bot_pos).distance(intersection)
                    std_dist = 1 - (dist / self.sensor_range)
                    self.sensor_values[i] = max(self.sensor_values[i], std_dist)
        return self.sensor_values
        """
        
    def draw(self, screen, pan_offset, zoom):
        """
        Draw each sensor line to the screen with value dependend coloring.
        """
        for i, val in enumerate(self.sensor_values):
            if val == 0:
                line_points = np.array(self.line[i].xy).T
                line_points_draw = (line_points * zoom + pan_offset).astype(int)
                col = self.gradient.get_color(0.1)
                pygame.draw.aaline(screen, col, *line_points_draw)
            else:
                bot_pos = np.array(self.line[i].xy).T[0]
                cross_point = np.array(self.line[i].interpolate(1-val, normalized=True).xy).T
                line_points = np.vstack([bot_pos, cross_point])
                line_points_draw = (line_points * zoom + pan_offset).astype(int)
                col = self.gradient.get_color(val)
                pygame.draw.aaline(screen, col, *line_points_draw)
                gfxdraw.aacircle(screen, *line_points_draw[-1], 2, self.color)
                pygame.draw.circle(screen, self.color, line_points_draw[-1], 2)
                
class HomingSensor:
    """
    Sensor that allow a bot to lock onto an object. The sensor selects the 
    closest object in area in front of a bot (defined by an angle) and returns
    a value that is based on the angle to that object relative to the current
    bot angle. A value of 1 is given at perfect allignment, and a value of 0 
    is given at the worst possible allignment (when it equals the sensor angle)
    OR when no object is close enough for selection.
    """

