import numpy as np
from algorithms.neural.nn import NeuralNetwork, Layer
import pygame
from pygame import gfxdraw
from math import pi, sin, cos
from algorithms.geometry.shapes import Line, Polygon
from pygametools.color.color import Color, ColorGradient


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
        network.add_layer(Layer(hidden_layer_dim, self.input_dim))
        network.add_layer(Layer(self.output_dim, hidden_layer_dim))
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

    def read(self, input_values):
        """
        Returns a (0, max_values) standardized array based on input_values.
        """
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
        self.gradient = ColorGradient(Color.GREY1, Color.GREEN3)
        self.polygons = self._create_polygon(bot_pos, bot_angle)

    def _create_polygon(self, bot_pos, bot_angle, resolution=None):
        """
        Creates a shapely MultiPolygon containing each sensor polygon.
        """
        resolution = pi/24
        polygons = []
        for (a1, a2) in self.sensor_angles:

            # Create extra sensor angles for nicer shape based on resolution
            outer_angles = np.append(np.arange(a1, a2, resolution), a2)

            # Create sensor polygon points
            points = [(0,0)]
            for angle in outer_angles:
                point = np.multiply(self.sensor_range, (cos(angle), sin(angle)))
                points.append(point)

            pol = Polygon(points)
            pol.translate(bot_pos)
            pol.rotate(bot_angle, bot_pos)
            polygons.append(pol)

        return polygons

    def rotate(self, delta_angle, bot_pos):
        """
        Rotate the sensor polyons relative to bot_pos.
        """
        for pol in self.polygons:
            pol.rotate(delta_angle, bot_pos)

    def translate(self, delta_pos):
        """
        Translate the sensor polyons.
        """
        for pol in self.polygons:
            pol.translate(delta_pos)

    def read(self, object_pos, object_distances):
        """
        Reads the standardized sensor value based on the closest food item in
        range. Value is between 1 (max closeness) and 0 (no food in range).
        """

        # Initialize closest object array and loop over objects / sensor polygons
        closest = np.full(len(self.sensor_values), self.sensor_range)
        for obj_i, obj in enumerate(object_pos):
            for pol_i, pol in enumerate(self.polygons):

                # If sensor pol contains object, get dist and skip to next object
                if pol.contains_point(obj):
                    closest[pol_i] = min(closest[pol_i], object_distances[obj_i])
                    break

        # Invert and normalize sensor values array
        self.sensor_values = (self.sensor_range - closest) / self.sensor_range
        return self.sensor_values

    def draw(self, screen, pan_offset, zoom):
        """
        Draw each sensor polygon to the screen with value dependend coloring.
        """
        for pol, val in zip(self.polygons, self.sensor_values):
            pol_points_draw = (pol.points * zoom + pan_offset).astype(int)
            col = self.gradient.get_color(val*0.2)
            pygame.draw.polygon(screen, col, pol_points_draw)
            gfxdraw.aapolygon(screen, pol_points_draw, col)



class LinearSensor:

    """
    Sensor that scans lines originating from the bot for objects and
    returns a value based on the closest object on each line, standardized to
    the line length.
    """

    def __init__(self, bot_pos, bot_angle, sensor_range, sensor_angles):
        self.nr_sensors = len(sensor_angles)
        self.sensor_range = sensor_range
        self.sensor_angles = np.array(sensor_angles)
        self.sensor_values = np.zeros(self.nr_sensors)
        self.color = Color.RED3
        self.gradient = ColorGradient(Color.GREY1, self.color)
        self.lines = self._create_line(bot_pos, bot_angle)

    def _create_line(self, bot_pos, bot_angle):
        """
        Creates a shapely MultiLineString containing each sensor line.
        """
        lines = []
        for angle in self.sensor_angles:
            vector = np.array((cos(angle), sin(angle)))
            endpoint = bot_pos + self.sensor_range * vector
            line = Line([bot_pos, endpoint])
            line.rotate(bot_angle, bot_pos)
            lines.append(line)

        return lines

    def rotate(self, delta_angle, bot_pos):
        """
        Rotate the sensor lines relative to bot_pos.
        """
        for line in self.lines:
            line.rotate(delta_angle, bot_pos)

    def translate(self, delta_pos):
        """
        Translate the sensor lines.
        """
        for line in self.lines:
            line.translate(delta_pos)

    def read(self, bot_pos, object_polygons):
        """
        Reads the standardized sensor values based on crossings between the
        sensor lines and barriers. When one or more barriers are detected,
        returns a value based on the closest barrier between 1 (closest) and
        0 (no barrier).
        """

        # Loop over object_polygon and sensor lines and initialize closest arr
        closest = np.full(len(self.sensor_values), self.sensor_range, float)
        for line_i, line in enumerate(self.lines):
            for pol_i, pol in enumerate(object_polygons):

                # cheaply test for intersections between line and bol
                if not line.intersect_pol_bool(pol):
                    continue

                # If sensor line crosses object pol, determine intersect dist
                intersections = line.intersect_pol(pol)
                dist = np.linalg.norm(intersections - bot_pos, axis=1).min()
                closest[line_i] = min(closest[line_i], dist)

        # Invert and normalize sensor values array
        self.sensor_values = (self.sensor_range - closest) / self.sensor_range
        return self.sensor_values

    def draw(self, screen, pan_offset, zoom):
        """
        Draw each sensor line to the screen with value dependend coloring.
        """
        dot_size = int(max(1, 4 * zoom))
        for i, val in enumerate(self.sensor_values):
            if val == 0:
                line_points = self.lines[i].points
                line_points_draw = (line_points * zoom + pan_offset).astype(int)
                col = self.gradient.get_color(0.1)
                pygame.draw.aaline(screen, col, *line_points_draw)
            else:
                bot_pos = self.lines[i].points[0]
                cross_point = bot_pos + self.lines[i].vec * (1-val)
                line_points = np.vstack([bot_pos, cross_point])
                line_points_draw = (line_points * zoom + pan_offset).astype(int)
                col = self.gradient.get_color(val)
                pygame.draw.aaline(screen, col, *line_points_draw)
                gfxdraw.aacircle(
                    screen, *line_points_draw[-1], dot_size, self.color)
                pygame.draw.circle(
                    screen, self.color, line_points_draw[-1], dot_size)



class HomingSensor:
    """
    Sensor that allow a bot to lock onto an object. The sensor selects the
    closest object in area in front of a bot (defined by an angle) and returns
    a value that is based on the angle to that object relative to the current
    bot angle. A value of 0 is given at perfect allignment, and a value of -1
    or 1 is given at the worst possible allignment (when it equals the sensor
    angles) OR when no object is close enough for selection.
    """

    #TODO