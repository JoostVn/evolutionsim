from math import pi, sin
import Tools
import pygame
from Color import ColorGradient, Color
import numpy as np


class Speed:
    
    def __init__(self, bot):
        """
        Very simple sensor that just returns the standardized bot speed.
        """
        self.bot = bot
        
    def read(self, **kwargs):
        return self.bot.speed / self.bot.max_speed
        
    def draw(self, screen, pan_offset, zoom):
        pass



class RadialArea:
    
    def __init__(self, bot, angles, radius, colors):
        """
        Sensor that returns the normalized distance of the closest object in an
        area defined by two angles and a radius from bot.pos. 
        bot (bot class instance): bot on which to attach sensor
        angles (tuple): The two angles relative to bot.angle that
        bound the sensor area. The angles are interpreted as positive 
        clockwise radians relative to bot.angle. Only the first angle can 
        be negative.
        """
        self.bot = bot
        self.angles = angles
        self.radius = radius
        self.areas = self.convert_negative_angle(angles)
        self.sensor_value = 1
        self.gradient = ColorGradient(*colors, 100).color_vector
        
        # Generate extra angle points for a better circle approximation draw
        extra_curves = int(abs(self.angles[0] - self.angles[1]) / (1/24 * pi))
        self.draw_angles = np.linspace(self.angles[0], self.angles[1], extra_curves)
        
    def convert_negative_angle(self, angles):
        """
        If the first angle is negative, split into two areas around zero, with 
        areas = (2pi - a1, 2pi) and (0, a2).
        """
        if self.angles[0] >= 0:
            return np.array((angles,))
        else:
            area_1 = (2 * pi + angles[0], 2 * pi)
            area_2 = (0, angles[1])
            return np.array((area_1, area_2))
            
    def read(self, **kwargs):
        """
        Checks for a np array of objects if their positions lie in the sensor
        area. The check works by computing the relative angle to each object
        and comparing it to the sensor area angles. Returns the distance of 
        the closest object, standardized by self.radius for a value 
        between 0 and 1. Objects should have a .pos attribute.
        """
        object_positions = kwargs['object_positions']
        object_distances = kwargs['object_distances']
        self.sensor_value = 1
                
        # Get all objects in sensor range
        inrange = object_distances < self.radius
        if inrange.any() == False:
            return 1
        
        # Filter positions and distances arrays
        object_positions = object_positions[inrange]
        object_distances = object_distances[inrange]
        
        # Compute direction vectors/angles/relativee angles to objects
        directions = object_positions - self.bot.pos
        angles = np.arctan2(*np.flip(directions, axis=1).T)
        relative_angles = angles - self.bot.angle
        relative_angles[relative_angles < 0] += 2*pi
        
        # Compute which objects are in sensor area based on relative angles
        ra = relative_angles
        minbound = self.areas.T[0]
        maxbound = self.areas.T[1]
        if len(self.areas) == 1:
            in_area = np.all((minbound <= ra, ra < maxbound), axis=0)
        elif len(self.areas) == 2:
            in_area1 = np.all((minbound[0] <= ra, ra < maxbound[0]), axis=0)
            in_area2 = np.all((minbound[1] <= ra, ra < maxbound[1]), axis=0)
            in_area = np.any((in_area1, in_area2), axis=0)
       
        #Return the standardized distance to the closest object in senor area
        if in_area.any() == False:
            return 1
        else:
            self.sensor_value = object_distances[in_area].min() / self.radius
            return self.sensor_value        
        
    def draw(self, screen, pan_offset, zoom):
        """
        Draws a polygon between the two sensor points and self.bot.pos. The
        polygon is colored according to the current self.sensor_value.
        """
        pos_draw = (self.bot.pos * zoom + pan_offset)
        rad_draw = zoom * self.radius
        polygon = [pos_draw]
        for angle in self.draw_angles:
            direction = Tools.convert_to_vector(angle + self.bot.angle)
            polygon.append(rad_draw * direction + pos_draw)
        color = self.gradient[int(round(99 * self.sensor_value))]
        pygame.draw.polygon(screen, color, polygon)
        
        

class LinearWallDetect:
    
    def __init__(self, bot, angle, max_distance):
        """
        Sensor that detects the distance to walls by drawing straight lines
        from the bot and returning their standardized distance.
        """
        self.bot = bot
        self.angle = angle
        self.max_distance = max_distance
        self.sensor_value = None
        
    def read(self, **kwargs):
        """
        Detects the distance to each wall and returns the min value. 
        the distance is calculated with c sin(B) = b sin(C).
        """
        walls = kwargs['walls']
        
        A = self.bot.angle + self.angle
        bot = self.bot.pos    
        wall_dis = []
        
        wall = walls[0]
        
        p1, p2 = wall.endpoints
        
        p1_vector = self.bot.pos - p1
        p2_vector = self.bot.pos - p2
        
        self.p1 = p1_vector
        self.p2 = p2_vector
           
        self.sensor_value = 500
        return self.sensor_value
        
    def draw(self, screen, pan_offset, zoom):
        """
        Draws a line between the bot and the detected surface
        """
        pos_draw = (self.bot.pos * zoom + pan_offset)
        direction = Tools.convert_to_vector(self.angle + self.bot.angle)        
        endpoint = pos_draw + (self.sensor_value * zoom * direction) 
        pygame.draw.line(screen, Color.RED, pos_draw, endpoint, 2)
        
    
    
        # Debug
        d1 = self.p1 * zoom
        d2 = self.p2 * zoom
        
        pygame.draw.line(screen, Color.BLUE, pos_draw, pos_draw-d1, 2)
        pygame.draw.line(screen, Color.GREEN, pos_draw, pos_draw-d2, 2)












