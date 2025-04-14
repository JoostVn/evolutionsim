from Color import Color, ColorGradient
import pygame
import numpy as np
from math import atan2, cos, sin, asin, acos



"""
TODO:
    - visual arrow key presses in window
    

"""





class Autopilot:
    
    def __init__(self, player, walls):
        self.player = player
        self.walls = walls
    
    def steer(self):
        sensor_values = self.read_sensors()
        course_actions = self.compute_course(sensor_values)
        self.execute_course(course_actions)
    
    def read_sensors(self):
        
        # Speed and direction
        speed = self.player.speed
        direction = self.player.direction
        
        # Distance to all walls
        wall_front = None
        wall_back = None
        wall_left = None
        wall_right = None
        
        # Close elements
        area_1 = None
        area_2 = None
        area_3 = None
        area_4 = None
        area_5 = None
        area_6 = None

        # Standardization
        sensor_values = None
        return sensor_values
    
    def compute_course(self, sensor_values):
        course_actions = np.random.randint(0,8)
        return course_actions
    
    def execute_course(self, action):
        p = self.player
        directions = {
            0:[p.arrow_up, p.arrow_left],
            1:[p.arrow_up, p.arrow_right],
            2:[p.arrow_up],
            3:[p.arrow_down, p.arrow_left],
            4:[p.arrow_down, p.arrow_right],
            5:[p.arrow_down],
            6:[p.arrow_left],
            7:[p.arrow_right]
            }
        for command in directions[action]:
            command()
            

class PlayerArrow:
    
    def __init__(self, position, size, color, turnspeed=0.15, acceleration=0.4, 
                 dampening=0.97, minspeed=0, maxspeed=15):
                 
        # Constants
        self.size = size
        self.color = color
        self.turnspeed = turnspeed
        self.acceleration = acceleration
        self.dampening = dampening
        self.minspeed = minspeed
        self.maxspeed = maxspeed
    
        # Variables
        self.pos = np.array(position)        
        self.speed = 0                       
        self.direction = np.array((1,0))
        self.pos_draw = np.array(position)   
        
        # Trail
        self.trail = np.full((40,2), self.pos)
        self.trail_gradient = ColorGradient(Color.LGREY, Color.MGREY, 40).color_vector
        
        # Autosteering
        self.auto_steering = False
        
    def add_autopilot(self, autopilot):
        self.autopilot = autopilot
        self.auto_steering = True

    def update(self):
        self.pos = self.pos + (self.speed * self.direction)
        self.speed = self.dampening * self.speed              
        self.trail = np.vstack([self.trail[1:,:], self.pos])
        if self.auto_steering:
            self.autopilot.steer()
        
    def draw(self, screen, pan_offset):
        self.draw_trail(screen, pan_offset)
        self.draw_body(screen, pan_offset)
        
    def draw_trail(self, screen, pan_offset):
        
        # Scramble trail
        amt = 1
        self.trail = self.trail + np.random.uniform(-amt, amt, self.trail.shape)
        
        # Draw trail
        trail_draw = (self.trail + pan_offset).astype(int)
        segments = zip(trail_draw[:-1], trail_draw[1:], self.trail_gradient)
        for start, end, col in segments:
            pygame.draw.line(screen, col, start, end, 3) 
            
    def draw_body(self, screen, pan_offset):
        pos_draw = (self.pos + pan_offset).astype(int)
        vertical = self.direction
        horizontal = np.flip(self.direction) * [-1, 1]
        p1 = (pos_draw + self.size * 1.0 * vertical)
        p2 = (pos_draw + self.size * 0.4 * horizontal)
        p3 = (pos_draw + self.size * 0.2 * vertical)
        p4 = (pos_draw - self.size * 0.4 * horizontal)
        point_list = np.array((p1,p2,p3,p4)).astype(int)
        
        pygame.draw.polygon(screen, self.color, point_list)
        pygame.draw.aalines(screen, self.color, True, point_list)

    def arrow_left(self):
        x, y = self.direction[0], self.direction[1]
        angle = atan2(y, x) - self.turnspeed
        self.direction = np.array((cos(angle), sin(angle)))
            
    def arrow_right(self):
        x, y = self.direction[0], self.direction[1]
        angle = atan2(y, x) + self.turnspeed
        self.direction = np.array((cos(angle), sin(angle)))

    def arrow_up(self):
        self.speed = min(self.maxspeed, self.speed + self.acceleration)
    
    def arrow_down(self):
        self.speed = max(self.minspeed, self.speed - self.acceleration)



class Walls:
        
    def __init__(self, corners, size, color):
        self.corners = np.array(corners)
        self.size = size
        self.color = color
    
    def update(self):
        pass
        
    def draw(self, screen, pan_offset):
        points_draw = self.corners + pan_offset
        pygame.draw.lines(screen, self.color, True, points_draw, self.size)
    
        

class ProgramTemplate:
    
    def __init__(self):
        
        # Program elements
        self.arrow = PlayerArrow(size=15, color=Color.NAVY, position=(100, 100))
        self.walls = Walls([[0,0],[0,80],[80,80],[0,80]], 2, Color.MGREY)
        self.elements = [self.arrow, self.walls]
        
        # Autopilot
        #pilot = Autopilot(self.arrow, None)
        #self.arrow.add_autopilot(pilot)
            
    def update(self, i):
        """
        Update all program elements with one tick increment.
        """
        for element in self.elements:
            element.update()
    
    def draw(self, screen, pan_offset):
        """
        Draw all program elements on the screen.
        """
        for element in self.elements:
            element.draw(screen, pan_offset)
          
    def information(self):
        info_list = []
        return info_list
            
    def mouse_left_draw(self, start_pos, end_pos, pan_offset):
        """
        Function that can utilize a line drawn in the simulation, defined by
        a click, hold and release of the left mouse button. Can also be 
        interpreted as a single click
        """
        pos = np.array((start_pos, end_pos)) - pan_offset
        pass
    
    def arrow_left(self):
        self.arrow.arrow_left()
            
    def arrow_right(self):
        self.arrow.arrow_right()

    def arrow_up(self):
        self.arrow.arrow_up()
    
    def arrow_down(self):
        self.arrow.arrow_down()
    
    
    







