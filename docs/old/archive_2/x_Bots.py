import numpy as np
import pygame
from Color import Color
import Tools
import Autopilot
from pygame_plot.plot import Plot
from math import floor

class SimpleArrowBot:
    
    def __init__(self, position, direction, **kwargs):

        # Parameters
        self.size = 8
        self.color = Color.DBLUE
        self.turnspeed = 0.2
        self.speed = 1
        self.acceleration = 0.1
        self.dampening = 0.98
        self.min_speed = 0
        self.max_speed = 12
        self.eat_range = 4
        
        # Fitness and starvation
        self.food = 0
        self.starvation = 0
        self.starve_threshold = 150
        self.starve_func = lambda speed: 1 + 20 * (speed/self.max_speed) ** 2
        self.alive = True
        self.fitness = 0
        self.fitness_func = lambda alive, food, time: 5 * alive + food + 5 * time / 400 
        
        # Positioning
        self.pos = position    
        self.direction = direction
        self.angle = Tools.convert_to_angle(self.direction)
       
        # Autopilot
        pilot_actions = [
            self.arrow_left, self.arrow_right, self.arrow_up, self.arrow_down]
        self.autopilot = Autopilot.GeneralPilot(
            self, pilot_actions, radius=50, hidden_layer_dim=6)
        
        # Statistics
        self.starvation_log = []
        
    def update(self, t, food):
        if self.alive:
            
            # Computing eucledian distance to each food object
            food_positions = np.asarray([f.pos for f in food])
            food_distances = np.linalg.norm(food_positions - self.pos, axis=1)
            
            # Steering and eating
            self.angle = Tools.convert_to_angle(self.direction)
            self.speed = self.dampening * self.speed              
            self.pos = self.pos + (self.speed * self.direction)
            self.eat(food, food_distances)
            self.autopilot.steer(food_positions, food_distances)
            
            # Starvation check and fitness
            self.starvation += self.starve_func(self.speed)
            self.starvation_log.append(self.starvation)
            if self.starvation >= self.starve_threshold:
                self.alive = False
                self.color = Color.GREY5
                
            # Fitness
            self.fitness = self.fitness_func(self.alive, self.food, t)
        
    def eat(self, food, food_distances):
        in_eat_range = np.where(food_distances < self.eat_range)[0]
        
        
        for food_index in in_eat_range:
            food.pop(food_index)
            self.food += 1
            self.starvation = 0 

    def get_genome(self):
        return self.autopilot.network.get_genome()
    
    def set_genome(self, genome):
        self.autopilot.network.set_genome(genome)
            
    def random_init(self):
        self.autopilot.network.random_init()
    
    def draw(self, screen, pan_offset, zoom, sensors=False):
        
        # Draw bot polygon
        pos_draw = (self.pos * zoom + pan_offset).astype(int)
        size_draw = max(1, self.size * zoom)
        vertical = self.direction
        horizontal = np.flip(self.direction) * [-1, 1]
        p1 = (pos_draw + size_draw * 0.6 * vertical)
        p2 = (pos_draw + size_draw * 0.22 * horizontal)
        p3 = (pos_draw - size_draw * 0.2 * vertical)
        p4 = (pos_draw - size_draw * 0.22 * horizontal)
        point_list = np.array((p1,p2,p3,p4)).astype(int)
        pygame.draw.polygon(screen, self.color, point_list)
        pygame.draw.aalines(screen, self.color, True, point_list)
        
        # Draw autpilot sensors
        if sensors and self.alive:
            self.autopilot.draw(screen, pan_offset, zoom)
    
    
    def real_time_analysis(self, t):
        """
        Creates and updates a realtime plot for the bot starvation.
        """
        x = np.arange(t)
        y = self.starvation_log
        xdomain = (0, max(x)+1)
        ydomain = (min(y)-1, max(y)+1)
        
        if len(y) < 2:
            self.rt_plot = Plot(xdomain, ydomain, (40,25), (245,120))
            self.rt_plot.border = False
            self.rt_plot.title = 'Starvation of focus bot'
            self.rt_plot.yaxis.nr_ticks = 8
            self.rt_plot.xaxis.lock_position = True
            self.rt_plot.add_legend('upper left', width=50, border=False)
        
        elif len(y) == 2:
            self.rt_plot.add_line(x, y, Color.MGREEN, 1, 'starvation')
        
        elif len(y) > 2: 
            self.rt_plot.elements['starvation'].add_data([x[-1]], [y[-1]])
            
        if t < 6:
            self.rt_plot.xaxis.set_labels(x, x.astype(str))
        else:
           ticks = np.arange(0, t+1, floor(t / 6))
           self.rt_plot.xaxis.set_labels(ticks, ticks.astype(str))
            
        self.rt_plot.update_dimensions(xdomain, ydomain)
        return self.rt_plot
        
        
        
        
    
    def arrow_left(self):
        adjusted_turnspeed = (1 - self.speed / self.max_speed) * self.turnspeed
        self.direction = Tools.convert_to_vector(self.angle - adjusted_turnspeed)
            
    def arrow_right(self):
        adjusted_turnspeed = (1 - self.speed / self.max_speed) * self.turnspeed
        self.direction = Tools.convert_to_vector(self.angle + adjusted_turnspeed)

    def arrow_up(self):
        self.speed = min(self.max_speed, self.speed + self.acceleration)
    
    def arrow_down(self):
        self.speed = max(self.min_speed, self.speed - self.acceleration)



if __name__=='__main__':
    pos = np.array((400,400))
    direction = Tools.convert_to_vector(0)
    bot = SimpleArrowBot(pos, direction)
    
    
    