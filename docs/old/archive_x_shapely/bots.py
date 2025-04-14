from sensors import AutoPilot, NormalValueSensor, RadialAreaSensor, LinearSensor
import numpy as np
import pygame
from pygame import gfxdraw
from math import pi, sin, cos, ceil
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, LineString
from scipy.spatial.distance import cdist
from pygame_plot import Color
import time
import tools

class Herbivore:
    
    TURN_LEFT = -1
    TURN_RIGHT = 1
    SPEED_UP = 1
    SPEED_DOWN = -1
    
    def __init__(self, bot_pos, bot_angle):
        
        # Body and positioning
        self.size = 6
        self.color = Color.BLUE1
        self.pos = np.array(bot_pos)
        self.angle = bot_angle
        self.polygon = self._create_polygon()
        
        # Movement
        self.turnspeed = 0.2
        self.speed = 0
        self.acceleration = 0.6
        self.dampening = 0.92
        self.turnspeed_dampening = 0.7
        self.min_speed = 0
        self.max_speed = 10
        
        # Creating sensors 
        self.rad_range = 60
        self.lin_range = 100
        radial_angles = [
            (-1/24*pi, 1/24*pi), (1/24*pi, 1/2*pi), (1/2*pi, pi), 
            (pi, 3/2*pi), (3/2*pi, 47/24*pi)]
        linear_angles = [-1/6*pi, 0, 1/6*pi]
        self.sensors = [
            NormalValueSensor(max_values=[self.max_speed]),
            RadialAreaSensor(self.pos, self.angle, self.rad_range, radial_angles),
            LinearSensor(self.pos, self.angle, self.lin_range, linear_angles)]
        
        # Creating autopilot
        total_nr_sensors = sum([sen.nr_sensors for sen in self.sensors])
        self.autopilot = AutoPilot(
            input_dim=total_nr_sensors, hidden_layer_dim=6, output_dim=4)

        # Evolution and statistics
        self.alive = True
        self.consumed = 0
        self.travel_dist = 0
        self.fitness = 0
        self.fitness_func = lambda consumed, travel_dist, alive: (
            (consumed + (travel_dist/40)**(1/5)) * (0.5 + 0.25*self.alive))
        
        #self.fitness_func = lambda consumed, travel_dist, alive: (
        #    (consumed + min(3, travel_dist/20)) * (0.5 + 0.25*self.alive))
        
    def _create_polygon(self):
        """
        Defines the bot shapely polygon that serves as a hitbox, food eating
        range and drawing shape.
        """
        polygon_points = [
            self.pos + (self.size, 0), 
            self.pos + (0, self.size/3), 
            self.pos - (self.size/3, 0), 
            self.pos - (0, self.size/3)]
        polygon = Polygon(LineString(polygon_points))
        polygon = rotate(
            polygon, self.angle, origin=self.pos, use_radians=True)
        return polygon
    
    def update(self, food, food_distances, barriers):
        """
        Steering, eating and updating fitness.
        """
        # Alive check
        if not self.alive:
            return
        
        # Update sensors
        inrange_food = food_distances <= self.rad_range
        sensor_input_dict = {
            'input_values': [self.speed],
            'bot_pos': self.pos, 
            'food_pos': food.pos[inrange_food], 
            'food_distances': food_distances[inrange_food],
            'barrier_polygons': barriers.polygon}
        sensor_values = [sen.read(**sensor_input_dict) for sen in self.sensors]        
        sensor_values = np.concatenate(sensor_values)
        
        # Execute actions
        actions = self.autopilot.action_cutoff(sensor_values, 0.5)        
        for action in actions:
            if action == 0:
                self.accelerate(self.SPEED_UP)
            elif action == 1:
                self.accelerate(self.SPEED_DOWN)
            elif action == 2:
                self.rotate(self.TURN_LEFT)
            elif action == 3:
                self.rotate(self.TURN_RIGHT)
        
        # Update position and apply speed/turnspeed dampening
        self.translate()
        self.speed = self.speed * self.dampening
        
        # Eat and check death
        inrange_food_indices = np.where(food_distances < self.size)[0]
        food_value = food.eat(inrange_food_indices)
        self.consumed += food_value
        self.check_death(barriers)
        
        # Update fitness
        self.fitness = self.fitness_func(self.consumed, self.travel_dist, self.alive)

    def accelerate(self, direction):
        """
        Updates the bot speed.
        """
        speed_new = (self.speed + direction * self.acceleration)
        self.speed = min(self.max_speed, max(self.min_speed, speed_new))

    def rotate(self, direction):
        """
        Rotates the bot and sensors. Turnspeed is adjusted based on the current
        bot speed, where higher speed equals lower turnspeed.
        """
        scaler = 1 - self.turnspeed_dampening * (self.speed / self.max_speed)
        adjusted_turnspeed = scaler * self.turnspeed        
        delta_angle = direction * adjusted_turnspeed
        self.angle += delta_angle
        self.polygon = rotate(
            self.polygon, delta_angle, origin=self.pos, use_radians=True)
        for sensor in self.sensors:
            sensor.rotate(self.pos, delta_angle)
        
        
    def translate(self):
        """
        Updates the position of the bot and sensors.
        """
        delta_pos = self.speed * np.array([cos(self.angle), sin(self.angle)])
        self.pos = self.pos + delta_pos
        
        
        self.polygon = translate(self.polygon, *delta_pos)
        
        
        
        for sensor in self.sensors:
            sensor.translate(delta_pos)
        self.travel_dist += np.linalg.norm(delta_pos)
    
    def check_death(self, barriers):
        """
        Checks for starvation and collision with barriers. 
        """
        for pol in barriers.polygon:
            if self.polygon.intersects(pol):
                self.alive = False
                self.color = Color.GREY4
    
    def random_genome(self):
        self.autopilot.network.random_init()
        
    def set_genome(self, genome):
        self.autopilot.network.set_genome(genome)

    def get_genome(self):
        return self.autopilot.network.get_genome()
        
    def draw(self, screen, pan_offset, zoom):
        pol_points = np.array(self.polygon.exterior.coords.xy).T        
        pol_points_draw = (pol_points * zoom + pan_offset).astype(int)
        gfxdraw.aapolygon(screen, pol_points_draw, self.color)
        pygame.draw.polygon(screen, self.color, pol_points_draw)
        
    def debug_draw(self, screen, pan_offset, zoom):
        for sensor in self.sensors:
            sensor.draw(screen, pan_offset, zoom)
        pol_points = np.array(self.polygon.exterior.coords.xy).T        
        pol_points_draw = (pol_points * zoom + pan_offset).astype(int)        
        gfxdraw.aapolygon(screen, pol_points_draw, self.color)
        pygame.draw.polygon(screen, self.color, pol_points_draw)



class HerbivorePopulation:
    """
    Population of bots that eat food objects.
    """
    
    def __init__(self, bot, pop_size, genetic_algorithm, use_debug_bot=False):
        self.bot = bot
        self.pop_size = pop_size
        self.genalg = genetic_algorithm
        
        # Initializing lists of all individuals and current batch individuals
        self.individuals = []
        self.batch_individuals = []
        self.current_batch = 0
        
        # Mutation function, statistics, and debug
        self.use_debug_bot = use_debug_bot
        self.debug_bot = None
    
    def initialize(self, object_sets):
        """
        Creates a new population of bots with random positions and genomes. 
        The object_sets parameter should contain a set of barriers from which
        a feasible spawn_area is determined.
        """
        
        # Resetting individuals and batch
        self.current_batch = 0
        self.individuals = []
        self.batch_individuals = []
        
        # Creating rest of population
        spawn_area = object_sets['barriers'].get_safe_area(30)
        for i in range(self.pop_size):
            bot_pos = object_sets['barriers'].get_safe_point(spawn_area)
            bot_angle = np.random.uniform(0, 2*pi)
            ind = self.bot(bot_pos, bot_angle)
            ind.random_genome()
            self.individuals.append(ind)
        
    def next_batch(self, nr_batches):
        """
        Moves the next batch of individuals from the individuals list to the
        batch_individuals list to prepare them for simulation. Only the total
        number of batches is given as a parameter such that the batch size can
        be determined for each population seperately.
        """
        batch_size = int(ceil(len(self.individuals) / nr_batches))        
        i_start = self.current_batch * batch_size
        i_end = i_start + batch_size
        self.batch_individuals = self.individuals[i_start:i_end]
        self.current_batch += 1 
     
        # Assinging debug bot to first individual in batch
        if self.use_debug_bot:
            self.debug_bot = self.batch_individuals[0]
            self.debug_bot.color = Color.RED2
        
    def update(self, object_sets, populations):
        """
        Creates a distance matrix between all herbive bots and food objects,
        and calls bot updates.
        """
        food = object_sets['food']
        bot_pos = np.array([ind.pos for ind in self.batch_individuals])
        food_distances = cdist(bot_pos, food.pos)
        barriers = object_sets['barriers']
        for i, ind in enumerate(self.batch_individuals):
            ind.update(food, food_distances[i], barriers)
     
    def get_fitness(self, batch_only=False):   
        """
        Returns an array of the fitness of each individual in the population.
        If batch_only==True, only returns individuals from the current batch.
        """
        fit_ind = self.batch_individuals if batch_only else self.individuals
        return np.array([ind.fitness for ind in fit_ind])
        
    def get_genomes(self, batch_only=False):
        """
        Returns an array of the genomes of each individual in the population.
        If batch_only==True, only returns individuals from the current batch.
        """
        if batch_only:
            return np.array([ind.get_genome() for ind in self.batch_individuals])
        else:
            return np.array([ind.get_genome() for ind in self.individuals])
        
    def set_genomes(self, new_genomes):
        for i in range(self.pop_size):
            self.individuals[i].set_genome(new_genomes[i])
      
    def evolve(self):
        """
        Creates a new population by evolving the current population.
        """
        genomes = self.get_genomes()
        fitness = self.get_fitness()
        new_genomes = self.genalg.evolve_population(genomes, fitness, (-1,1))
        return new_genomes

    def draw(self, screen, pan_offset, zoom):
        """
        Draws all indivuals on the screen
        """
        for ind in self.batch_individuals:
            if ind is self.debug_bot:
                ind.debug_draw(screen, pan_offset, zoom)
            else:
                ind.draw(screen, pan_offset, zoom)

    def arrow_key_left(self):
        self.debug_bot.rotate(self.debug_bot.TURN_LEFT)

    def arrow_key_right(self):
        self.debug_bot.rotate(self.debug_bot.TURN_RIGHT)
            
    def arrow_key_up(self):
        self.debug_bot.accelerate(self.debug_bot.SPEED_UP)
            
    def arrow_key_down(self):
        self.debug_bot.accelerate(self.debug_bot.SPEED_DOWN)

