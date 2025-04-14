from sensors import AutoPilot, RadialAreaSensor, LinearSensor
from interface import Color
import numpy as np
import pygame
from math import pi, sin, cos
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, LineString
import genetic_algorithm as ga
from scipy.spatial.distance import cdist
from scipy import dot, array
import time

"""
TODO:
    - https://geoffboeing.com/2016/10/r-tree-spatial-index-python/
    - Starvation with bot coloring based on its level
    - Allowed starting area based on barriers
"""

class Herbivore:
    
    TURN_LEFT = -1
    TURN_RIGHT = 1
    SPEED_UP = 1
    SPEED_DOWN = -1
    
    def __init__(self, bot_pos, bot_angle):
        
        # Body and positioning
        self.size = 8
        self.color = Color.DBLUE
        self.pos = np.array(bot_pos)
        self.angle = bot_angle
        self.polygon = self._create_polygon()
        
        # Movement
        self.turnspeed = 0.1
        self.speed = 1
        self.acceleration = 0.1
        self.dampening = 0.99
        self.min_speed = 0
        self.max_speed = 20
        
        # Sensors and autopilot
        self.rad_range = 80
        self.lin_range = 200
        
        self.autopilot = AutoPilot(input_dim=9, hidden_layer_dim=6, output_dim=4)
        radial_angles = [
            (-1/24*pi, 1/24*pi), (1/24*pi, 1/2*pi), (1/2*pi, pi), 
            (pi, 3/2*pi), (3/2*pi, 47/24*pi)]
        linear_angles = [-1/6*pi, 0, 1/6*pi]
        self.sensors = [
            RadialAreaSensor(self.pos, self.angle, self.rad_range, radial_angles),
            LinearSensor(self.pos, self.angle, self.lin_range, linear_angles)]

        # Evolution and statistics
        self.alive = True
        self.consumed = 0
        self.travel_dist = 0
        self.fitness = 0
        self.fitness_func = lambda consumed, travel_dist, alive: (
            (consumed + min(3, travel_dist/20)) * (0.75 * self.alive))
    
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
        
        sensor_values = [sensor.read(
            bot_pos = self.pos, 
            food_pos = food.pos[inrange_food], 
            food_distances = food_distances[inrange_food],
            barrier_polygons = barriers.polygon)
            for sensor in self.sensors]
        sensor_values.append([self.speed / self.max_speed])
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
        
        # Update position and apply speed dampening
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
        Rotates the bot and sensors.
        """
        delta_angle = direction * self.turnspeed
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
    
    def random_init(self):
        self.autopilot.network.random_init()
        
    def set_genome(self, genome):
        self.autopilot.network.set_genome(genome)

    def get_genome(self):
        return self.autopilot.network.get_genome()
        
    def draw(self, screen, pan_offset, zoom):
        pol_points = np.array(self.polygon.exterior.coords.xy).T        
        pol_points_draw = (pol_points * zoom + pan_offset).astype(int)
        pygame.draw.polygon(screen, self.color, pol_points_draw)
        
    def debug_draw(self, screen, pan_offset, zoom):
        for sensor in self.sensors:
            sensor.draw(screen, pan_offset, zoom)
        pol_points = np.array(self.polygon.exterior.coords.xy).T        
        pol_points_draw = (pol_points * zoom + pan_offset).astype(int)        
        pygame.draw.polygon(screen, self.color, pol_points_draw)


class HerbivorePopulation:
    """
    Population of bots that eat food objects.
    """
    
    def __init__(self, bot, pop_size, window_size):
        self.pop_size = pop_size
        self.window_size = window_size
        self.bot = bot
        self.individuals = []
        self.mutation_func = lambda fit: round(max(0.05, 0.3 - fit * 0.4),3)
        self.debug_bot = None
    
    def initialize(self):
        """
        Creates a new population of bots with random positions and genomes.
        """
        self.individuals = []
        self.debug_bot = self.bot((100,100), 0.5 * pi)
        self.debug_bot.color = Color.MRED
        self.individuals.append(self.debug_bot)
        
        for i in range(self.pop_size):
            bot_pos = np.random.uniform((0,0), self.window_size, 2)
            bot_angle = np.random.uniform(0, 2*pi)
            ind = self.bot(bot_pos, bot_angle)
            ind.random_init()
            self.individuals.append(ind)
      
        
    def update(self, object_sets, populations):
        """
        Creates a distance matrix between all herbive bots and food objects,
        and calls bot updates.
        """
        food = object_sets['food']
        bot_pos = np.array([ind.pos for ind in self.individuals])
        food_distances = cdist(bot_pos, food.pos)
        barriers = object_sets['barriers']
        for i, ind in enumerate(self.individuals):
            ind.update(food, food_distances[i], barriers)
      
    def evolve(self):
        """
        Creates a new population by evolving the current population.
        """
        genomes = np.array([ind.get_genome() for ind in self.individuals])
        fitness = np.array([ind.fitness for ind in self.individuals])
        selection = ga.SelectionTournament(k=5)
        crossover = ga.CrossoverMultipoint(n=2)
        mutations = [
            ga.MutationUniformReplacement(p=self.mutation_func(fitness.mean())), 
            ga.MutationAdjustment(p=0.05, adjustment_range=(-0.1,0.1))]
        genalg = ga.GeneticAlgorithm(
            genomes, fitness, selection, crossover, mutations, elitism=1, 
            copy_fract=0.1, judgement_day_std=0.0)
        new_genomes = genalg.evolve_population()
        self.initialize()
        for i in range(self.pop_size):
            self.individuals[i].set_genome(new_genomes[i])
        
        # Debug
        self.debug_bot = self.individuals[0]
    

    def draw(self, screen, pan_offset, zoom):
        """
        Draws all indivuals on the screen
        """
        for ind in self.individuals:
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

