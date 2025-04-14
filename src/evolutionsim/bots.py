from sensors import AutoPilot, NormalValueSensor, RadialAreaSensor, LinearSensor
import numpy as np
import pygame
from pygame import gfxdraw
from math import pi, sin, cos, ceil
from algorithms.geometry.shapes import Polygon
from scipy.spatial.distance import cdist
from pygametools.color.color import Color, ColorGradient
from abc import ABC, abstractmethod


class ArrowBot(ABC):

    """
    Arrow shaped bot that can accelerate, break and steer.
    """

    # Actions (autopilot decisions):
    TURN_LEFT = -1
    TURN_RIGHT = 1
    SPEED_UP = 1
    SPEED_DOWN = -1

    def __init__(self, bot_pos, bot_angle, **kwargs):

        # Position and orientation variables
        self.pos = np.array(bot_pos)
        self.angle = bot_angle

        # Body and color
        self.size = kwargs.get('size', 12)
        self.color = kwargs.get('color', Color.BLUE3)
        self.polygon = self._create_polygon()

        # Movement paramters
        self.turn_speed = kwargs.get('turn_speed', 0.2 )
        self.speed = kwargs.get('speed', 0)
        self.acceleration = kwargs.get('acceleration', 0.6)
        self.speed_dampening = kwargs.get('speed_dampening', 0.92)
        self.turnspeed_dampening = kwargs.get('turnspeed_dampening', 0.7)
        self.min_speed = kwargs.get('', 0)
        self.max_speed = kwargs.get('', 10)

        # IMPLEMENT IN SUB CLASSES
        self.autopilot = None

        # Sensor list used for external analysis
        self.sensors = []

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
        polygon = Polygon(polygon_points)
        polygon.rotate(self.angle, self.pos)
        return polygon

    def accelerate(self, direction):
        """
        Updates the bot speed. The directon parameter can be either
        self.SPEED_DOWN or self.SPEED_DOWN.
        """
        speed_new = (self.speed + direction * self.acceleration)
        self.speed = min(self.max_speed, max(self.min_speed, speed_new))

    def rotate(self, direction):
        """
        Rotates the bot and sensors. Turnspeed is adjusted based on the current
        bot speed, where higher speed equals lower turnspeed. The direction
        parameter can be either self.TURN_LEFT or self.TURN_RIGHT.
        """

        # Compute turn_speed adjusted delta angle
        scaler = 1 - self.turnspeed_dampening * (self.speed / self.max_speed)
        adjusted_turnspeed = scaler * self.turn_speed
        delta_angle = direction * adjusted_turnspeed

        # Always set angle in 0, 2pi domain
        while self.angle + delta_angle >= 2*pi:
            delta_angle -= 2*pi
        while self.angle + delta_angle < 0:
            delta_angle += 2*pi

        # Set angle and rotate self and sensors
        self.angle += delta_angle
        self.polygon.rotate(delta_angle, self.pos)
        for sensor in self.sensors:
            sensor.rotate(delta_angle, self.pos)

    def translate(self):
        """
        Updates the position of the bot and sensors.
        """
        delta_pos = self.speed * np.array([cos(self.angle), sin(self.angle)])
        self.pos = self.pos + delta_pos
        self.polygon.translate(delta_pos)
        for sensor in self.sensors:
            sensor.translate(delta_pos)
        self.speed = self.speed * self.speed_dampening

    def random_genome(self):
        """
        Randomly initialize the bot genome.
        """
        self.autopilot.network.random_init()

    def set_genome(self, genome):
        """
        Set the bot genome acording with a numpy array.
        """
        self.autopilot.network.set_genome(genome)

    def get_genome(self):
        """
        Fetch the bot genome as a numpy array.
        """
        return self.autopilot.network.get_genome()

    def draw(self, screen, pan_offset, zoom):
        """
        Draw the bot on the given screen.
        """
        pol_points_draw = (self.polygon.points * zoom + pan_offset).astype(int)
        gfxdraw.aapolygon(screen, pol_points_draw, self.color)
        pygame.draw.polygon(screen, self.color, pol_points_draw)

    def debug_draw(self, screen, pan_offset, zoom):
        """
        Draw the bot on the given screen in debug mode.
        """
        for sensor in self.sensors:
            sensor.draw(screen, pan_offset, zoom)
        pol_points_draw = (self.polygon.points * zoom + pan_offset).astype(int)
        gfxdraw.aapolygon(screen, pol_points_draw, self.color)
        pygame.draw.polygon(screen, self.color, pol_points_draw)

    @abstractmethod
    def update(self):
        """
        IMLPEMENT IN SUB CLASS
        Called at each simulation tick. Should read sensors, activate
        autopilot, accelerate/rotate/translate, updated any stats and
        bot attributes and compute self.fitness.
        """
        pass



class Population(ABC):
    """
    Population of bots that eat food objects.
    """

    def __init__(self, bot, pop_size, genetic_algorithm, use_debug_bot):
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

    def initialize(self, object_sets=None, populations=None):
        """
        Creates a new population of bots with random positions and genomes.
        The object_sets and populations can be dictionaries of objects and
        other populations in the simulation that may be used for creating
        the bot (such as making sure a bot is not initialized on a wall.).
        """
        # Resetting individuals and batch
        self.current_batch = 0
        self.individuals = []
        self.batch_individuals = []

        # Creating individuals
        for i in range(self.pop_size):
            ind = self.create_bot(object_sets, populations)
            ind.random_genome()
            self.individuals.append(ind)

    def next_batch(self, nr_batches):
        """
        Moves the next batch of individuals from the individuals list to the
        batch_individuals list to prepare them for simulation. Only the total
        number of batches is given as a parameter such that the batch size can
        be determined for different populations seperately.
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
        gen_ind = self.batch_individuals if batch_only else self.individuals
        return np.array([ind.get_genome() for ind in gen_ind])

    def set_genomes(self, new_genomes):
        """
        Sets the genomes for a whole population based on a given numpy
        array with shape (num_individuals, genome_len).
        """
        for i in range(self.pop_size):
            self.individuals[i].set_genome(new_genomes[i])

    def evolve(self):
        """
        Creates a new population by evolving the current population.
        """
        genomes = self.get_genomes()
        fitness = self.get_fitness()
        new_genomes = self.genalg.evolve_population(genomes, fitness)
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

    @abstractmethod
    def key_input(self):
        """
        Define input behaviour for population or bot. Can for user input such
        as controlling bots with arrowkeys.
        """
        pass

    @abstractmethod
    def create_bot(self):
        """
        IMLPEMENT IN SUB CLASS
        Create a new bot to be added to the population.
        """
        pass

    @abstractmethod
    def update(self):
        """
        IMLPEMENT IN SUB CLASS
        Called at each simulation tick. Update each bot in the population.
        """
        pass





class Herbivore(ArrowBot):

    def __init__(self, bot_pos, bot_angle, **kwargs):
        super().__init__(bot_pos, bot_angle, **kwargs)

        # Individual attributes and statistics
        self.max_vitality = 10000
        self.vitality = 10000
        self.start_pos = self.pos.copy()
        self.alive = True
        self.consumed = 0
        self.travel_dist = 0
        self.fitness = 0

        # Color for maximum vitality
        self.col_full = self.color
        self.col_empty = Color.ORANGE3
        self.col_gradient = ColorGradient(self.col_empty, self.col_full)

        # Initialize sensors
        self.sens_norm = NormalValueSensor(
            max_values=[self.max_speed, self.max_vitality])
        self.rad_range = 70
        radial_angles = [
            (0, 1/3*pi),
            (1/3*pi, pi),
            (pi, 5/3*pi),
            (5/3*pi, 2*pi)]
        self.sens_rad = RadialAreaSensor(
            self.pos, self.angle, self.rad_range, radial_angles)
        self.lin_range = 120
        linear_angles = [-1/5*pi, -1/22*pi, 1/22*pi, 1/5*pi]
        self.sens_lin = LinearSensor(
            self.pos, self.angle, self.lin_range, linear_angles)

        # Create list of all sensors
        self.sensors = [self.sens_norm, self.sens_rad, self.sens_lin]

        # Initialize autopilot
        nr_sensors = sum([sens.nr_sensors for sens in self.sensors])
        self.autopilot = AutoPilot(
            input_dim=nr_sensors, hidden_layer_dim=6, output_dim=4)

    def update(self, food, food_distances, barriers):
        """
        Reads sensors, executing actions, computes fitness.
        """
        # Alive check
        if not self.alive:
            return

        # Update sensors
        val_norm = self.sens_norm.read([self.speed, self.vitality])
        inrange_food = food_distances <= self.rad_range
        val_rad = self.sens_rad.read(
            food.pos[inrange_food], food_distances[inrange_food])
        val_lin = self.sens_lin.read(self.pos, barriers.polygons)
        self.sensor_values = np.concatenate([val_norm, val_rad, val_lin])

        # Execute actions
        actions = self.autopilot.action_cutoff(self.sensor_values, 0.5)
        for action in actions:
            if action == 0:
                self.accelerate(self.SPEED_UP)
            elif action == 1:
                self.accelerate(self.SPEED_DOWN)
            elif action == 2:
                self.rotate(self.TURN_LEFT)
            elif action == 3:
                self.rotate(self.TURN_RIGHT)
        self.translate()

        # Eat (fitnes and vitality)
        inrange_food_indices = np.where(food_distances < self.size)[0]
        food_value = food.eat(inrange_food_indices)
        self.consumed += food_value

        # Compute vitality and new color
        self.vitality = self.vitality - 0.2 - (self.speed / self.max_speed)
        if food_value > 0:
            self.vitality = self.max_vitality
        self.color = self.col_gradient.get_color(self.vitality / 100)

        # check death
        if self.vitality < 0:
                self.alive = False
                self.color = Color.GREY4
        for pol in barriers.polygons:
            if self.polygon.intersect_pol_bool(pol):
                self.alive = False
                self.color = Color.GREY4
                break

        # Compute furthest distance from start (fitness)
        dist_from_start = np.linalg.norm(self.start_pos - self.pos)
        self.travel_dist = max(self.travel_dist, dist_from_start)

        # Computing fitness
        self.fitness = (
            self.consumed + (self.travel_dist/40)**(1/5)) * (0.5 + 0.25*self.alive)



class HerbivorePopulation(Population):

    def __init__(self, bot, pop_size, genetic_algorithm, use_debug_bot=False):
        super().__init__(bot, pop_size, genetic_algorithm, use_debug_bot)

    def create_bot(self, object_sets, populations):
        """
        Create a new bot to be added to the population.
        """
        bot_pos = object_sets['barriers'].get_safe_point()
        bot_angle = np.random.uniform(0, 2*pi)
        return self.bot(bot_pos, bot_angle)

    def update(self, object_sets, populations):
        """
        Creates a distance matrix between all herbive bots and food objects,
        and calls bot updates.
        """
        food, barriers = object_sets['food'], object_sets['barriers']

        bot_pos = np.array([ind.pos for ind in self.batch_individuals])
        food_distances = cdist(bot_pos, food.pos)

        for i, ind in enumerate(self.batch_individuals):
            ind.update(food, food_distances[i], barriers)

    def key_input(self):
        """
        Define input behaviour for population or bot. Can for user input such
        as controlling bots with arrowkeys.
        """
        # TODO: debug bot control
        pass






class FlockBot(ArrowBot):

    def __init__(self, bot_pos, bot_angle, **kwargs):
        super().__init__(bot_pos, bot_angle, **kwargs)

        self.turn_speed = kwargs.get('turn_speed', 0.3 )
        self.turnspeed_dampening = kwargs.get('turnspeed_dampening', 0.2)
        self.min_speed = kwargs.get('', 0)
        self.max_speed = kwargs.get('', 8)

        # Initialize sensors
        self.sens_norm = NormalValueSensor(
            max_values=[self.max_speed, 10, 2*pi])
        self.rad_range = 100
        radial_angles = [
            (-1/4*pi, 1/4*pi),
            (1/4*pi, 3/4*pi),
            (3/4*pi, 5/4*pi),
            (5/4*pi, 7/4*pi)]
        self.sens_rad = RadialAreaSensor(
            self.pos, self.angle, self.rad_range, radial_angles)
        self.lin_range = 120
        linear_angles = [-1/5*pi, -1/22*pi, 1/22*pi, 1/5*pi]
        self.sens_lin = LinearSensor(
            self.pos, self.angle, self.lin_range, linear_angles)

        # Create list of all sensors
        self.sensors = [self.sens_norm, self.sens_rad, self.sens_lin]

        # Initialize autopilot
        nr_sensors = sum([sens.nr_sensors for sens in self.sensors])
        self.autopilot = AutoPilot(
            input_dim=nr_sensors, hidden_layer_dim=6, output_dim=4)

        # Evolution and statistics
        self.alive = True
        self.fitness = 0

    def update(self, flock_distance, flock_pos, flock_angle, barriers):
        """
        Steering and updating fitness.
        """
        # Alive check
        if not self.alive:
            return

        # Update sensors
        close_bots = (flock_distance < self.rad_range)
        flock_num = min(10, close_bots.sum())
        if flock_num == 0:
            flock_angle_diff = 2*pi
        else:
            mean_angle = flock_angle[close_bots].mean()
            diff1 = mean_angle - self.angle
            diff2 = mean_angle - 2*pi - self.angle
            diff3 = mean_angle + 2*pi - self.angle
            diff = np.array([diff1, diff2, diff3])
            flock_angle_diff = diff[np.argmin(np.abs(diff))]
        val_norm = self.sens_norm.read([self.speed, flock_num, flock_angle_diff])
        val_rad = self.sens_rad.read(
            flock_pos[close_bots], flock_distance[close_bots])
        val_lin = self.sens_lin.read(self.pos, barriers.polygons)
        self.sensor_values = np.concatenate([val_norm, val_rad, val_lin])

        # Execute actions
        actions = self.autopilot.action_cutoff(self.sensor_values, 0.5)
        for action in actions:
            if action == 0:
                self.accelerate(self.SPEED_UP)
            elif action == 1:
                self.accelerate(self.SPEED_DOWN)
            elif action == 2:
                self.rotate(self.TURN_LEFT)
            elif action == 3:
                self.rotate(self.TURN_RIGHT)
        self.translate()

        # check death


        for pol in barriers.polygons:
            if len(self.polygon.intersect_pol(pol)) > 0:
                self.alive = False
                self.color = Color.GREY4
        """

        if (flock_distance[close_bots] < self.size).any():
            self.alive = False
            self.color = Color.GREY4
        """

        # Update fitness
        speed, flock_num, flock_angle_diff = val_norm
        self.fitness += speed * (1 - flock_angle_diff) * flock_num


class FlockPopulation(Population):

    def __init__(self, bot, pop_size, genetic_algorithm, use_debug_bot=False):
        super().__init__(bot, pop_size, genetic_algorithm, use_debug_bot)

    def create_bot(self, object_sets, populations):
        """
        Create a new bot to be added to the population.
        """
        bot_pos = object_sets['barriers'].get_safe_point()
        bot_angle = np.random.uniform(0, 2*pi)
        return self.bot(bot_pos, bot_angle)

    def update(self, object_sets, populations):
        """
        Creates a distance matrix between all herbive bots and food objects,
        and calls bot updates.
        """

        flock_pos = np.array([ind.pos for ind in self.batch_individuals])
        flock_angle = np.array([ind.angle for ind in self.batch_individuals])
        flock_distance = cdist(flock_pos, flock_pos)
        barriers = object_sets['barriers']

        for i, ind in enumerate(self.batch_individuals):
            m = np.ones(len(self.batch_individuals)).astype(bool)
            m[i] = False
            ind.update(
                flock_distance[i][m], flock_pos[m], flock_angle[m], barriers)

