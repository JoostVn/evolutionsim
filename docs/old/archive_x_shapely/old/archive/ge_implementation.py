from Color import Color, ColorGradient
import pygame
import numpy as np
from NeuralNetwork import NeuralNetwork, Layer, Activation
from math import sin, cos, tan, atan2, pi, floor, ceil
import Tools
from matplotlib import pyplot as plt
from GA import Selection, Crossover, Mutation

"""
TODO:
    - Better sensor visualization with circle
    - Plotting and visualization of statistics after echt generation
    - Visual sensors for best individual
    - X and Y distance to mid as extra sensor OR walls with dead
    - Real time neural net activations
    
    - Sensor ideas:
        - Distance to center
        - Distance to wall (directional)
        - Combine front sensor into on
        
    
    
    - Main menu and options:
        - Custom generation size and duration
        - Start geneneration option
        - Start x generations option
        - Saving best individual
        - Loading individual
"""
            
    
class Autopilot:
    
    def __init__(self, bot, sensor_rad):
        self.bot = bot
        self.sensor_rad = sensor_rad
        self.sensor_angles = [
            (0, 1/6*pi),
            (1/6*pi, 3/6*pi),
            (3/6*pi, 6/6*pi),
            (6/6*pi, 9/6*pi),
            (9/6*pi, 11/6*pi),
            (11/6*pi, 12/6*pi)]
        
        self.nr_sensors = len(self.sensor_angles) + 1
        self.net = self.creat_nn()
        self.sensor_values = np.zeros(self.nr_sensors)
    
    def creat_nn(self):
        net = NeuralNetwork(input_dim=self.nr_sensors)
        net.add(Layer(size=4, input_dim=self.nr_sensors, 
                      activation_func=Activation.sigmoid))
        net.add(Layer(size=4, input_dim=4, 
                      activation_func=Activation.sigmoid))
        return net
    
    
    
    def nn_init(self, genome=None):
        """
        Sets all weights and biases in the neural network according to 
        a genome. If no genome is given, randomly intitializes.
        """
        if genome is None:
            self.net.random_init(domain=[-1,1])
        else: 
            self.net.set_genome(genome)
        
    def steer(self, food):
        """
        Updates all sensor and interprets their values to trigger an action
        in self.bot.
        """
        self.read_sensors(food)
        output_values, selection = self.net.forward_pass(self.sensor_values)

        actions = [
            self.bot.arrow_up,
            self.bot.arrow_down,
            self.bot.arrow_left,
            self.bot.arrow_right]
        
        for i, val in enumerate(output_values):
            if val > 0.5:
                actions[i]()
        
    def read_sensors(self, food):
        """
        Checks food sources within range and computes the closest food source
        in each sensor area. Also adds additional sensor information such
        as speed.
        """
        r = self.sensor_rad
        close_food = [f for f in food if 
                      Tools.distance(f.pos, self.bot.pos) < r]
        for i, angles in enumerate(self.sensor_angles):
            if not close_food:
                self.sensor_values[i] = 1
            else:
                dist = min([self.check_sensor_area(f, angles) for f in close_food])
                self.sensor_values[i] = dist / r 
        
        # Read current speed and standardize by max speed
        self.sensor_values[-1] = (self.bot.speed / self.bot.maxspeed)

    def check_sensor_area(self, food_item, angles):
        """
        Checks if a food source lies in a sensor area by computing the angle
        to the food source relative to the direction angle, and checking if 
        this angle lies between the two area defining angles.
        """
        food_vector = food_item.pos - self.bot.pos
        food_angle = Tools.convert_to_angle(food_vector)
        relative_angle = Tools.standard_angle(food_angle - self.bot.angle)
        if (angles[0] <= relative_angle < angles[1]):
            return Tools.distance(self.bot.pos, food_item.pos)
        else:
            return self.sensor_rad   
        
    def draw_sensors(self, screen, pan_offset):
        """
        Converts all sensor angles to points on the screen and draws as 
        simple traingle between the points of each sensor area according
        to self.sensor_rad. Triangles are colored according to sensor value.
        """
        pos_draw = (self.bot.pos + pan_offset).astype(int)
        gradient = ColorGradient(Color.LGREEN, Color.LGREY, 100).color_vector
        for angles, value in zip(self.sensor_angles, self.sensor_values):
            vector1 = Tools.convert_to_vector(angles[0] + self.bot.angle)
            vector2 = Tools.convert_to_vector(angles[1] + self.bot.angle)
            point1 = self.sensor_rad * vector1 + pos_draw
            point2 = self.sensor_rad * vector2 + pos_draw
            triangle = [pos_draw, point1, point2]
            col = gradient[int(round(99 * value))]
            pygame.draw.polygon(screen, col, triangle)

        


class Bot:
    
    def __init__(self, position, direction, speed):

        # Constants
        self.size = 25
        self.color = Color.random_dull()
        self.turnspeed = 0.2
        self.speed = 1     
        self.acceleration=0.20 
        self.dampening=0.97
        self.minspeed=0
        self.maxspeed=5     
               
        # Variables
        self.pos = np.array(position)        
        self.direction = direction
        self.angle = Tools.convert_to_angle(self.direction)
        
        # Fitness and autopilot
        self.alive = True
        self.starvation = 0
        self.fitness = 0
        self.autopilot = Autopilot(self, sensor_rad=160)
        
    def update(self, food):
        if self.alive:
            self.starvation += 1
            self.angle = Tools.convert_to_angle(self.direction)
            self.speed = self.dampening * self.speed              
            self.pos = self.pos + (self.speed * self.direction)
            self.eat(food)
            self.starve()
            self.autopilot.steer(food)
        
    def eat(self, food):
        for f in food:
            dist = Tools.distance(self.pos, f.pos)
            if dist < 6:
                self.fitness += 1
                self.starvation = 0
                food.remove(f)
                break
    
    def starve(self):
        if self.starvation > 300:
            self.alive = False
            self.color = Color.MGREY
    
    def draw(self, screen, pan_offset, sensors=False):
        pos_draw = (self.pos + pan_offset).astype(int)
        vertical = self.direction
        horizontal = np.flip(self.direction) * [-1, 1]
        
        # sensors
        if sensors:
            self.autopilot.draw_sensors(screen, pan_offset)
        
        # Body
        p1 = (pos_draw + self.size * 0.6 * vertical)
        p2 = (pos_draw + self.size * 0.22 * horizontal)
        p3 = (pos_draw - self.size * 0.2 * vertical)
        p4 = (pos_draw - self.size * 0.22 * horizontal)
        point_list = np.array((p1,p2,p3,p4)).astype(int)
        pygame.draw.polygon(screen, self.color, point_list)
        
    def arrow_left(self):
        new_angle = self.angle - self.turnspeed
        self.direction = Tools.convert_to_vector(new_angle)
            
    def arrow_right(self):
        new_angle = self.angle + self.turnspeed
        self.direction = Tools.convert_to_vector(new_angle)

    def arrow_up(self):
        self.speed = min(self.maxspeed, self.speed + self.acceleration)
    
    def arrow_down(self):
        self.speed = max(self.minspeed, self.speed - self.acceleration)

    def get_genome(self):
        return self.autopilot.net.get_genes()
        



class Food:
    
    def __init__(self, domain):
        self.pos = np.random.randint(*domain, 2)
    
    def draw(self, screen, pan_offset):
        pos_draw = (self.pos + pan_offset).astype(int)
        pygame.draw.circle(screen, Color.DGREY, pos_draw, 2)



class Button:
    
    def __init__(self, dimensions, text, shape_col, text_col, action):
        self.left, self.top, self.width, self.height = dimensions
        self.text = text
        self.shape_col = shape_col
        self.text_col = text_col
        self.font = pygame.font.SysFont('monospace', 15) 
        self.font_pos = np.array([self.left + 5, self.top + self.height/2 - 8]) 
        self.action = action

    def draw(self, screen):
        rect = pygame.Rect(self.left, self.top, self.width, self.height)
        pygame.draw.rect(screen, self.shape_col, rect) 
        textblock = self.font.render(self.text, True, self.text_col)
        screen.blit(textblock, self.font_pos)

    def check_click(self, pos):
        if ((self.left <= pos[0] <= self.left + self.width) and 
            (self.top <= pos[1] <= self.top + self.height)):
            return True
        else:
            return False


class SimpleGE:
    
    # States determine the current activity 
    INIT = 0
    HOLD = 1
    START_GENERATION = 2
    RUN_GENERATION = 3
    END_GENERATION = 4
    
    def __init__(self, nr_bots=20, food_quantity=60, generation_len=500):
        self.nr_bots = nr_bots
        self.food_quantity = food_quantity
        self.generation_len = generation_len
        self.generation = 0
        self.buttons = self.create_interface()
        self.state = self.INIT
        self.population = None
        self.food = None
        self.t = 0
        self.compute_mutation = lambda pop_fitness: 1/(5*pop_fitness+25)
        self.stats_fitness_avg = []
        self.stats_fitness_max = []
        self.stats_alive = []
        
    def update(self, i):
        """
        Update all program elements with one tick increment. Function gets 
        called from the Main class.
        """
        if self.state == self.START_GENERATION:  
            self.population = self.create_population(self.nr_bots)
            self.food = [Food((50, 750)) for i in range(self.food_quantity)]
            self.t = 0
            self.generation += 1
            self.state = self.RUN_GENERATION
    
        if self.state == self.RUN_GENERATION:  
            for bot in self.population:
                bot.update(self.food)
            while len(self.food) < self.food_quantity:
                self.food.append(Food((50, 750)))
            self.t += 1
            if self.t == self.generation_len:
                self.state = self.END_GENERATION
                
        if self.state == self.END_GENERATION:
            
            # Extract genetics and fitness from sorted individual list
            sorted_bots = sorted(
                self.population.copy(), key=lambda bot: bot.fitness, reverse=True)
            pop_fitness = np.array([bot.fitness for bot in sorted_bots])
            pop_genomes = np.array([bot.autopilot.net.get_genome() for bot in sorted_bots])
            
            # Reserve the best 2 genomes for the next generation
            nr_elites = 2
            elites = pop_genomes[:nr_elites]
            
            # Evolution
            parents = Selection.ranked(pop_genomes, num_parents=5)   
            number_parent_pairs = int(self.nr_bots/2 - nr_elites/2)
            pairs = Selection.select_pairs(parents, number_parent_pairs)
            offspring = Crossover.single_point(pairs)
            mutation_prob = self.compute_mutation(pop_fitness.mean())
            mutated_offspring = Mutation.uniform_replacement(offspring, mutation_prob)
            new_genomes = np.concatenate((elites, mutated_offspring))
            
            # Create next generation and prepare simulation
            self.population = self.create_population(
                self.nr_bots, new_genomes)
            self.food = [Food((50, 750)) for i in range(self.food_quantity)]
            self.t = 0
            self.generation += 1
            self.state = self.RUN_GENERATION
            
            # Analyze geration
            self.analyze_generation(pop_fitness, pop_genomes)
            
            # Set focus bot for new generation and visualize network
            self.focus_bot.autopilot.net.visualize_structure()
            
    def create_population(self, nr_bots, new_genomes=None):
        """
        Creates a population of bots. If no genome is given, randomly intializes
        all weights and biases for the bot's neural networks.
        """
        
        # Generate circular starting positions
        angles = np.linspace(0, 2*pi, nr_bots)
        directions = np.array([Tools.convert_to_vector(a) for a in angles])
        positions = np.array([400,400]) + directions * 100
        
        # Adding bots to the population
        population = []
        for i in range(nr_bots):
            bot = Bot(positions[i], directions[i], speed=4)
            
            if new_genomes is None:
                bot.autopilot.nn_init()
            else:
                bot.autopilot.nn_init(new_genomes[i])
                
            population.append(bot)
            
        # The first bot of a simulation (best of previous) is in focus
        self.focus_bot = population[0]
        self.focus_bot.color = Color.RED
        
        return population    
    
    def draw(self, screen, pan_offset):
        """
        Draw all program elements on the screen. Draws dead bots first, 
        to keep bots that are still alive on the front.  
        """
        
        # Draw simulation elements
        if self.state == self.RUN_GENERATION:
            self.focus_bot.draw(screen, pan_offset, sensors=True)
            for bot in self.population[1:]:
                if not bot.alive:
                    bot.draw(screen, pan_offset, sensors=False)
            for bot in self.population[1:]:
                if bot.alive:
                    bot.draw(screen, pan_offset, sensors=False)
            for food in self.food:
                food.draw(screen, pan_offset)
            
        # Draw interface
        for button in self.buttons:
            button.draw(screen)
        
    def information(self):
        info_list = []
        if self.state != self.INIT:
            fitness = np.array([bot.fitness for bot in self.population])
            info_list.append(f'Generation:      {self.generation}')
            info_list.append(f'Average fitness: {round(fitness.mean(),2)}')
            info_list.append(f'Max fitness:     {fitness.max()}')
            info_list.append(f'Bots alive:      {len(self.population)}')
            info_list.append(f'current t:       {self.t}')
        return info_list
    
    def create_interface(self):
        """
        Generates buttons on the screen for starting generations.
        """
        buttons = []
        
        # Run generation button
        left, top, width, height = 15, 750, 140, 30
        b1 = Button(
            dimensions = (left, top, width, height),
            text = 'Run generation',
            shape_col = Color.MGREY,
            text_col = Color.DGREY,
            action = self.START_GENERATION)
        buttons.append(b1)
        
        return buttons
        
    def analyze_generation(self, pop_fitness, pop_genomes):
        
        
        # computing stats
        mean_fitness = round(pop_fitness.mean(),2)
        max_fitness = round(pop_fitness.max(),2)
        self.stats_fitness_avg.append(mean_fitness)
        self.stats_fitness_max.append(max_fitness)
        
        # Moving averages
        n = 8
        fit = np.concatenate((np.zeros(n-1),self.stats_fitness_avg))
        fitness_avg_moving = [fit[i:i+n].mean() for i in range(self.generation-1)]
        best = np.concatenate((np.zeros(n-1),self.stats_fitness_max))
        fitness_max_moving = [best[i:i+n].mean() for i in range(self.generation-1)]
        
        # Plots
        plt.figure(figsize=(12,8))
        plt.plot(self.stats_fitness_avg, label='Average fitness', 
                 color='steelblue', alpha=0.5)
        plt.plot(self.stats_fitness_max, label='Max fitness', 
                 color='maroon', alpha=0.5)
        plt.plot(fitness_avg_moving, label='Moving average (avg fitness)', 
                 color='steelblue')
        plt.plot(fitness_max_moving, label='Moving average (max fitness)', 
                 color='maroon')
        plt.legend(loc='upper left', fontsize=14).get_frame().set_linewidth(0)
        plt.xlabel('Generation', fontsize=14)
        plt.title('Generation progression', fontsize=25)
        plt.show()
        
    def mouse_left_draw(self, start_pos, end_pos, pan_offset):
        """
        Interpeted as a click. If start_pos and end_pos are on a button,
        the button is triggered.
        """
        for button in self.buttons:
            if button.check_click(start_pos):
                self.state = button.action
                break
            
    def arrow_left(self):
        for bot in self.population:
            bot.arrow_left()
            
    def arrow_right(self):
        for bot in self.population:
            bot.arrow_right()

    def arrow_up(self):
        for bot in self.population:
            bot.arrow_up()
    
    def arrow_down(self):
        for bot in self.population:
            bot.arrow_down()
    
    









