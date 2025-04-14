import Tools
from Generation import Generation
from Color import Color
import numpy as np
from Interface import Button, SideBar
from Food import Food

class Simulation:
    """
    Handles the simulation interface and triggers the startup and completion
    of generations. Also handles food or obstacles in the simulation.
    """
    
    HOLD = 0
    INITIALIZE = 1
    RUN_GENERATION = 2
    END_GENERATION = 3
    EVOLVE_GENERATION = 4
    ANALYZE_GENERATION = 5
    RUN_FROM_GENOME = 6
    
    def __init__(self, window_size, midpoint, generation, generation_len, food_quantity, **kwargs):
        self.window_size = np.array(window_size)
        self.midpoint = np.array(midpoint)
        self.state = self.HOLD
        self.gen = generation
        self.food_quantity = food_quantity
        self.food = []
        self.generation_len = generation_len
        self.t = 0

        # Creating buttons and sidebar    
        self.btn_start = Button(
            dimensions = (15, 650, 158, 30),
            text = 'Start generation',
            shape_col = Color.GREY5,
            text_col = Color.GREY2,
            action = self.INITIALIZE)
        self.btn_load = Button(
            dimensions = (188, 750, 110, 30),
            text = 'Load genome',
            shape_col = Color.GREY5,
            text_col = Color.GREY2,
            action = self.RUN_FROM_GENOME)
        self.sidebar = SideBar(
            dimensions = (900, 0, 320, 700),
            background_color = Color.GREY5)

        self.interface = [self.btn_start, self.btn_load, self.sidebar]

    
    def update(self, i):
        """
        Update all program elements with one tick increment. Function gets 
        called from the Main class. Only calls to other functions contained
        here.
        """
        if self.state == self.INITIALIZE:  
            self.t = 0
            self.gen.current_gen = 0
            self.gen.create()
            self.gen.init_genomes()
            self.state = self.RUN_GENERATION
    
        if self.state == self.RUN_GENERATION:
            self.restock_food()
            self.gen.update(self.t, self.food)
            self.t += 1
            fitness_plot, bot_plot = self.gen.real_time_analysis(self.t)
            self.sidebar.insert_plot(fitness_plot, 0)
            self.sidebar.insert_plot(bot_plot, 1)
            if self.t == self.generation_len:
                self.state = self.END_GENERATION
            
        if self.state == self.END_GENERATION:
            self.gen.end_generation()
            self.state = self.ANALYZE_GENERATION
            
        if self.state == self.ANALYZE_GENERATION:
            plot = self.gen.analyze_generation()
            self.sidebar.insert_plot(plot, 2)
            self.state = self.EVOLVE_GENERATION
        
        if self.state == self.EVOLVE_GENERATION:
            new_genomes = self.gen.evolution()
            self.food = []
            self.gen.create()
            self.gen.init_genomes(new_genomes)
            self.t = 0
            self.state = self.RUN_GENERATION
        
        if self.state == self.RUN_FROM_GENOME:
            print('todo')
            self.state = self.HOLD
        
    def restock_food(self):
        """
        Restocks any eaten food at a random location
        """
        while len(self.food) < self.food_quantity:
            f = Food()
            f.normal_position(self.midpoint, spread=300)
            self.food.append(f)

    def draw(self, screen, pan_offset, zoom):
        """
        Draws all program elements on the screen.
        """
        if self.state == self.RUN_GENERATION:
            self.gen.individuals[0].autopilot.draw(screen, pan_offset, zoom) 
            for food in self.food:
                food.draw(screen, pan_offset, zoom)
            for bot in self.gen.individuals:
                bot.draw(screen, pan_offset, zoom)
        
        for element in self.interface:
            element.draw(screen)

    def information(self):
        """
        Returns a list of textual current generation information to display 
        on the screen.
        """
        information = []
        if self.state == self.RUN_GENERATION:
            information += self.gen.status_information(self.t)
        return information

    def mouse_left_click(self, pos, pan_offset):
        """
        Interpeted as a click at the location of start_pos.
        """
        for element in self.interface:
            if element.check_click(pos):
                self.state = element.action
                break
            
    def arrow_left(self):
        for bot in self.gen.individuals:
            bot.arrow_left()
            
    def arrow_right(self):
        for bot in self.gen.individuals:
            bot.arrow_right()

    def arrow_up(self):
        for bot in self.gen.individuals:
            bot.arrow_up()
    
    def arrow_down(self):
        for bot in self.gen.individuals:
            bot.arrow_down()




