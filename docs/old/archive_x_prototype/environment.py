import pygame
import time
from Color import Color
import numpy as np
import Tools
from math import pi
from Interface import Button, SideBar


"""
NEW HIERACRHY:
    - Window
        - Environment (passed to window)
            - > 1 Populations (passsed to Environment)
                - n Bots (passed to Population)
                    - Autopilot
                    - Sensors
            - > 1 Object_set 
"""

class Application:
    """
    Handles the main components of a simulation. Contains the main loop and 
    functionality for panning and drawing lines on the screen. All actual 
    simulation in handled in the Simulation object, to which pan_offset and
    drawn lines are passed.
    """
    
    HOLD = 0
    INITIALIZE = 1
    START = 2
    RUN = 3
    END = 4
    EVOLVE = 5
    ANALYZE = 6
    LOAD = 7
    EXIT = 8
    
    def __init__(self, window_size):
        
        # Application parameters
        self.window_size = np.array(window_size)
        self.ticker = Ticker(start_time=time.time(), tick_len=1/25)
        self.font = pygame.font.SysFont('monospace', 15) 
        self.bg = Color.GREY7
        self.screen = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pan_offset = np.array([0,0])
        self.state = self.HOLD
               
        # Interface    
        self.btn_start = Button(
            dimensions = (10, window_size[1]-40, 92, 30),
            text = 'Start sim',
            shape_col = Color.GREY5,
            text_col = Color.GREY2,
            action = self.INITIALIZE)
        self.btn_load = Button(
            dimensions = (112, window_size[1]-40, 110, 30),
            text = 'Load genome',
            shape_col = Color.GREY5,
            text_col = Color.GREY2,
            action = self.LOAD)
        self.buttons = [self.btn_start, self.btn_load]
        self.sidebar = SideBar(
            dimensions = (window_size[0]-200, 0, 200,  window_size[1]),
            background_color = Color.GREY5,
            nr_figures=4)
        
    def main_loop(self, generation_len):
        """
        Pygame main loop. Uses the ticker class to create consistent timespaces
        between ticks. In the main loop: Updates simulation, checks events,
        updats statistics on screen, draws screen en increments tick.
        """
        while True:  
            
            self.ticker.next_tick()     
            self.handle_events()
            self.draw()
            
            if self.state == self.HOLD:
                pass
            
            elif self.state == self.INITIALIZE:
                pass
            
            elif self.state == self.START:
                pass
            
            elif self.state == self.RUN:
                pass 
            
            elif self.state == self.END:
                pass
            
            elif self.state == self.EVOLVE:
                pass
         
            elif self.state == self.ANALYZE:
                pass

            elif self.state == self.EXIT:
                break

        pygame.quit()
        
    def draw(self):
        self.screen.fill(self.bg)
        self.environment.draw(self.screen, self.pan_offset)
        for btn in self.buttons:
            btn.draw(self.screen)
        self.sidebar.draw(self.screen)
        information = []
        self.display_textlist(information, Color.GREY2, 15, 5)
        pygame.display.flip()    

    def handle_events(self):
        """
        Loops over current events and calls the associated methods.
        """
        for event in pygame.event.get():    
            if event.type == pygame.QUIT:
                self.state = self.EXIT
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_left_click()
                if event.button == 2:
                    self.mouse_pan()
                if event.button == 4:
                    self.mouse_zoom_in()
                if event.button == 5:
                    self.mouse_zoom_out()
                    
    def mouse_left_click(self):
        """
        Loops over all interface element to check for a match with the mouse
        click position, and executes the associated action.
        """
        pos = pygame.mouse.get_pos()
        self.sidebar.check_click(pos)
        for btn in self.buttons:
            if btn.check_click(pos):
                self.state = btn.action
                break
        
    def mouse_pan(self, program, ticker):
        """
        Pans the program screen using the middle mouse button.
        """
        initial_offset = self.pan_offset.copy() 
        mouse_start = np.array(pygame.mouse.get_pos())
        while True:
            mouse_current = np.array(pygame.mouse.get_pos())
            self.pan_offset = initial_offset + mouse_current - mouse_start
            self.update_screen(program, ticker)
            for event in pygame.event.get():    
                if event.type == pygame.MOUSEBUTTONUP:
                    return
    
    def mouse_zoom_in(self, program):
        """
        Calls self.change_zoom with positive increment.
        """
        self.change_zoom(program, increment=0.1)

    def mouse_zoom_out(self, program):
        """
        Calls self.change_zoom with negative increment.
        """
        self.change_zoom(program, increment=-0.1)
        
    def change_zoom(self, program, increment):
        """
        Zooms in or out and adjusts self.pan offset to allign the screen.
        """
        prev_zoom = self.zoom
        self.zoom = max(self.zoom + increment, 0.2)
        midpoint = self.window_size / 2
        delta = midpoint - (self.zoom / prev_zoom) * midpoint
        self.pan_offset = self.pan_offset + delta
    
    def display_text(self, text, color, x, y):
        """
        Draws one line of text on specified coordinates.
        """
        textblock = self.font.render(text, True, color)
        self.screen.blit(textblock, (x,y))
    
    def display_textlist(self, text_list, color, x, y):
        """
        Draws a list of text on specified coordinates.
        """
        for line in text_list:  
            self.display_text(line, color, x, y)
            y += 15


class Ticker:
    
    def __init__(self, start_time, tick_len):
        self.start_time = start_time
        self.tick_start = start_time
        self.tick_len = tick_len
        self.i = 0

    def next_tick(self):
        """
        Records the used computational time of tick. Then pauses the programm
        until the time until the next tick has elapsed.
        """
        t = time.time() - self.tick_start 
        time.sleep(max(0, self.tick_len - t))
        self.tick_start = time.time()
   


    
    
class HerbivorePopulation:
    """
    Population of bots that eat food objects.
    """
    
    def __init__(self, pop_size, midpoint):
        self.pop_size = pop_size
        self.midpoint = midpoint
        self.bots = []
    
    
    def initialize(self):
        """
        Creates a new population of bots with random positions and genomes.
        """
        pass
    
    def evolve(self):
        """
        Creates a new population by evolving the current population.
        """
        pass
    
    def update(self, objects, populations):
        """
        Creates a distance matrix between all herbive bots and food objects,
        and calls bot updates.
        """
        pass
        


class FoodSupply:
    
    def __init__(self, quantiy, midpoint, spread):
        pass

    def initialize(self):
        """
        Places food objects at random locations.
        """
        pass
    
    def update(self, populations, object_sets):
        """
        Restocks the food supply.
        """
        pass


def standalone_simulation(
        object_sets, populations, midpoint, generation_len, nr_generations):
    
    # Initialize populations and objects
    for pop in populations:
        pop.initialize()
    for objset in object_sets:
        objset.initialize()
    
    # Run generations
    for gen in range(nr_generations):
        
        for t in range(generation_len):
            
            # Update all bots and objects
            for pop in populations:
                pop.update(populations, object_sets)
            for objset in object_sets:
                objset.update(populations, object_sets)
        
        # Run evolution
        for pop in populations:
            pop.evolve()




if __name__ == '__main__':
    
    pygame.init() 
    window_size = np.array((500,400))
    midpoint = (window_size/2).astype(int)
    
    food = FoodSupply(50, midpoint, 20)
    pop1 = SimpleHerbivorePopulation(80, midpoint)
    
    
    standalone_simulation(
        object_sets = [food],
        populations = [pop1],
        midpoint = midpoint,
        generation_len = 300,
        nr_generations = 20
        )
    
    """
    app = Application(window_size, env)
    app.main_loop()
    """
        
        
        
    
    
    
    