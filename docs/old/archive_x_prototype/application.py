import pygame
import time
from Color import Color
import numpy as np
import Bots
import Tools
from math import pi
from Simulation import Simulation
from Generation import Generation



class Window:
    """
    Handles the main components of a simulation. Contains the main loop and 
    functionality for panning and drawing lines on the screen. All actual 
    simulation in handled in the Simulation object, to which pan_offset and
    drawn lines are passed.
    """
    
    def __init__(self, window_size, background_colour):
        self.window_size = np.array(window_size)
        self.font = pygame.font.SysFont('monospace', 15) 
        self.bg = background_colour
        self.screen = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pan_offset = np.array([0,0])
        self.zoom = 1
        self.running = False
        
    def main_loop(self, program):
        """
        Pygame main loop. Uses the ticker class to create consistent timespaces
        between ticks. In the main loop: Updates simulation, checks events,
        updats statistics on screen, draws screen en increments tick.
        """
        self.running = True
        ticker = Ticker(start_time=time.time(), tick_len=1/25)
        while self.running:  
            self.handle_events(program, ticker)
            self.update_screen(program, ticker)
        pygame.quit()
        
    def update_screen(self, program, ticker):
        """
        updates the program, clear the screen, and the redraws the screen. 
        Also iterates the tiker object by one tick and displays tick info.
        Any additional draw functions should be added here.
        """
        program.update(ticker.i)
        self.screen.fill(self.bg)
        program.draw(self.screen, self.pan_offset, self.zoom)
        information = ticker.string_stats() + program.information()
        self.display_textlist(information, Color.GREY2, 15, 5)
        pygame.display.flip()    
        ticker.next_tick()          

    def handle_events(self, program, ticker):
        """
        Loops over current events and calls functions associated with those
        events. Current events:
        """
        # Mouse trigger and quit events
        for event in pygame.event.get():    
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_left_click(program, ticker)
                if event.button == 2:
                    self.mouse_pan(program, ticker)
                if event.button == 4:
                    self.mouse_zoom_in(program)
                if event.button == 5:
                    self.mouse_zoom_out(program)
                    
        # Key hold events
        keys = pygame.key.get_pressed() 
        if keys[pygame.K_LEFT]:
            self.arrow_left(program)
        if keys[pygame.K_RIGHT]:
            self.arrow_right(program)
        if keys[pygame.K_UP]:
            self.arrow_up(program)
        if keys[pygame.K_DOWN]:
            self.arrow_down(program)
        
    def mouse_left_click(self, program, ticker):
        """
        Lets the user draw a line with their left mouse button. The line info
        is then passed on the the simulation which can in turn interpret it
        in different ways.
        """
        pos = pygame.mouse.get_pos()
        program.mouse_left_click(pos, self.pan_offset)
        
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

    def arrow_left(self, program):
        program.arrow_left()
    
    def arrow_right(self, program):
        program.arrow_right()

    def arrow_up(self, program):
        program.arrow_up()
    
    def arrow_down(self, program):
        program.arrow_down()



class Ticker:
    
    def __init__(self, start_time, tick_len):
        self.start_time = start_time
        self.tick_start = start_time
        self.tick_len = tick_len

        # Initializing Tick load statistics
        self.i = 0
        self.load_list = [0 for i in range(20)]    
        self.avg_load = 0
        self.max_load = 0
        self.min_load = 0
        self.run_time = 0
        self.iter_len = 0

    def next_tick(self):
        """
        Records the used computational time of tick. Then pauses the programm
        until the time until the next tick has elapsed.
        """
        t = time.time() - self.tick_start 
        self.update_statistics(t)
        time.sleep(max(0, self.tick_len - t))
        self.tick_start = time.time()
   
    def update_statistics(self, t):
        """
        Updates the tick statistics for the current tick.
        """
        load = 1 if self.tick_len == 0 else round(t / self.tick_len, 2)       
        self.load_list.pop(0)                    
        self.load_list.append(load)               
        self.i += 1                                
        self.avg_load = round(sum(self.load_list) / len(self.load_list),2)
        self.max_load = round(max(self.load_list),2)
        self.min_load = round(min(self.load_list),2)
        self.run_time = max(0.01, round(time.time() - self.start_time,2))
        self.iter_len = round(self.i / self.run_time, 2)
        
    def string_stats(self):
        """
        Returns all current stats values as a list of printeable strings
        """        
        stats_list = []
        stats_list.append('TICK AND LOAD STATISTICS')
        stats_list.append(f'average load:     {self.avg_load}')
        stats_list.append(f'maximum load:     {self.max_load}')
        stats_list.append(f'minimum load:     {self.min_load}')
        stats_list.append(f'runtime:          {self.run_time}')
        stats_list.append(f'tick length:      {self.iter_len}')
        stats_list.append(f'current tick:     {self.i}')
        return stats_list
        
        

# Pygame init
pygame.init() 
window_size = (1220, 700)
midpoint = (400,400)


# Creating generation
gen = Generation(
    pop_size = 35, 
    selection_size=10, 
    mutation_func = lambda pop_fitness: 1/(7*(pop_fitness+10)),
    midpoint = midpoint,
    bot_type = Bots.SimpleArrowBot)

# Creating simulation program
program = Simulation(
    window_size=window_size,
    midpoint = midpoint,
    generation=gen, 
    generation_len=400,
    food_quantity=400)

w = Window(window_size, Color.GREY7)
w.main_loop(program)
    





