import pygame
import time
import numpy as np
from interface import Button, SideBar
from pygame_plot import Color




class Application:
    """
    Handles the main components of a simulation. Contains the main loop and 
    functionality for panning and drawing lines on the screen. All actual 
    simulation in handled in the Simulation object, to which pan_offset and
    drawn lines are passed.
    """
    
    def __init__(self, window_size, simulation, analysis):
        
        pygame.init() 
        
        # Application parameters
        self.window_size = np.array(window_size)
        self.ticker = Ticker(start_time=time.time(), tick_len=1/25)
        self.font = pygame.font.SysFont('monospace', 15) 
        self.bg = Color.GREY7
        self.screen = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pan_offset = np.array([0,0])
        self.window_open = True
        self.zoom = 1
        self.max_speed = False
              
        # Simulations and analysis instances
        self.simulation = simulation
        self.analysis = analysis
        
        # Interface    
        self.btn_start = Button(
            dimensions = (10, window_size[1]-40, 55, 30),
            text = 'Start',
            shape_col = Color.GREY5,
            text_col = Color.GREY2,
            action = self.start)
        self.btn_load = Button(
            dimensions = (75, window_size[1]-40, 55, 30),
            text = 'Load',
            shape_col = Color.GREY5,
            text_col = Color.GREY2,
            action = lambda: None)
        self.btn_speed = Button(
            dimensions = (140, window_size[1]-40, 55, 30),
            text = 'Speed',
            shape_col = Color.GREY5,
            text_col = Color.GREY2,
            action = self.speed_up)
        self.buttons = [self.btn_start, self.btn_load, self.btn_speed]
        self.sidebar = SideBar(
            dimensions = (window_size[0]-270, 0, 270,  window_size[1]),
            background_color = Color.GREY5,
            nr_figures=4)
        
    def start(self):
        """
        Called from on-screen button. Starts the simulation.
        """
        self.simulation.state = self.simulation.INITIALIZE
        
    def speed_up(self):
        """
        Called from on-screen button. Toggles full speed ticks.
        """
        if self.max_speed:
            self.max_speed = False
            self.ticker.tick_len = 1/25
        else:
            self.max_speed = True
            self.ticker.tick_len = 1/1000
        
    def main_loop(self):
        """
        Pygame main loop. Uses the ticker class to create consistent timespaces
        between ticks. In the main loop: Updates simulation, checks events,
        updates statistics on screen, draws screen en increments tick.
        """
        self.analysis.set_pygame_plot_dimensions(self.sidebar)
        while self.window_open:
            self.ticker.next_tick() 
            self.handle_events()
            self.simulation.update()
            self.analysis.update(self.sidebar)
            self.draw()
        pygame.quit()
        
    def draw(self):
        """
        Draw all simulation elements to the screen.
        """
        self.screen.fill(self.bg)
        for pop in self.simulation.populations.values():
            pop.draw(self.screen, self.pan_offset, self.zoom)
        for objset in self.simulation.object_sets.values():
            objset.draw(self.screen, self.pan_offset, self.zoom)
        for btn in self.buttons:
            btn.draw(self.screen)
        self.display_textlist(self.ticker.get_stats(), Color.GREY2, 15, 5)
        self.sidebar.draw(self.screen)
        pygame.display.flip()  
          
    def handle_events(self):
        """
        Loops over current events and calls the associated methods.
        """
        for event in pygame.event.get():    
            if event.type == pygame.QUIT:
                self.window_open = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_left_click()
                if event.button == 2:
                    self.mouse_pan()
                if event.button == 4:
                    self.mouse_zoom_in()
                if event.button == 5:
                    self.mouse_zoom_out()
                    
        # Key hold events: get passed to first population
        keys = pygame.key.get_pressed() 
        pop = list(self.simulation.populations.values())[0]
        if keys[pygame.K_LEFT]:
            pop.arrow_key_left()
        if keys[pygame.K_RIGHT]:
            pop.arrow_key_right()
        if keys[pygame.K_UP]:
            pop.arrow_key_up()
        if keys[pygame.K_DOWN]:
            pop.arrow_key_down()
                    
    def mouse_left_click(self):
        """
        Loops over all interface element to check for a match with the mouse
        click position, and executes the associated action.
        """
        pos = pygame.mouse.get_pos()
        self.sidebar.check_click(pos)
        for btn in self.buttons:
            if btn.check_click(pos):
                btn.action()
                break
        
    def mouse_pan(self):
        """
        Pans the program screen using the middle mouse button.
        """
        initial_offset = self.pan_offset.copy() 
        mouse_start = np.array(pygame.mouse.get_pos())
        while True:
            mouse_current = np.array(pygame.mouse.get_pos())
            self.pan_offset = initial_offset + mouse_current - mouse_start
            self.draw()
            for event in pygame.event.get():    
                if event.type == pygame.MOUSEBUTTONUP:
                    return
    
    def mouse_zoom_in(self):
        """
        Calls self.change_zoom with positive increment.
        """
        self.change_zoom(increment=0.1)

    def mouse_zoom_out(self):
        """
        Calls self.change_zoom with negative increment.
        """
        self.change_zoom(increment=-0.1)
        
    def change_zoom(self, increment):
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
        self.hist = np.zeros(20)
        
    def next_tick(self):
        """
        Records the used computational time of tick. Then pauses the programm
        until the time until the next tick has elapsed.
        """
        t = time.time() - self.tick_start 
        time.sleep(max(0, self.tick_len - t))
        self.i += 1
        self.tick_start = time.time()
        self.update_stats(t)
   
    def update_stats(self, t):
        """
        Record the last 20 tick durations.
        """
        self.hist = np.roll(self.hist, 1)
        self.hist[0] = t / self.tick_len
    
    def get_stats(self):
        """
        Returns all current stats values as a list of printeable strings
        """        
        stats_list = []
        stats_list.append('TICK AND LOAD STATISTICS')
        stats_list.append(f'average load:     {self.hist.mean().round(2)}')
        stats_list.append(f'maximum load:     {self.hist.max().round(2)}')
        stats_list.append(f'minimum load:     {self.hist.min().round(2)}')
        stats_list.append(f'current tick:     {self.i}')
        return stats_list



class StandaloneSimulation:
    
    HOLD = 0
    INITIALIZE = 1
    START = 2
    RUN = 3
    BATCH_END = 4
    GEN_END = 5
    EXIT = 6
    
    def __init__(self, object_sets, populations, generation_len, 
                 nr_generations, nr_batches, custom_genome=None, verbose=0):
        
        # Parameters
        self.object_sets = object_sets
        self.populations = populations
        self.generation_len = generation_len
        self.nr_generations = nr_generations
        self.nr_batches = nr_batches
        self.custom_genome = custom_genome
        self.verbose = verbose

        # State and timing variables
        self.state = self.HOLD
        
        # Time variables
        self.t = 0
        self.cur_gen = 0
        self.cur_batch = 0
        
        # Statistics
        self.prev_state = self.state
        self.stats_gentime = 0
        self.stats_gentime_log = []
   
    def update(self):
        
        new_state = self.prev_state != self.state
        if new_state and self.verbose >= 2:
            print(f'> STATE {self.state}')
        self.prev_state = self.state
        
        if self.state == self.HOLD:
            self.state_hold()
        elif self.state == self.INITIALIZE:
            self.state_initialize()
        elif self.state == self.START:
            self.state_start()
        elif self.state == self.RUN:
            self.state_run()
        elif self.state == self.BATCH_END:
            self.state_batch_end()
        elif self.state == self.GEN_END:
            self.state_gen_end()
        elif self.state == self.EXIT:
            self.state_exit()
    
    def state_hold(self):
        """
        Passive state that waits for the simulation to be started.
        """
        pass
        
    def state_initialize(self):
        """
        Only triggered once per simulation. Any setup actions are applied here.
        """
        self.t = 0
        self.cur_gen = 0
        self.cur_batch = 0
        self.stats_gentime = 0
        self.stats_gentime_log = []
        self.state = self.START
    
    def state_start(self):
        """
        Triggered at the start of each new simulation run. Creates a random
        population for the first generation, and then either calls next 
        batches or evolves the population. Also resets objects.
        """
        self.t = 0
        
        # Verbsose 1 debug
        if self.verbose >= 1:
            print(f'> G{self.cur_gen}B{self.cur_batch}')
        
        # Initialize objects
        for objset in self.object_sets.values():
            objset.initialize()
        
        # Initialize populations
        for pop in self.populations.values():
            
            # Random population for first generation
            if self.cur_gen == 0 and self.cur_batch == 0:
                pop.initialize(self.object_sets)
                self.stats_gentime = time.time()
                if not self.custom_genome is None:
                    pop.individuals[0].set_genome(self.custom_genome)

            # Evolve population for first batch of each generation
            elif self.cur_gen > 0 and self.cur_batch == 0:
                new_genomes = pop.evolve()
                pop.initialize(self.object_sets)
                pop.set_genomes(new_genomes)
                self.stats_gentime = time.time()
            pop.next_batch(self.nr_batches)
        
        self.state = self.RUN
        
    def state_run(self):
        """
        Updates all populations and objects in the simulation.
        """
        self.t += 1
        for pop in self.populations.values():
            pop.update(self.object_sets, self.populations)
        for objset in self.object_sets.values():
            objset.update(self.object_sets, self.populations)
        if self.t == self.generation_len:
            self.state = self.BATCH_END
        
    def state_batch_end(self):
        """
        Triggered at the end of each simulation run. Determines whether a new
        batch or a new generation should be initialized.
        """
        self.cur_batch += 1
        if self.cur_batch < self.nr_batches:
            self.state = self.START
        elif self.cur_batch == self.nr_batches:
            if self.cur_gen == self.nr_generations - 1:
                self.state = self.EXIT
            else:
                self.cur_gen += 1
                self.cur_batch = 0
                self.state = self.GEN_END
        
    def state_gen_end(self):
        """
        End the generation and print generation statistics.
        """
        generation_time =  time.time() - self.stats_gentime
        self.stats_gentime_log.append(generation_time)
        sec_gen = round(np.mean(self.stats_gentime_log),2)
        min_left = round((sec_gen * (self.nr_generations - self.cur_gen))/60, 2)
        print(f'> G{self.cur_gen}/{self.nr_generations} completed ({sec_gen} sec/gen, {min_left} min remaining)')
        self.state = self.START
                    
    def state_exit(self):
        """
        Triggered when the last generation is completed.
        """
        print('simulation comleted.')
        self.state = self.HOLD






