import pygame
import time
import numpy as np
from interface import Button, SideBar, Color
from bots import Herbivore, HerbivorePopulation
from objects import FoodSupply, Barrier
from matplotlib import pyplot as plt
import warnings



"""
TODO, BOTS:
    - https://geoffboeing.com/2016/10/r-tree-spatial-index-python/
    - Starvation with bot coloring based on its level
    - Allowed starting area based on barriers
    - Pass genetic alg as parameter to population with set_fitness and set_genomes
    as method.
    - Streamlining custom genome and debug bot
    - Create seperate simulation mode for testing custom genomes
    
TODO, BARRIERS:
    - Safe area function with offset that can be used for food and bot spawning
    positions:
    
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
    START = 3
    RUN = 4
    END = 5
    EVOLVE = 6
    ANALYZE = 7
    LOAD = 8
    EXIT = 9
    
    def __init__(self, window_size, object_sets, populations):
        
        # Application parameters
        self.window_size = np.array(window_size)
        self.ticker = Ticker(start_time=time.time(), tick_len=1/25)
        self.font = pygame.font.SysFont('monospace', 15) 
        self.bg = Color.GREY7
        self.screen = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pan_offset = np.array([0,0])
        self.state = self.HOLD
        self.zoom = 1
              
        # Object sets, populations and generation len
        self.object_sets = object_sets
        self.populations = populations
        
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
        
    def main_loop(self, generation_len, nr_batches, custom_genome=None):
        """
        Pygame main loop. Uses the ticker class to create consistent timespaces
        between ticks. In the main loop: Updates simulation, checks events,
        updates statistics on screen, draws screen en increments tick. The
        custom_genome parameter allows a genome to be passed to the debug bot
        of each generation
        """
        current_batch = 0
        current_gen = 0
        stats_quants = []
        
        while True:  
            
            self.ticker.next_tick() 
            self.handle_events()
            self.draw()
            
            # Hold: wait for user input to start simulation
            if self.state == self.HOLD:
                pass
            
            # Initialize: only triggered once per simulation. 
            elif self.state == self.INITIALIZE:
                self.state = self.START
            
            # Start: triggered at the start of each new simulation run.
            elif self.state == self.START:
                t = 0
                print(f'> G{current_gen}B{current_batch}')
                
                for objset in self.object_sets.values():
                    objset.initialize()
                
                for pop in self.populations.values():
                
                    # Initialize random first generation
                    if current_gen == 0 and current_batch == 0:
                        pop.initialize(self.object_sets)
                        if not custom_genome is None and pop.use_debug_bot:
                            pop.debug_bot.set_genome(custom_genome)
                
                    # Evolve generation at first batch and nonzero generation
                    elif current_gen > 0 and current_batch == 0:
                        new_genomes = pop.evolve()
                        pop.initialize(self.object_sets)
                        pop.set_genomes(new_genomes)
                    
                    # Always call the next batch
                    pop.next_batch(nr_batches)
                
                self.state = self.RUN
               
            # Run: triggered each tick during the simulation sequence.
            elif self.state == self.RUN:
                t += 1
                for pop in self.populations.values():
                    pop.update(self.object_sets, self.populations)
                for objset in self.object_sets.values():
                    objset.update(self.object_sets, self.populations)
                if t == generation_len:
                    self.state = self.END
            
            # End: start new batch or trigger generation end
            elif self.state == self.END:
                current_batch += 1
                if current_batch < nr_batches:
                    self.state = self.START
                elif current_batch == nr_batches:
                    current_gen += 1
                    self.state = self.ANALYZE
            
            # Analyze: same as end, but all analyses are done here.
            elif self.state == self.ANALYZE:
                    
                # Quantiles plot
                fitness = pop[0].get_fitness()
                quants = np.arange(0.1, 1.1, 0.1)
                quant_vals = np.quantile(fitness, quants)
                stats_quants.append(quant_vals)
                if len(stats_quants) > 2:
                    plt.figure(figsize=(8,5))
                    for q in np.array(stats_quants).T:
                        plt.plot(q, color='black', alpha=0.4)
                    plt.show()
                
                self.state = self.START

            # Exit: triggered at program end: when the user closes the window.
            elif self.state == self.EXIT:
                break

        pygame.quit()
        
    def draw(self):
        """
        Draw all simulation elements to the screen.
        """
        self.screen.fill(self.bg)
        for pop in self.populations.values():
            pop.draw(self.screen, self.pan_offset, self.zoom)
        for objset in self.object_sets.values():
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
                    
        # Key hold events: get passed to first population
        keys = pygame.key.get_pressed() 
        pop = self.populations.values()[0]
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
                self.state = btn.action
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



class standalone_simulation:
    
    HOLD = 0
    INITIALIZE = 1
    START = 3
    RUN = 4
    END = 5
    EVOLVE = 6
    ANALYZE = 7
    LOAD = 8
    EXIT = 9
    
    def __init__(self, object_sets, populations, generation_len, 
                 nr_generations, nr_batches, custom_genome=None):
        
        # Parameters
        self.object_sets = object_sets
        self.populations = populations
        self.generation_len = generation_len
        self.nr_generations = nr_generations
        self.nr_batches = nr_batches
        self.custom_genome = custom_genome

        # State and timing variables
        self.state = self.HOLD
        self.t = 0
        self.cur_gen = 0
        self.cur_batch = 0
   
    def update(self):
        
        stats_quants = []
    
    
    
    def state_hold(self):
        """
        Passive state that waits for the simulation to be started.
        """
        
    def state_initialize(self):
        """
        Only triggered once per simulation. Any setup actions are applied here.
        """
        self.cur_gen = 0
        self.cur_batch = 0
        self.state = self.START
    
    def state_start(self):
        """
        Triggered at the start of each new simulation run. Creates a random
        population for the first generation, and then either calls next 
        batches or evolves the population. Also resets objects.
        """
        self.t = 0
        print(f'> G{self.cur_gen}B{self.cur_batch}')
        for objset in self.object_sets.values():
            objset.initialize()

        for pop in self.populations.values():
            if self.cur_gen == 0 and self.cur_batch == 0:
                pop.initialize(self.object_sets)
                if not self.custom_genome is None and pop.use_debug_bot:
                    pop.debug_bot.set_genome(self.custom_genome)
            elif self.cur_gen > 0 and self.cur_batch == 0:
                new_genomes = pop.evolve()
                pop.initialize(self.object_sets)
                pop.set_genomes(new_genomes)
            pop.next_batch(self.nr_batches)
        
        self.state = self.RUN
        
        
    def state_run(self):
        """
        Updates all populations and objects in the simulation.
        """
        
        
    def state_analyze(self):
        """
        TODO: move to analyze class higher up that holds standalone_sim as a 
        attribute
        """
        
    def state_end(self):
        """
        Triggered at the end of each simulation run. 
        """
        
    def state_exit(self):
        """
        Triggered when the last generation is completed.
        """
        
        
    
        while True:  
            
            # Hold: wait for user input to start simulation
            if self.state == self.HOLD:
                pass
            
            # Initialize: 
            elif self.state == self.INITIALIZE:
                self.state = self.START
            
            # Start: triggered at the start of each new simulation run.
            elif self.state == self.START:
                t = 0
                print(f'> G{current_gen}B{current_batch}')
                
                for objset in self.object_sets.values():
                    objset.initialize()
                
                for pop in self.populations.values():
                
                    # Initialize random first generation
                    if current_gen == 0 and current_batch == 0:
                        pop.initialize(self.object_sets)
                        if not custom_genome is None and pop.use_debug_bot:
                            pop.debug_bot.set_genome(custom_genome)
                
                    # Evolve generation at first batch and nonzero generation
                    elif current_gen > 0 and current_batch == 0:
                        new_genomes = pop.evolve()
                        pop.initialize(self.object_sets)
                        pop.set_genomes(new_genomes)
                    
                    # Always call the next batch
                    pop.next_batch(nr_batches)
                
                self.state = self.RUN
               
            # Run: triggered each tick during the simulation sequence.
            elif self.state == self.RUN:
                t += 1
                for pop in self.populations.values():
                    pop.update(self.object_sets, self.populations)
                for objset in self.object_sets.values():
                    objset.update(self.object_sets, self.populations)
                if t == generation_len:
                    self.state = self.END
            
            # End: start new batch or trigger generation end
            elif self.state == self.END:
                current_batch += 1
                if current_batch < nr_batches:
                    self.state = self.START
                elif current_batch == nr_batches:
                    current_gen += 1
                    self.state = self.ANALYZE
            
            # Analyze: same as end, but all analyses are done here.
            elif self.state == self.ANALYZE:
                    
                # Quantiles plot
                fitness = pop[0].get_fitness()
                quants = np.arange(0.1, 1.1, 0.1)
                quant_vals = np.quantile(fitness, quants)
                stats_quants.append(quant_vals)
                if len(stats_quants) > 2:
                    plt.figure(figsize=(8,5))
                    for q in np.array(stats_quants).T:
                        plt.plot(q, color='black', alpha=0.4)
                    plt.show()
                
                self.state = self.START

            # Exit: triggered at program end: when the user closes the window.
            elif self.state == self.EXIT:
                break





class standalone_simulation2:
    
    def __init__(self, object_sets, populations):
        self.object_sets = object_sets
        self.populations = populations
        
    def run(self, generation_len, nr_generations, nr_batches):
        generation_quants = []
        generation_time = []
        best_individuals = []
        
        # Initialize first populations
        for pop in self.populations.values():
            pop.initialize()
        
        # Loop over generations and batches
        for current_gen in range(nr_generations):
            t_start = time.time()
            for current_batch in range(nr_batches):
                print(f'> G{current_gen}B{current_batch}')
                
                # Initializing batch and objects
                for pop in self.populations.values():
                    pop.next_batch(nr_batches)
                for objset in self.object_sets.values():
                    objset.initialize()
                    
                # Updating all individuals and objects
                for t in range(generation_len):
                    for pop in self.populations.values():
                        pop.update(self.object_sets, self.populations)
                    for objset in self.object_sets.values():
                        objset.update(self.object_sets, self.populations)
            
            # Run analysis on the first population
            for pop in self.populations.values():
                
                # Computing fitness and quantiles
                fitness = np.array([ind.fitness for ind in pop.individuals])
                quants = np.arange(0.2, 1.2, 0.2)
                quant_vals = np.quantile(fitness, quants)
                generation_quants.append(quant_vals)
                
                # Saving best fitness genome
                best_individuals.append(pop.individuals[np.argmax(fitness)])
                
                # Fitness quantiles plot
                plt.figure(figsize=(8,5))
                for q in np.array(generation_quants).T:
                    plt.plot(q, color='black', alpha=0.35)
                plt.show()
    
                # Evolution and printing stats
                pop.evolve()
                print(f'\nGeneration {current_gen} results:')
                print(f'    fitness: {round(fitness.mean(),3)}')
                print(f'    Mutation prob: {pop.mutation_func(fitness.mean())}')
    
                # Saving best genome
                recent_best = best_individuals[-10:]
                fitness = [ind.fitness for ind in recent_best]
                genomes = [ind.get_genome() for ind in recent_best]
                best_genome = genomes[np.argmax(fitness)]
                str_genome = '\n'.join(best_genome.astype(str))
                filename = f'{pop.__class__.__name__}_bestgenome.txt'
                with open(filename, 'w') as file:
                    file.writelines(str_genome)
    
    
            # Recording time
            generation_time.append(time.time() - t_start)
            t_avg = round(np.mean(generation_time), 2)
            minutes_to_end = round((t_avg * (nr_generations - current_gen))/60,2)
            print(f'    Averige seconds/gen: {t_avg}')
            print(f'    Minutes left: {minutes_to_end}\n')
        
    


if __name__ == '__main__':

    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    
    STANDALONE = 0
    VISUAL = 1
    VISUAL_DEBUG = 2
    
    runtype = VISUAL
    
    
    
    if runtype == STANDALONE:
        window_size = np.array((600,500))
        food = FoodSupply(60, window_size)
        barriers = Barrier(
            window_size, nr_barriers=12, barrier_size=15, wall_width=2)
        herbivore = HerbivorePopulation(Herbivore, 80)
        sim = standalone_simulation(
            object_sets = {'food':food, 'barriers':barriers},
            populations = {'herbivore':herbivore})
        sim.run(300, 50, 4)
        
    elif runtype == VISUAL:
        pygame.init() 
        window_size = np.array((600,500))
        food = FoodSupply(60, window_size)
        barriers = Barrier(
            window_size, nr_barriers=12, barrier_size=15, wall_width=2)
        herbivore = HerbivorePopulation(Herbivore, 80, True)
        app = Application(
            window_size, 
            object_sets = {'food':food, 'barriers':barriers},
            populations = {'herbivore':herbivore})
        app.main_loop(generation_len=300, nr_batches=4)
        
    elif runtype == VISUAL_DEBUG:
        pygame.init() 
        window_size = np.array((300,300))
        food = FoodSupply(40, window_size)
        barriers = Barrier(
            window_size, nr_barriers=2, barrier_size=15, wall_width=10)
        herbivore = HerbivorePopulation(Herbivore, 1, True)
        app = Application(
            window_size, 
            object_sets = {'food':food, 'barriers':barriers},
            populations = {'herbivore':herbivore})
        
        with open('HerbivorePopulation_bestgenome.txt.', 'r') as file:
            str_genome = file.readlines()
            genome = [float(gen.replace('\n', '')) for gen in str_genome]
            best_genome = np.array(genome)
        
        
        best_genome = np.zeros(len(best_genome))
        
        app.main_loop(
            generation_len=10000, nr_batches=1, custom_genome=best_genome)
    
