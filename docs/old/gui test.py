import pygame
import time
import numpy as np
from zzpgplot import Color
from zzpgui.elements import Button, TextBox, Label
import tkinter
from tkinter import filedialog






class Environment:
    
    def __init__(self):
        """
        Class that imlements environments and returns a population and 
        simulation instance when called. Environment implementations inherit
        from this class. Each environment can have different parameters in
        the __call__ method that are defined in the Pygame GUI.

        Returns
        -------
        None.

        """
    
        self.parameters = []
    
    
    def select(self):
        pass
    
    def __call__(self):
        raise NotImplementedError








class HerbivoreEnvironment:
    
    
    
    
    
    def __call__(self):
        pass
    
    
def create_herbivore_environment(environment_size, nr_bots, generation_len,
                                  empty_genome=False):

    # Genetic algorithm
    genalg = GeneticAlgorithm(
        selection = selection.Tournament(k=3),
        crossover = crossover.Multipoint(1), 
        mutations = [
            mutation.UniformReplacement(
                lambda fitness: 0.02), 
            mutation.Adjustment(
                lambda fitness: 0.3, adjustment_domain=(-0.1,0.1))], 
        disaster = disaster.SuperMutation(
            similarity_threshold = 0.85,
            mutations = [mutation.Adjustment(
                lambda fitness: 1, adjustment_domain=(-0.3,0.3))]),
        num_elites = 3) 
    
    # Object sets and populations
    barr = Barriers(environment_size, quantity=8, size=180, wall_width=2)
    food = FoodSupply(environment_size, quantity=120)
    herbivore_pop = HerbivorePopulation(
        Herbivore, nr_bots, genalg, use_debug_bot=True)
    
    # Simulation instance
    sim = StandaloneSimulation(
        object_sets = {'food':food, 'barriers':barr},
        populations = {'herbivore': herbivore_pop},
        generation_len = generation_len,
        nr_generations = 100,
        nr_batches = 1,
        custom_genome = None,
        verbose = 1)
    
    # Clear genome fOr debug bot (used for debugging)
    if empty_genome:
        sim.custom_genome = herbivore_pop.bot((0,0),0).get_genome()

    return herbivore_pop, sim












class Ticker:
    
    def __init__(self, start_time, tick_len):
        self.start_time = start_time
        self.tick_start = start_time
        self.tick_len = tick_len
        
    def next_tick(self):
        t = time.time() - self.tick_start 
        time.sleep(max(0, self.tick_len - t))
        self.tick_start = time.time()



class Frame:
    
    def __init__(self, container):
        """
        Handles GUI and events. Inherited from this class to create frames
        with more specific behaviour.
        """
        self.container = container
        self.dim = container.dim
        self.font = pygame.font.SysFont('monospace', 12) 
        self.bg = Color.GREY1
        self.gui = self.initialize_gui()
        
    def draw(self, screen):
        """
        Called from container. Fills screen and calls window-specific drawing.
        """
        screen.fill(self.bg)
        self.draw_frame(screen)
        for element in self.gui.values():
            element.draw(screen)
        pygame.display.flip()  

    def update(self, events):
        """
        Called from container. Updates GUI and calls window-specific update.
        """
        for element in self.gui.values():
            element.update(events)
        self.update_frame()

    def initialize_gui(self):
        """
        Called from init. Implement gui in sub classes as a list under self.gui.
        """
        pass

    def mouse_press(self, key):
        """
        Called from container. Implement mouse button behaviour in sub clases.
        """
        pass
    
    def draw_frame(self, screen):
        """
        Called from self.draw. Implement drawing behaviour in sub classes.
        """
        raise NotImplementedError

    def update_frame(self):
        """
        called from self.update. Implement update behaviour in sub classes.
        """
        raise NotImplementedError



class OptionsWindow(Frame):
    
    def initialize_gui(self):
        x, y = 20,20
        cell_height = 20
        cell_width = 120
        y_pad1 = cell_height + 5
        x_pad1 = cell_width + 10
        gui = {
            'lbl_env':Label(
                'Environment', cell_width, cell_height),
            'txt_env':TextBox(
                '', cell_width, cell_height),
            'btn_env_select':Button(
                self.btn_env_select, 42, cell_height, text='Select'),
            'lbl_genalg':Label(
                'Genetic algorithm', cell_width, cell_height),
            'txt_genalg':TextBox(
                '', cell_width, cell_height),
            'btn_genalg_select':Button(
                self.btn_genalg_select, 42, cell_height, text='Select'),
            'lbl_genlen':Label(
                'Generation length', cell_width, cell_height),
            'txt_genlen':TextBox(
                '400', cell_width, cell_height, dtype=int),
            'lbl_gennum':Label(
                'Number of generations', cell_width, cell_height),
            'txt_gennum':TextBox(
                '100', cell_width, cell_height, dtype=int),
            'lbl_batchnum':Label(
                'Number of batches', cell_width, cell_height),
            'txt_batchnum':TextBox(
                '1', cell_width, cell_height, dtype=int),
            'btn_start':Button(
                self.btn_start, 50, 25, text='start'),
            }
       
        gui['lbl_env'].place(
            (x + x_pad1 * 0, y + y_pad1 * 0))
        gui['txt_env'].place(
            (x + x_pad1 * 1, y + y_pad1 * 0))
        gui['btn_env_select'].place(
            (x + x_pad1 * 2, y + y_pad1 * 0))
        gui['lbl_genalg'].place(
            (x + x_pad1 * 0, y + y_pad1 * 1))
        gui['txt_genalg'].place(
            (x + x_pad1 * 1, y + y_pad1 * 1))
        gui['btn_genalg_select'].place(
            (x + x_pad1 * 2, y + y_pad1 * 1))
        gui['lbl_genlen'].place(
            (x + x_pad1 * 0, y + y_pad1 * 2))
        gui['txt_genlen'].place(
            (x + x_pad1 * 1, y + y_pad1 * 2))
        gui['lbl_gennum'].place(
            (x + x_pad1 * 0, y + y_pad1 * 3))
        gui['txt_gennum'].place(
            (x + x_pad1 * 1, y + y_pad1 * 3))
        gui['lbl_batchnum'].place(
            (x + x_pad1 * 0, y + y_pad1 * 4))
        gui['txt_batchnum'].place(
            (x + x_pad1 * 1, y + y_pad1 * 4))
        gui['btn_start'].place(
            (self.dim[0]-55, self.dim[1]-30))
        return gui

    def btn_start(self):
        self.container.active_frame = self.container.frames['simulation']

    def btn_env_select(self):
        print('select env')
        folder_path = filedialog.askopenfile(
            title='test', filetypes=[('Environment files', 'json')], 
            initialdir='environments')

        
    def btn_genalg_select(self):
        print('select genalg')

    def mouse_press(self, key):
        pass
    
    def draw_frame(self, screen):
        pass

    def update_frame(self):
        pass
    
        
    
class SimulationWindow(Frame):
    
    def initialize_gui(self):
        gui = {
            'btn_back':Button(self.goto_options, 50, 25, text='Back'),
            }
        
        gui['btn_back'].place((5, self.dim[1]-30))
        return gui

    def goto_options(self):
        self.container.active_frame = self.container.frames['options']

    def mouse_press(self, key):
        pass
    
    def draw_frame(self, screen):
        pass

    def update_frame(self):
        pass
    



class AppContainer:
    
    def __init__(self, dim):
        pygame.init() 
        tkinter.Tk().withdraw()
        self.dim = np.array(dim)
        self.ticker = Ticker(start_time=time.time(), tick_len=1/30)
        self.screen = pygame.display.set_mode(dim)
        self.frames = {
            'options':OptionsWindow(self),
            'simulation':SimulationWindow(self)}
        self.active_frame = self.frames['options']
        self.running = True

    def main_loop(self):
        while self.running:
            self.ticker.next_tick() 
            events = self.handle_events()
            self.active_frame.update(events)
            self.active_frame.draw(self.screen)
        pygame.quit()
        
    def handle_events(self):
        events = pygame.event.get()
        for event in events:    
            if event.type == pygame.QUIT:
                self.running = False
        return events
                
                
        
   



if __name__ == '__main__':
    
    app = AppContainer((400,400))
    app.main_loop()