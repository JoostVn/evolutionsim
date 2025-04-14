import pygame
import time
import numpy as np
from interface import SideBar
from zzpgplot import Color
from zzpgui.base import Application


class SimApplication(Application):
    
    # TODO: arrow control of debug bot
    
    def __init__(self, window_size, simulation, analysis):
        super().__init__(
            window_size, tick_len=1/30, name='Evolution Learning', 
            theme_name='default_dark')
        
        pygame.init() 
        
        # Application parameters
        self.font = pygame.font.SysFont('monospace', 12) 
              
        # Simulations instance
        self.simulation = simulation
        
        # Analysis and sidebar
        # TODO: move to GUI        

        self.analysis = analysis
        self.sidebar = SideBar(
            dimensions = (window_size[0]-270, 0, 270,  window_size[1]),
            background_color = Color.GREY3,
            nr_figures=4)
        self.analysis.set_pygame_plot_dimensions(self.sidebar)
        
        # Starting simulation
        self.simulation.state = self.simulation.INITIALIZE
        
        
    def update(self):        
        self.simulation.update()
        
        
        # TODO: move to gui
        self.analysis.update(self.sidebar)
        if pygame.BUTTON_LEFT in self.key_events['down']:
            pos = pygame.mouse.get_pos()
            self.sidebar.check_click(pos)
            print(pos)

        
    def draw(self):
        """
        Draw all simulation elements to the screen.
        """
        for pop in self.simulation.populations.values():
            pop.draw(self.screen, self.pan_offset, self.zoom)
        for objset in self.simulation.object_sets.values():
            objset.draw(self.screen, self.pan_offset, self.zoom)

        # TODO: move to GUI
        self.sidebar.draw(self.screen)

    
          






class ApplicationOld:
    """
    Handles the main components of a application. Contains the main loop and 
    functionality for panning and zooming. All actual simulation in handled
    in the Simulation object.
    """
    
    def __init__(self, window_size, simulation, analysis):
        
        pygame.init() 
        
        # Application parameters
        self.window_size = np.array(window_size)
        self.ticker = Ticker(start_time=time.time(), tick_len=1/30)
        self.font = pygame.font.SysFont('monospace', 12) 
        self.bg = Color.GREY1
        self.screen = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pan_offset = np.array([0,0])
        self.window_open = True
        self.zoom = 1
        self.max_speed = False
              
        # Simulations and analysis instances
        self.simulation = simulation
        self.analysis = analysis
        
        # Mouse information
        self.mouse_click = False
        self.mouse_pos = pygame.mouse.get_pos()
        self.mouse_down = pygame.mouse.get_pressed()[0]
        
        # Interface  
        self.sidebar = SideBar(
            dimensions = (window_size[0]-270, 0, 270,  window_size[1]),
            background_color = Color.GREY3,
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
        
    def run(self):
        """
        Pygame main loop. Uses the ticker class to create consistent timespaces
        between ticks. In the main loop: Updates simulation, checks events,
        updates statistics on screen, draws screen en increments tick.
        """
        
        
        self.start()
        
        self.analysis.set_pygame_plot_dimensions(self.sidebar)
        while self.window_open:
            self.ticker.next_tick() 
            events = self.handle_events()
            self.update_gui(events)
            self.simulation.update()
            self.analysis.update(self.sidebar)
            self.draw()
            self.debug()
        pygame.quit()
        
    def handle_events(self):
        """
        Loops over current events and calls the associated methods.
        """
        # Exit events and mouse click events
        events = [event for event in pygame.event.get()]
        self.mouse_click = False
        for event in events:    
            if event.type == pygame.QUIT:
                self.window_open = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_click = True
                if event.button == 2:
                    self.mouse_pan()
                if event.button == 4:
                    self.mouse_zoom_in()
                if event.button == 5:
                    self.mouse_zoom_out()
                    
        # Other mouse information
        self.mouse_pos = pygame.mouse.get_pos()
        self.mouse_down = pygame.mouse.get_pressed()[0]
                    
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
        
        return events
        
    def update_gui(self, events):
        """
        Loops over all interface element to check for a match with the mouse
        click position, and executes the associated action.
        """
        
        # TODO: sidebar to pygame_gui
        
        if self.mouse_click:
            pos = pygame.mouse.get_pos()
            self.sidebar.check_click(pos)

        
    def draw(self):
        """
        Draw all simulation elements to the screen.
        """
        self.screen.fill(self.bg)
        for pop in self.simulation.populations.values():
            pop.draw(self.screen, self.pan_offset, self.zoom)
        for objset in self.simulation.object_sets.values():
            objset.draw(self.screen, self.pan_offset, self.zoom)


        self.display_textlist(self.ticker.get_stats(), Color.GREY3, 5, 5)
        self.sidebar.draw(self.screen)
        pygame.display.flip()  
          
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
        Calls self.change_zoom with increasing scaler.
        """
        self.change_zoom(scaler=1.2)

    def mouse_zoom_out(self):
        """
        Calls self.change_zoom with decreasing scaler.
        """
        self.change_zoom(scaler=0.8)
        
    def change_zoom(self, scaler):
        """
        Zooms in or out (centered on mouse position) and adjusts self.pan 
        offset to allign the screen.
        """
        prev_zoom = self.zoom
        self.zoom = min(max(self.zoom * scaler, 0.2), 5)
        midpoint = np.array(pygame.mouse.get_pos()) - self.pan_offset
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

    def debug(self):
        """
        Called every tick. Use to print stats.
        """
        #print(pygame.mouse.get_pos())



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
        t_min = self.hist.min().round(2)
        t_max = self.hist.max().round(2)
        t_avg = self.hist.mean().round(2)
        stats_list = []
        tick_str = 'load min/max/avg: '
        tick_str += str(t_min if t_min < 1 else '>1,0').ljust(4) + '/'
        tick_str += str(t_max if t_max < 1 else '>1.0').ljust(4) + '/'
        tick_str += str(t_avg if t_avg < 1 else '>1.0').ljust(4)
        stats_list.append(tick_str)
        return stats_list




