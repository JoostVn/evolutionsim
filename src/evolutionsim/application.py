import pygame
from .interface import SideBar
from pygametools.color.color import Color
from pygametools.gui.base import Application
from pygametools.gui.elements import Button
import numpy as np



class SimApplication(Application):

    def __init__(self, window_size, simulation, analysis):
        super().__init__(
            window_size, tick_len=1/30, name='Evolution Learning',
            theme_name='default_dark')

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

        # Starting orientation
        self.pan_offset = np.array([20,50])
        self.zoom = 0.42
        self.sidebar.extending = True

        # TEST GUI

        def toggle_tickspeed():
            if self.ticker.tick_len > 1/35:
                self.ticker.tick_len = 1/100
            else:
                self.ticker.tick_len = 1/30

        self.set_gui([Button(
                text='Toggle speed', 
                func=toggle_tickspeed,
                pos=(10,10), 
                width=90,
                height=20)])


    def update(self):
        """
        Called from parent class. Updates simulation and analysis.
        """
        self.simulation.update()

        #TODO: move to gui
        self.analysis.update(self.sidebar)
        if pygame.BUTTON_LEFT in self.key_events['down']:
            pos = pygame.mouse.get_pos()
            self.sidebar.check_click(pos)

    def draw(self):
        #TODO: Move to simulation.draw
        """
        Called from parent class. Draw all simulation elements to the screen.
        """
        for pop in self.simulation.populations.values():
            pop.draw(self.screen, self.pan_offset, self.zoom)
        for objset in self.simulation.object_sets.values():
            objset.draw(self.screen, self.pan_offset, self.zoom)

        # TODO: move to GUI
        self.sidebar.draw(self.screen)




