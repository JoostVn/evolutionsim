import numpy as np
import pygame
import numpy as np
from pygametools.color.color import Color



class SideBar:

    def __init__(self, dimensions, background_color, nr_figures):

        # Sidebar surface
        self.left, self.top, self.width, self.height = dimensions
        self.background_color = background_color
        self.surface = pygame.Surface((self.width, self.height))
        self.surface.fill(self.background_color)

        # Extending and collapsing sidebar
        self.extended = 0
        self.collapsing, self.extending = False, False
        self.draw_left = self.left + self.width

        # Figure dimensions and positions
        y_pad, x_pad = 10, 10
        self.fig_height = int((self.height - (1 + nr_figures) * y_pad) / nr_figures)
        self.fig_width = int(self.width - 2 * x_pad)
        self.figure_pos = [(x_pad, (i + 1) * y_pad + i * self.fig_height) for i in range(nr_figures)]
        self.figure_dim = (self.fig_width, self.fig_height)

        # Creating PlotFigures: simple surfaces on which plots are rendered
        self.figures = []
        for pos in self.figure_pos:
            fig = Figure(self.surface, pos, self.figure_dim)
            fig.clear()
            self.figures.append(fig)

    def draw(self, screen):
        """
        Calls the slide function to update the draw positions if the
        sidebar is currently extending or collapsing. Then, draws the
        sidebar surface including all plots to the screen.
        """
        self.slide(80)
        screen.blit(self.surface, (self.draw_left, self.top))

    def slide(self, speed):
        """
        Checks if the sidebar is collapsing or extending. If it is, updates
        the draw_left position (left x coordinate of the sidebar on the screen)
        and stops extending/collapsing if the max boundaries are reached.
        """
        if self.collapsing:
            self.draw_left += speed
            self.extended -= speed
            if self.extended <= 0:
                self.draw_left = self.left + self.width
                self.collapsing = False
                self.extended = 0
        if self.extending:
            self.extended += speed
            self.draw_left -= speed
            if self.extended >= self.width:
                self.draw_left = self.left
                self.extending = False
                self.extended = self.width

    def insert_canvas(self, plot_object, figure_index):
        """
        Insert a Pygame Plot object at the given figure index of the sidebar.
        """
        self.figures[figure_index].clear()
        self.figures[figure_index].draw(plot_object)

    def check_click(self, pos):
        """
        Checks if a click is within the sidebar region and starts collapsing
        or extending accordingly.
        """
        if ((self.left <= pos[0] <= self.left + self.width) and
            (self.top <= pos[1] <= self.top + self.height)):
            if self.extended == self.width:
                self.collapsing = True
            if self.extended == 0:
                self.extending = True


class Figure:

    def __init__(self, parent_surface, pos, dim):
        self.pos = pos
        self.dim = dim
        self.parent_surface = parent_surface
        self.surface = pygame.Surface(self.dim)

    def clear(self):
        self.surface.fill(Color.WHITE)
        self.parent_surface.blit(self.surface, self.pos)

    def draw(self, canvas):
        canvas.draw(self.surface)
        self.parent_surface.blit(self.surface, self.pos)


