from pygame_plot.axis import Axis
from pygame_plot.plot_types import Line, Scatter, Bar
from pygame_plot.plotdraw import PlotDraw
from pygame_plot.legend import Legend
import numpy as np


"""
TODO:
    - AA pygame functions
    - Everything to numpy arrays
    - Axis labels
    - Add support for plotting text/labels in graph
    - Add NN plot compatibility
    - Opacity for plot elements       
    - Save plots as images
    - Convert plot components to image or surface and redraw that 
      instead of rerunning all draw methods each iteration 
"""


class Plot:
    
    """
    Container that holds all plot components and attributes.
    """
    
    def __init__(self, xdomain, ydomain, pos, dim):
        
        # Dimensions and positioning        
        self.domain = np.vstack((xdomain, ydomain))
        self.d_len = np.diff(self.domain, axis=1).T[0]
        self.d_min = np.min(self.domain, axis=1)
        self.d_max = np.max(self.domain, axis=1)
        self.pos = np.array(pos)
        self.dim = np.array(dim)
        
        # PlotDraw instance
        self.pdraw = PlotDraw(self)
        
        # plot components
        self.xaxis = Axis(self, Axis.X)
        self.yaxis = Axis(self, Axis.Y)
        self.title = ''
        self.legend = None
        self.elements = {}
        self.border = True

    def update_dimensions(self, xdomain=None, ydomain=None, pos=None, dim=None):
        """
        Updates the position, dimensions and domain of the plot. Use this
        function to change any of these values for an already initialized
        instance. Some domain len, min, and max are pre calculated for fast
        coordinate conversion. 
        """
        if xdomain is not None or ydomain is not None:
            self.domain = np.vstack((xdomain, ydomain))
            self.d_len = np.diff(self.domain, axis=1).T[0]
            self.d_min = np.min(self.domain, axis=1)
            self.d_max = np.max(self.domain, axis=1)
            if xdomain is not None:
                self.xaxis.update_dimensions()
            if ydomain is not None:
                self.yaxis.update_dimensions()
                
        if pos is not None:
            self.pos = np.array(pos)
        if dim is not None:
            self.dim = np.array(dim)

        if self.legend is not None:
            self.legend.update_dimensions()

        for element in self.elements.values():
            element.update_dimensions()
       
    def add_legend(self, location='outer right', width=80, border=True):
        """
        Add a legend to the plot. Kwargs are passed to the Legend instance. 
        """
        self.legend = Legend(self, location, width, border)
        self.legend.update_dimensions()
    
    def add_line(self, x, y, color, line_width, label):
        """
        Adds a line plot to the plot elements.
        """
        self.elements[label] = Line(self, x, y, color, line_width, label)
        if self.legend is not None:
            self.legend.update_dimensions()
    
    def add_scatter(self, x, y, color, marker_size, label, marker='o'):
        """
        Adds a scatter plot to the plot elements.
        """
        self.elements[label] = Scatter(self, x, y, color, marker_size, label, marker)
        if self.legend is not None:
            self.legend.update_dimensions()
            
    def add_bar(self, x, height, color, width, label):
        """
        Adds a bar plot to the plot elements.
        """
        self.elements[label] = Bar(self, x, height, color, width, label)
        if self.legend is not None:
            self.legend.update_dimensions()
            
    def draw(self, screen):
        """
        Draw all plot elements in the Pygame plot.
        """
        
        # Draw background
        top_left = np.diagonal(self.domain)
        lower_right = np.diagonal(np.fliplr(self.domain))
        self.pdraw.grc_rect(
            screen, self.pdraw.color_bg, top_left, lower_right, self.border)
        
        # Draw plot elements and axes
        for element in self.elements.values():
            element.draw(screen)
        self.xaxis.draw(screen)
        self.yaxis.draw(screen)
        
        # Draw title
        centerx = self.pos[0] + self.dim[0] / 2
        title_pos = (self.pos[0], self.pos[1] - 22)
        self.pdraw.pgc_text(
            screen, self.title, title_pos, self.pdraw.font_l, centerx)
        
        # Draw legend
        if self.legend:
            self.legend.draw(screen)

    def surface_draw(self, screen):
        """

        WIP

        When this method is first called, it creates a surface object and
        draws all plot components to that surface instead of the screen.
        The surface object is then saved as self.surface, and drawn to the
        screen with self.surface.blit(). Each next call to this method just
        redraws the surface instead of building the graph from scratch. When
        self.update_dimensions is called, the surface has to be recreated.
        therefore, this function is faster for static plots, but slower for
        dynamic plots.

        https://stackoverflow.com/questions/55969514/is-there-a-way-to-copy-a-certain-section-of-a-surface-using-surface-copy
        """
        pass

    def draw_background(self, screen):
        """
        Draw the plot background rectangle.
        """

    def draw_title(self, screen):
        """
        Draw the plot title centered above the graph area. 
        """
    
