import pygame
import numpy as np

class PlotDraw:    

    """
    Contains methods for drawing plot elements to the pygame
    screen. Plotting methods are either based on pygame 
    coordinates (pgc), which refer to positions in the screen,
    or graph coordinates (grc), which refer to postions in the
    graph with respect to its domain. Functions based on the latter
    coordinate system first have to convert to pgc prior to drawing.
    """

    FONT = 'msreferencesansserif'

    def __init__(self, plot):
        pygame.init()
    
        # Parent plot instance and dimensions
        self.plot = plot
        
        # Fonts
        self.font_l = (pygame.font.SysFont(self.FONT, 15), (0,0,0))
        self.font_m = (pygame.font.SysFont(self.FONT, 11), (0,0,0)) 
        self.font_s = (pygame.font.SysFont(self.FONT, 10), (0,0,0)) 

        # Standard colors
        self.color_bg = (255,255,255)
        self.color_lines = (0,0,0)

    def coords(self, point):
        """
        Converts graph coordinate (grc) points to Pygame coordinates (pgc)
        for drawing. Adjusts for scaling and reverses the y-coordinates.
        if a list of points is passed, it should have shape (nr_points, 2)
        """
        scaled = (point - self.plot.d_min) / self.plot.d_len
        scaled.T[1] = 1 - scaled.T[1]
        coords = (self.plot.dim * scaled + self.plot.pos).astype(int)
        return coords

    def inscope_points(self, x, y):
        """
        Return a boolean vector that indicates for each given point
        wether it is within the drawing domain. Used for preventing plot
        points are drawn outside the plot border.
        """
        in_scope = np.all((
            x >= self.plot.domain[0].min(),
            x <= self.plot.domain[0].max(),
            y >= self.plot.domain[1].min(),
            y <= self.plot.domain[1].max()),
            axis=0)
        return in_scope

    def grc_lines(self, screen, color, points, width=1):
        """
        >> Graph coordinates
        Draws a line on the screen trough a list of points. 
        """
        points = self.coords(points)
        pygame.draw.aalines(screen, color, False, points, width)

    def pgc_lines(self, screen, color, points, width=1):
        """
        >> Pygame coordinates
        Draws a line on the screen trough a list of points. 
        """
        pygame.draw.aalines(screen, color, False, points, width)

    def grc_line(self, screen, color, endpoints, width=1):
        """
        >> Graph coordinates
        Draws a line on the screen between two endpoints. 
        """
        endpoints = self.coords(endpoints)
        pygame.draw.aaline(screen, color, *endpoints, width)

    def pgc_line(self, screen, color, endpoints, width=1):
        """
        >> Pygame coordinates
        Draws a line on the screen between two endpoints. 
        """
        pygame.draw.aaline(screen, color, *endpoints, width)

    def grc_tick(self, screen, color, position, offset):
        """
        >> Graph coordinates / Pygame coordinates
        Draws a line from a graph coordinate point to a given offset in
        pgc. Used from drawing lines of fixed lenght from graph points,
        such as axis ticks and scatterpoint '+' markers.
        """
        position = self.coords(position)
        pygame.draw.line(screen, color, position, position + offset)
  
    def grc_point(self, screen, color, point, size):
        """
        >> Graph coordinates
        Draws a point on the screen with a given size.
        """
        point = self.coords(point)
        pygame.draw.circle(screen, color, point, size)

    def pgc_point(self, screen, color, point, size):
        """
        >> Pygame coordinates
        Draws a point on the screen with a given size.
        """
        pygame.draw.circle(screen, color, point, size)
        
    def grc_rect(self, screen, color, top_left, lower_right, border=False):
        """
        >> Graph coordinates
        Draws a rectangle on the screen with optional border.
        """
        dimensions = self.coords(np.vstack((top_left, lower_right)))
        self.pgc_rect(screen, color, *dimensions, border)

    def pgc_rect(self, screen, color, top_left, lower_right, border=False):
        """
        >> Pygame coordinates
        Draws a rectangle on the screen with optional border.
        """
        (left, top), (right, bottom) = (top_left, lower_right)
        rect = pygame.Rect(left, top, right-left, bottom-top)
        pygame.draw.rect(screen, color, rect)
        if border:
            corners = ((left,top),(right,top),(right,bottom),(left,bottom))
            pygame.draw.lines(screen, self.color_lines, True, corners, 1)

    def grc_text(self, screen, text, pos, font, offset=(0,0)):
        """
        >> Graph coordinates
        Draws centerd text at a given position. The optional offset parameter
        offsets the text position by some pygame coodinate value for x and y.
        """
        pos = self.coords(pos) + offset
        self.pgc_text(screen, text, pos, font)

    def pgc_text(self, screen, text, pos, font, centerx=False):
        """
        >> Pygame coordinates
        Draws left-alligned text at a given position. When centerx (int) is given,
        the x-value of pos is overriden, and the text is centered relative to the
        centerx coordinate.
        """
        font_type, font_color = font
        pygame_text = font_type.render(text, True, font_color)
        if centerx:
            pos = (centerx - pygame_text.get_rect().width / 2, pos[1])
        screen.blit(pygame_text, pos)

