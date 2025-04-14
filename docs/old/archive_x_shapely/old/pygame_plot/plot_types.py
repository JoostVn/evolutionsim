import numpy as np

class BasicGraph:
    
    """
    Parent class for basic plot elements. The size parameter is interpreted
    based on the plot context (marker size / line width / bar width etc.).
    Plot types can inherit from this class, and add a custom draw method.
    """

    def __init__(self, plot, x, y, color, size, label):
        self.plot = plot
        self.x = np.array(x)
        self.y = np.array(y)
        self.color = color
        self.size = size
        self.label = label
        self.in_scope = None
        self.update_dimensions()

    def update_dimensions(self):
        """
        Determines the inscope (x,y) points with respect to the plot domain.
        Call this function if the parent plot domain changes. self.in_scope
        is a boolean array that is True for each in-scope point.
        """
        self.in_scope = self.plot.pdraw.inscope_points(self.x, self.y)
        
    def draw(self, screen):
        """
        Draw function is to be overidden by sub classes.
        """
        pass

    def add_data(self, x, y):
        """
        Add x and y data to an already existing plot instance. Useful for
        dynamically changing animated plots. X and y should be passed as
        lists or arrays.
        """
        self.x = np.concatenate((self.x, x))
        self.y = np.concatenate((self.y, y))
        new_in_scope = self.plot.pdraw.inscope_points(x, y)
        self.in_scope = np.concatenate((self.in_scope, new_in_scope))
     
    def replace_data(self, x, y):
        """
        Replace x and y data of an already existing plot instance. Useful for
        dynamically changing animated plots. Computationally more expensive
        than add_data because all inscope points need to be recomputed.
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.update_dimensions()



class Line(BasicGraph):
    
    def __init__(self, plot, x, y, color, line_width, label):
        super().__init__(plot, x, y, color, line_width, label)
        
    def draw(self, screen):
        """
        Splits the data into in-scope line segments and draws them using
        the plotdraw instance.
        """
        indices = np.nonzero(self.in_scope[1:] != self.in_scope[:-1])[0] + 1
        segments = np.split(np.transpose((self.x, self.y)), indices)
        segments = segments[0::2] if self.in_scope[0] else segments[1::2]
        for segment in segments:
            if len(segment) >= 2:
                self.plot.pdraw.grc_lines(screen, self.color, segment, self.size)
        
    
    
class Scatter(BasicGraph):
    
    def __init__(self, plot, x, y, color, markersize, label, marker):
        super().__init__(plot, x, y, color, markersize, label)
        self.marker = marker
        
    def draw(self, screen):
        """
        Uses the plotdraw instance to draw each data point.
        """
        points = np.transpose((self.x[self.in_scope], self.y[self.in_scope]))

        for point in points:

            if self.marker == 'o':
                self.plot.pdraw.grc_point(screen, self.color, point, self.size)

            elif self.marker == '+':
                off = self.size/2
                self.plot.pdraw.grc_tick(screen, self.color, point, (-off, 0))
                self.plot.pdraw.grc_tick(screen, self.color, point, (off, 0))
                self.plot.pdraw.grc_tick(screen, self.color, point, (0, -off))
                self.plot.pdraw.grc_tick(screen, self.color, point, (0, off))

            elif self.marker == 'x':
                off = self.size/2
                self.plot.pdraw.grc_tick(screen, self.color, point, (off, off))
                self.plot.pdraw.grc_tick(screen, self.color, point, (-off, off))
                self.plot.pdraw.grc_tick(screen, self.color, point, (off, -off))
                self.plot.pdraw.grc_tick(screen, self.color, point, (-off, -off))

                
class Bar(BasicGraph):
    
    def __init__(self, plot, x, y, color, height, label):
        super().__init__(plot, x, y, color, height, label)

    def draw(self, screen):
        """
        Uses the plotdraw instance to draw each bar.
        """
        bars = zip(self.x[self.in_scope], self.y[self.in_scope])
        for center, top in bars: 
            top_left = (center - self.size/2, top)
            lower_right = (center + self.size/2, 0)
            self.plot.pdraw.grc_rect(screen, self.color, top_left, lower_right)






