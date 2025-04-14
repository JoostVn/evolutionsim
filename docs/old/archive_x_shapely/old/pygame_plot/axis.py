import math
import numpy as np


class Axis:
    
    """
    Contains the start and end points of an X or Y axis and handles the
    ticks and labels. The ax parameter determines x (0) or y (1) axis.
    """
    
    X = 0
    Y = 1
    
    def __init__(self, plot, ax):

        # Axis domain and specifications
        self.plot = plot
        self.ax = ax
        self.nr_ticks = 10

        # Other attributes
        self.custom_ticks = False
        self.lock_position = False
    
        # Ticks/label/positioning
        self.endpoints = None
        self.cross = None
        self.ticks = None
        self.labels = None
        self.tick_offset = None
        self.label_offset = None
        self.update_dimensions()

    def update_dimensions(self):
        """
        Computes the ax endpoints, tick locations and labels. The positioning
        of the axis is based on the zero-coordinate of the other axis, such
        that the two axis cross at (0,0). If the other axis domain does not
        contain zero, the position is set at the plot border (either left or bottom).
        Coordinates are in grc. If no custom ticks are given, this function also
        generates ticks based on the domain and nr_ticks attributes. 
        """  
        dom_self = self.plot.domain[self.ax]
        dom_othr = self.plot.domain[1 - self.ax]

        if self.lock_position or not (dom_othr.min() < 0 < dom_othr.max()):
            self.cross = dom_othr.min()
        else:
            self.cross = 0

        if self.ax == 0:
            self.endpoints = ((dom_self[0], self.cross), (dom_self[1], self.cross))
        if self.ax == 1:
            self.endpoints  = ((self.cross, dom_self[0]), (self.cross, dom_self[1]))

        if not self.custom_ticks:
            rounding = (- math.floor(math.log10(abs(dom_self).max()))) + 2
            numlabels = np.linspace(*dom_self, num=self.nr_ticks)
            self.labels = numlabels.round(rounding).astype(str)
            self.ticks = np.linspace(*self.endpoints, num=self.nr_ticks)
            self.offset_labels()

    def offset_labels(self):
        """
        Determines the offset for ticks and labels based on self.ax and
        label length. Also centers (x) or right-adjusts (y) all labels.
        """
        label_len = max(len(label) for label in self.labels)
        if self.ax == self.X:
            self.label_offset = (-label_len * 3, 3)
            self.tick_offset = (0,3)
            self.labels = [label.center(label_len) for label in self.labels]
        elif self.ax == self.Y:
            self.label_offset = (-label_len * 7 - 3, -7)
            self.tick_offset = (-3,0)
            self.labels = [label.center(label_len) for label in self.labels]
        
    def set_labels(self, ticks, labels):
        """
        Sets the self.ticks and self.labels attributes according to given
        lists or arrays. The ticks parameter is given as a list of single
        (either x or y, depending on the axis) coordinates that indicate
        the tick locations. This list is the converted to a list of (x,y)
        coordinate (grc) points.
        """
        self.ticks = np.vstack((ticks, np.full(len(ticks), self.cross))).T
        if self.ax == self.Y:
            self.ticks = np.flip(self.ticks, axis=1)
        self.labels = labels
        self.offset_labels()
        self.custom_ticks = True
    
    def draw(self, screen):
        """
        Draws the axis line, tick marks and labels. The positioning of the tick
        marks and labels relative to the absolute tick locations are determined
        by labl_offset and tick_offset. If no tick locations and labels are
        specified, generates them with self.generate_ticks().
        """
        self.update_dimensions()
        font = self.plot.pdraw.font_s
        color = self.plot.pdraw.color_lines
        self.plot.pdraw.grc_line(screen, color, self.endpoints)
        for tick, label in zip(self.ticks, self.labels):
            self.plot.pdraw.grc_text(screen, label, tick, font, self.label_offset)
            self.plot.pdraw.grc_tick(screen, color, tick, self.tick_offset)
