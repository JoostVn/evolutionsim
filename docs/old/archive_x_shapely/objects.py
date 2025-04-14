import pygame
from pygame import gfxdraw
import numpy as np
from numpy.random import uniform, randint
from math import pi
from pygame_plot import Color
from shapely.affinity import rotate, scale
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from matplotlib import pyplot as plt
from descartes.patch import PolygonPatch


"""
Each object should at least have .initialize(), .update() and .draw() methods.
"""


class FoodSupply:
    
    def __init__(self, environment_size, quantity):
        self.quantity = quantity
        self.environment_size = environment_size
        self.pos = np.empty(0)

    def initialize(self):
        """
        Places food objects at random locations.
        """
        self.pos = np.random.uniform((0,0), self.environment_size, (self.quantity,2))
        self.value = np.ones(self.quantity)
        self.available = np.ones(self.quantity, bool)
    
    def eat(self, indices):
        """
        Check if the food at the given indices is still available. Then returns 
        the sum of food values for all available food items.
        """
        available = indices[self.available[indices]]
        self.available[available] = False
        return self.value[available].sum()
        
    def update(self, populations, object_sets):
        """
        Restocks the food supply.
        """
        new_pos = np.random.uniform((0,0), self.environment_size, (self.quantity,2))
        new_value = np.ones(self.quantity)
        self.pos[~self.available] = new_pos[~self.available]
        self.value[~self.available] = new_value[~self.available]
        self.available = np.ones(self.quantity, bool)
   
    def draw(self, screen, pan_offset, zoom):
        for pos in self.pos:
            pos_draw = (pos * zoom + pan_offset).astype(int)
            pygame.draw.circle(screen, Color.GREY5, pos_draw, 2) 
            

class Barriers:
    
    def __init__(self, environment_size, quantity, size, wall_width):
        self.environment_size = environment_size
        self.quantity = quantity
        self.size = size
        self.wall_width = wall_width
        self.polygon = []
        self.color = Color.GREY5
        
    def _create_polygon(self):
        """
        Create a MultiPolygon of random shapes to act as barriers. The window
        size parameters is given in (x,y) and determines the placement of the
        barriers. Also creates walls bounding the window region.
        """
        # Walls
        polygons = []
        rect = box(0, 0, *self.environment_size)
        points = np.transpose(rect.exterior.xy)
        for wall_segment in zip(points[:-1], points[1:]):
            wall_pol = LineString(wall_segment).buffer(
                self.wall_width/2, resolution=1, cap_style=1)
            polygons.append(wall_pol)
        
        # Random (non overlapping) barriers
        nr_walls = len(polygons)
        while len(polygons) < self.quantity + nr_walls:
            pos = randint((0,0), self.environment_size, 2)
            barr_pol = Point(pos).buffer(self.size, resolution=2)
            barr_pol = scale(barr_pol, *uniform(0.5, 2, 2))
            barr_pol = rotate(barr_pol, uniform(0, 2*pi), pos, use_radians=True)
            if barr_pol.intersects(unary_union(polygons)):
                continue
            else:
                polygons.append(barr_pol)
                
        return MultiPolygon(polygons)
        
    def initialize(self):
        """
        Places food objects at random locations. Can be called multiple times
        to reset barriers to new random positions.
        """
        self.polygon = self._create_polygon()
    
    def get_safe_area(self, margin):
        """
        Returns a shapely MultiPolygon of the area bonuded by self.environment_size,
        with a given margin around all objects. This area can then be used with
        the MultiPolygon.representative_point() method to generate safe spawn
        points for bots, food, and other objects.
        """
        area = box(0, 0, *self.environment_size)
        for pol in self.polygon:
            area = area.difference(pol)
        area = area.buffer(-margin)
        return area
    
    def get_safe_point(self, safe_area):
        """
        Similar to get_safe_area, but returns a random (x,y) point in a given 
        safe_area MultiPolygon.
        """
        bounds = np.reshape(safe_area.bounds, (2,2))
        while True:
            pos = uniform(*bounds, 2)
            if safe_area.contains(Point(pos)):
                return pos
    
    
    def update(self, populations, object_sets):
        """
        TODO: implement moving barriers.
        """
        pass
    
    def draw(self, screen, pan_offset, zoom):
        """
        Draw the barrier polygons to a Pygame screen.
        """
        for pol in self.polygon:
            pol_points = np.array(pol.exterior.coords.xy).T   
            pol_points_draw = (pol_points * zoom + pan_offset).astype(int)
            gfxdraw.aapolygon(screen, pol_points_draw, self.color)
            pygame.draw.polygon(screen, self.color, pol_points_draw)


    def plot(self, safe_area_margin=None):
        """
        Matplotlib plot of barriers for debugging purposes. Can also show an
        exampel of the Barrier.get_safe_area() method.
        """
        fig, ax = plt.subplots(figsize=(8,8))
        for pol in self.polygon:
            patch = PolygonPatch(
                pol, facecolor='grey', alpha=0.7, edgecolor='none')
            ax.add_patch(patch)
        if not safe_area_margin is None:
            area = self.get_safe_area(safe_area_margin)
            patch = PolygonPatch(
                area, facecolor='green', alpha=0.1, edgecolor='none')
            ax.add_patch(patch)
        ax.set_xlim(-20, self.environment_size[0] + 20)
        ax.set_ylim(-20, self.environment_size[1] + 20)
        plt.show()
        
        
        
        
