import pygame
from pygame import gfxdraw
import numpy as np
from numpy.random import uniform
from math import pi
from pygametools.color.color import Color
import matplotlib
from math import sin, cos
from algorithms.geometry.shapes import Polygon
from matplotlib import pyplot as plt


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
            dot_size = int(max(1, 3 * zoom))
            gfxdraw.aacircle(
                screen, *pos_draw, dot_size, Color.GREY2)
            gfxdraw.filled_circle(screen, *pos_draw, dot_size, Color.GREY2)



class Barriers:

    def __init__(self, environment_size, quantity, size, wall_width):
        self.environment_size = environment_size
        self.quantity = quantity
        self.size = size
        self.wall_width = wall_width
        self.polygons = []
        self.color = Color.GREY3

    def _create_polygon(self):
        """
        Create a MultiPolygon of random shapes to act as barriers. The window
        size parameters is given in (x,y) and determines the placement of the
        barriers. Also creates walls bounding the window region.
        """
        # Walls
        polygons = []
        x, y = self.environment_size
        w = self.wall_width / 2
        polygons.append(Polygon([(-w,-w),(x+w,-w),(x+w,w),(-w,w)]))
        polygons.append(Polygon([(x-w,-w),(x+w,-w),(x+w,y+w),(x-w,y+w)]))
        polygons.append(Polygon([(-w,y-w),(x+w,y-w),(x+w,y+w),(-w,y+w)]))
        polygons.append(Polygon([(-w,-w),(w,-w),(w,y+w),(-w,y+w)]))

        # Random (non overlapping) barriers
        nr_walls = len(polygons)

        while len(polygons) < self.quantity + nr_walls:
            pos = np.random.uniform((0,0), self.environment_size, 2)
            angles = np.arange(0, 2*pi, 1/3*pi)
            vec = np.array([(cos(a), sin(a)) for a in angles])
            scramble = np.random.uniform(-self.size/8, self.size/8, vec.shape)
            points = pos + vec * (self.size/2) + scramble
            barr_polygon = Polygon(points)
            inter = [len(barr_polygon.intersect_pol(pol)) for pol in polygons]

            if sum(inter) > 0:
                continue
            polygons.append(barr_polygon)

        return polygons

    def initialize(self):
        """
        Places food objects at random locations. Can be called multiple times
        to reset barriers to new random positions.
        """
        self.polygons = self._create_polygon()

    def get_safe_point(self):
        """
        Similar to get_safe_area, but returns a random (x,y) point in a given
        safe_area MultiPolygon.
        """
        while True:
            xy = uniform((0,0), self.environment_size, 2)
            pol_contains = [pol.contains_point(xy) for pol in self.polygons]
            if sum(pol_contains) == 0:
                return xy
            else:
                continue

    def update(self, populations, object_sets):
        """
        TODO: implement moving barriers.
        """
        pass

    def draw(self, screen, pan_offset, zoom):
        """
        Draw the barrier polygons to a Pygame screen.
        """
        for pol in self.polygons:
            pol_points_draw = (pol.points * zoom + pan_offset).astype(int)
            gfxdraw.aapolygon(screen, pol_points_draw, self.color)
            pygame.draw.polygon(screen, self.color, pol_points_draw)


    def plot(self, safe_area_margin=None):
        """
        Matplotlib plot of barriers for debugging purposes. Can also show an
        exampel of the Barrier.get_safe_area() method.
        """
        fig, ax = plt.subplots(figsize=(8,8))

        # Create matplotlib patches from polygons
        patches = []
        for pol in self.polygons:
            patches.append(matplotlib.patches.Polygon(pol.points))
        collection = matplotlib.collections.PatchCollection(patches)
        collection.set_alpha(0.4)
        collection.set_color('grey')
        collection.set_zorder(5)
        ax.add_collection(collection)

        # Plot points
        points = np.array([self.get_safe_point() for i in range(1000)])
        ax.scatter(*points.T, color='red', alpha=0.2, zorder=5, s=20)

        # Set limits and show plot
        ax.set_xlim(-50, self.environment_size[0] + 50)
        ax.set_ylim(-50, self.environment_size[1] + 50)
        plt.show()





if __name__ == '__main__':

    environment_size = np.array((600,600))
    b = Barriers(environment_size, quantity=5, size=90, wall_width=2)
    b.initialize()
    b.plot(20)
