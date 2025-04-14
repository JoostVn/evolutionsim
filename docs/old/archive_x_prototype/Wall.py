from Color import Color
import numpy as np
import pygame

class Wall:

    def __init__(self, origin, destination, size):
        self.endpoints = np.array((origin, destination))
        self.size = size
        self.color = Color.DGREY

    def draw(self, screen, pan_offset, zoom):
        pos_draw = (self.endpoints * zoom + pan_offset).astype(int)
        size_draw = int(max(1, self.size * zoom))
        pygame.draw.line(screen, self.color, *pos_draw, size_draw)


