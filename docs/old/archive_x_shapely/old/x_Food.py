import numpy as np
from Color import Color
import pygame

class Food:
    """
    Simple food class that contains a position and a draw function.
    """
    
    UNIFORM = 0
    NORMAL = 1
    
    def __init__(self):
        self.pos = None
        self.size = 2
        self.color = Color.GREY4
        
    def uniform_position(self, domain):
        self.pos = np.random.randint(*domain, 2)
    
    def normal_position(self, midpoint, spread):
        self.pos = np.random.normal(midpoint,spread,2).astype(int)
    
    def draw(self, screen, pan_offset, zoom):
        pos_draw = (self.pos * zoom + pan_offset).astype(int)
        size_draw = int(max(1, self.size * zoom))
        
        
        x, y = pos_draw
        
        
        rect = pygame.Rect(x+size_draw, y+size_draw, size_draw, size_draw) 
        
        
        pygame.draw.rect(screen, self.color, rect) 
        
        
        
        
        
        #pygame.draw.circle(screen, self.color, pos_draw, size_draw)
