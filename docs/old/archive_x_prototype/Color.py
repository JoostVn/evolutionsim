from random import sample
import numpy as np

class Color:
    
    
    BLACK = (0,0,0)
    GREY1 = (40,40,40)
    GREY2 = (80,80,80)
    GREY3 = (120,120,120)
    GREY4 = (160,160,160)
    GREY5 = (200,200,200)
    GREY6 = (230,230,230)
    GREY7 = (240,240,240)
    WHITE = (255,255,255)
    
    DBLUE = (20,20,80)
    MBLUE = (40,40,200)
    LBLUE = (110,110,255)
    
    DGREEN = (40,100,40)
    MGREEN = (70,200,70)
    LGREEN = (150,230,150)

    DRED = (80,0,0)
    MRED = (200,40,40)
    LRED = (250,90,90)

    DCYAN = (20, 120, 130)
    MCYAN = (50, 160, 180)
    LCYAN = (150, 220, 230)
    
    GOLD = (240, 190, 50)
    
    
    def random_vibrant():
        while True:
            R, G, B = sample(range(0,256), 3)
            diff_sum = abs(R-G) + abs(G-B) + abs(R-B)
            if diff_sum > 200:
                return (R, G, B)
            
    def random_dull():
        while True:
            R, G, B = sample(range(0,256), 3)
            diff_sum = abs(R-G) + abs(G-B) + abs(R-B)
            totl_sum = R + G + B
            if diff_sum > 80 and diff_sum < 200 and totl_sum  > 200 and totl_sum < 500:
                return (R, G, B)
            
            

class ColorGradient:
    
    def __init__(self, color_1, color_2, nr_partitions):
        self.c1 = color_1
        self.c2 = color_2
        self.parts = nr_partitions
        color_mapper = np.vectorize(self.get_color) 
        self.color_vector = np.array(color_mapper(np.arange(nr_partitions))).T
        
    def get_color(self, part):
        """
        Get RGB color values of the color that sits between color1 and color2
        at the desired partition part should be >= 0  and <= nr_partitions. 
        """
        R_width = self.c2[0] - self.c1[0]
        G_width = self.c2[1] - self.c1[1]
        B_width = self.c2[2] - self.c1[2]
        R = int(self.c1[0] + (part / self.parts) * R_width)
        G = int(self.c1[1] + (part / self.parts) * G_width)
        B = int(self.c1[2] + (part / self.parts) * B_width)
        return (R, G, B)
        
    
    
    
    
if __name__ == '__main__':
    
    g = ColorGradient(Color.BLUE, Color.WHITE, 10)
    v = g.color_vector
    
    print(v)
    
    
    
    
        
        
