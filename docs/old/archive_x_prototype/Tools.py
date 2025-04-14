from math import pi, ceil, floor, atan2, cos, sin, acos, sqrt
import numpy as np
import time


def convert_to_angle(vector):
    """
    Converts a direction vector to an angle in radians.
    """
    return atan2(vector[1], vector[0])


def convert_to_vector(angle):
    """
    Converts an angle to a direction vector. 
    """
    return np.array([cos(angle), sin(angle)])


def standard_angle(angle):
    """
    Standardizedes an angle in radians to a range of [0, 2pi]
    """
    if (angle > 2*pi):
        angle -= floor(angle/(2*pi)) * 2 * pi
    if (angle < 0):
        angle -= ceil(angle/(2*pi)) * 2 * pi
        angle += 2*pi
    return angle
    

def distance(a, b):
    """
    return the eucledian distance between points a and b
    """
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def vector_angle(vec1, vec2):
    """
    Computes the angle between two vectors.
    """
    x1, y1 = vec1
    x2, y2 = vec2
    angle = acos((x1 * x2 + y1 * y2) / (sqrt(x1**2 + y1**2) * sqrt(x2**2 + y2**2)))
    return angle


def save_genome(genome, file):
    """
    Writes a genome to txt file.
    """
    gene_file = open(file, 'w')
    lines = '\n'.join(genome.astype(str))
    gene_file.writelines(lines)
    gene_file.close()
    
    
def load_genome(file):
    """
    Opens a genome from txt file.
    """
    gene_file = open(file)
    str_genes = gene_file.readlines()
    genome = np.array([gene.replace('\n','') for gene in str_genes]).astype(float)
    gene_file.close()
    return genome


