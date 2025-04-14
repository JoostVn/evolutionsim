import tkinter as tk
from tkinter import filedialog
from os import getcwd
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon, MultiLineString

def genomefile_open(scaledown=False):
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfile(
        mode='r',
        title='Select genome file to open', 
        initialdir=f'{getcwd()}\\genomes', 
        filetypes=[('Genome .txt file','*.txt')],
        multiple=False)
    genome = [float(gen.replace('\n', '')) for gen in file.readlines()]
    custom_genome = np.array(genome)
    file.close()
    return custom_genome

