from math import pi, sin, cos
import numpy as np
import shapely.vectorized
from shapely.affinity import rotate, translate, scale
from shapely.geometry import Point, LineString, LinearRing, Polygon, MultiPolygon, MultiLineString, MultiPoint
from matplotlib import pyplot as plt
import matplotlib
from scipy.spatial.distance import cdist
import warnings
from descartes.patch import PolygonPatch

class Bot:
    
    def __init__(self, position, angle, size):
        self.pos = position  
        self.angle = angle
        self.size = size
        self.polygon = self._create_polygon()
        self.rotate(self.angle)
        
        # Creating radialsensors
        angles = [
            (-1/24*pi, 1/24*pi), (1/24*pi, 1/2*pi), (1/2*pi, pi), (pi, 3/2*pi),
            (3/2*pi, 47/24*pi)]
        self.radial_sensors = RadialAreaSensor(self.pos, self.angle, 3, angles)
      
        # Creating line sensors
        angles = [-1/6*pi, 0, 1/6*pi]
        self.linear_sensors = LinearSensor(self.pos, self.angle, 10, angles)
      
    def _create_polygon(self):
        """
        Defines the bot shapely polygon that serves as a hitbox, food eating
        range and drawing shape.
        """
        polygon_points = [
            self.pos + (self.size, 0), 
            self.pos + (0, self.size/3), 
            self.pos - (self.size/3, 0), 
            self.pos - (0, self.size/3)]
        return Polygon(LineString(polygon_points))
    
    def rotate(self, delta_angle):
        """
        Rotates the bot and sensors.
        """
        self.polygon = rotate(
            self.polygon, delta_angle, origin=self.pos, use_radians=True)
      
    def draw(self, ax):
        """
        Matplotlib implementation of draw function using patches.
        """
        points = np.array(self.polygon.exterior.coords.xy).T
        bot_patch = matplotlib.patches.Polygon(points)
        ax.add_patch(bot_patch)
        bot_patch.set_color('white')
        bot_patch.set_edgecolor('black')
        bot_patch.set_linewidth(1.5)
        bot_patch.set_zorder(100)
    

class RadialAreaSensor:
    
    """
    Sensor that scans multiple circular areas around a bot for items and 
    returns a value for each area based on the closest item.
    """
    
    def __init__(self, bot_pos, bot_angle, sensor_range, sensor_angles):
        self.nr_sensors = len(sensor_angles)
        self.sensor_range = sensor_range
        self.sensor_angles = np.array(sensor_angles)
        self.polygon_base = self._create_polygon()
        self.polygon_current = self._update_position(bot_pos, bot_angle)
        self.sensor_values = np.zeros(self.nr_sensors)

    def _create_polygon(self):
        """
        Creates a shapely MultiPolygon that consists of multiple polygons
        bounding each sensor area. This method only initializes the base
        polygon, which is rotated and translated each tick to create the
        current polygon.
        """
        sensor_polygons = []
        for (a1, a2) in self.sensor_angles:
            outer_angles = np.append(np.arange(a1, a2, pi/24), a2)
            points = [(0,0)]
            for angle in outer_angles:
                point = np.multiply(self.sensor_range, (cos(angle), sin(angle)))
                points.append(point)
            sensor_polygons.append(Polygon(LineString(points)))
        return MultiPolygon(sensor_polygons)

    def _update_position(self, bot_pos, bot_angle):
        """
        Translates and rotates the MultiPolygon according to the current
        bot_pos and bot_angle.
        """
        new_pol = translate(self.polygon_base, *bot_pos)
        new_pol = rotate(
            new_pol, bot_angle, origin=tuple(bot_pos), use_radians=True)
        return new_pol

    def read(self, bot_pos, bot_angle, food_pos):
        """
        Reads the standardized sensor value based on the closest food item in
        range. Value is between 1 (max closeness) and 0 (no food in range).
        """
        self.sensor_values = np.zeros(self.nr_sensors)
        self.polygon_current = self._update_position(bot_pos, bot_angle)
        for i, pol in enumerate(self.polygon_current):
            in_area = shapely.vectorized.contains(pol, *food_pos.T)
            if in_area.any():
                min_dist = cdist(bot_pos.reshape(1,2), food_pos[in_area]).min()
                self.sensor_values[i] =  1 - min_dist / self.sensor_range
        return self.sensor_values

    def draw(self, ax):
        """
        Matplotlib implementation of draw function using patches.
        """
        cmap = plt.get_cmap('Greys')
        colors = cmap(self.sensor_values/3)
        patches = []
        for pol in self.polygon_current:
            points = np.array(pol.exterior.coords.xy).T
            patches.append(matplotlib.patches.Polygon(points))
        sensor_collection = matplotlib.collections.PatchCollection(patches)
        ax.add_collection(sensor_collection)
        sensor_collection.set_color(colors)
        sensor_collection.set_alpha(1)
        sensor_collection.set_edgecolor('grey')
        sensor_collection.set_linewidth(0.2)
        sensor_collection.set_zorder(10)
        

class LinearSensor:
    
    """
    Sensor that scans lines originating from the bot for objects and
    returns a value based on the closest object.
    """
    def __init__(self, bot_pos, bot_angle, sensor_range, sensor_angles):
        self.nr_sensors = len(sensor_angles)
        self.sensor_range = sensor_range
        self.sensor_angles = np.array(sensor_angles)
        self.line_base = self._create_line()
        self.line_current = self._update_position(bot_pos, bot_angle)
        self.sensor_values = np.zeros(self.nr_sensors)

    def _create_line(self):
        """
        Creates a shapely MultiLineString that consists of multiple sensor 
        lines. This method only initializes the base MultiLineString, which 
        is rotated and translated each tick to create the current 
        MultiLineString.
        """
        sensor_lines = []
        for angle in self.sensor_angles:
            end = np.multiply(self.sensor_range, (cos(angle), sin(angle)))
            line = LineString(((0,0), end))
            sensor_lines.append(line)
        return MultiLineString(sensor_lines)
        
    def _update_position(self, bot_pos, bot_angle):
        """
        Translates and rotates the MultiLineString according to the current
        bot_pos and bot_angle.
        """
        new_line = translate(self.line_base, *bot_pos)
        new_line = rotate(
            new_line, bot_angle, origin=tuple(bot_pos), use_radians=True)
        return new_line
    
    def read(self, bot_pos, bot_angle, barrier_polygons):
        """
        Reads the standardized sensor values based on crossings between the
        sensor lines and barriers. When one or more barriers are detected,
        returns a value based on the closest barrier between 1 (closest) and
        0 (no barrier).
        """
        self.sensor_values = np.zeros(self.nr_sensors)
        self.line_current = self._update_position(bot_pos, bot_angle)
        for i, line in enumerate(self.line_current):
            for polygon in barrier_polygons:
                if line.crosses(polygon):
                    intersection = line.intersection(polygon).boundary[0]
                    dist = Point(bot_pos).distance(intersection)
                    std_dist = 1 - (dist / self.sensor_range)
                    self.sensor_values[i] = max(self.sensor_values[i], std_dist)
        return self.sensor_values
        
    def draw(self, ax):
        """
        Matplotlib implementation of draw function.
        """
        for line, val, angle in zip(self.line_current, self.sensor_values, self.sensor_angles):
            
            # Plotting line
            ax.plot(*np.array(line.xy), color='red', alpha=0.05, linewidth=4, zorder=30)
            
            # Plotting intersection point
            if val > 0:
                dist = (1 - self.sensor_range) * val
                cross_point = line.interpolate(1-val, normalized=True).xy
                bot_pos = line.boundary[0].xy
                ax.scatter(*cross_point, s=90, color='red', 
                           edgecolor='black', alpha=0.7, zorder=400)
                line_points = np.vstack((np.array(bot_pos).T, np.array(cross_point).T))
                ax.plot(*line_points.T, zorder=50, color='darkred', alpha=0.8)
                
            


class Barrier:
    
    def __init__(self, nr_barriers, window_size):
        self.nr_barriers = nr_barriers
        self.polygon = self._create_polygon(window_size)
        
    def _create_polygon(self, window_size):
        """
        Create a MultiPolygon of random shapes to act as barriers. The window
        size parameters is given in (x,y) and determines the placement of the
        walls.
        """
        polygons = []
        
        # Random shapes
        offset = 2
        barrier_spawn_area = ([offset,offset], np.subtract(window_size,offset))
        for i in range(self.nr_barriers):
            pos = np.random.randint(*barrier_spawn_area, 2)
            size = np.random.uniform(0.2,1)
            scaling = np.random.uniform(0.5,2)
            rotation = np.random.uniform(0, 2*pi)
            circle = Polygon(Point(pos).buffer(size, resolution=2))
            points = np.array(circle.exterior.xy).T
            pol = rotate(
                scale(Polygon(LineString(points)), scaling), rotation, 
                origin=pos, use_radians=True)
            polygons.append(pol)
            
        # Walls
        xmax, ymax = window_size
        coords = np.array([(0,0), (xmax, 0), (xmax, ymax), (0, ymax), (0,0)])
        for p1, p2 in zip(coords[0:-1], coords[1:]):
            pol = LineString((p1, p2)).buffer(0.1, resolution=1, cap_style=1)
            polygons.append(pol)
            
        return MultiPolygon(polygons)
        
    def draw(self, ax):
        """
        Matplotlib implementation of draw function using patches.
        """
        patches = []
        for pol in self.polygon:
            points = np.array(pol.exterior.coords.xy).T
            patches.append(matplotlib.patches.Polygon(points))
        collection = matplotlib.collections.PatchCollection(patches)
        ax.add_collection(collection)
        collection.set_alpha(1)
        collection.set_color('grey')
        collection.set_zorder(40)


def simtest():
    """
    Creating a bot with random orientation and sensors that rotate relative
    to the linked bot. Then, assigns a value to each sensor based on the
    proximity of the closest point within sensor range.
    """
    x, y = 20, 20
    window_size = np.array((x,y))
    midpoint = window_size / 2

    # Defining barriers and safe spawn area
    barr = Barrier(8, window_size)   

    spawn_area = Polygon([(0,0),(x,0),(x,y),(0,y)])
    for barr_pol in barr.polygon:
        spawn_area = spawn_area.difference(barr_pol)
    spawn_area = spawn_area.buffer(-2)

    # Defining bot, random food points and barriers
    bot_pos = np.transpose(spawn_area.representative_point().xy)[0]
    bot = Bot(bot_pos, np.random.randint(0,2 * pi), 0.8) 
    food_pos = np.random.uniform((0,0), window_size, (30,2))
    bot.radial_sensors.read(bot.pos, bot.angle, food_pos)
    bot.linear_sensors.read(bot.pos, bot.angle, barr.polygon)

    # Plotting
    fig, ax = plt.subplots(figsize=(8,8))
    barr.draw(ax)
    bot.radial_sensors.draw(ax)
    bot.linear_sensors.draw(ax)
    bot.draw(ax)
    
    for p in food_pos:
        ax.scatter(*p, s=10, color='navy', alpha=0.8, zorder=30)
        ax.scatter(*p, s=30, color='navy', alpha=0.05, zorder=30)
        ax.scatter(*p, s=90, color='navy', alpha=0.05, zorder=30)


    
    spawn_patch = PolygonPatch(
        spawn_area.buffer(0), facecolor='green', alpha=0.05, zorder=1000)
    ax.add_patch(spawn_patch)



    ax.set_xlim(-1, window_size[0] + 1)
    ax.set_ylim(-1, window_size[1] + 1)
    plt.show()



warnings.simplefilter(action='ignore', category=FutureWarning)
simtest()
    


