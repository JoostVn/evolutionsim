from math import pi, sin, cos
import numpy as np
import time


import shapely

from matplotlib import pyplot as plt
import matplotlib
from scipy.spatial.distance import cdist



def polygon_point_contains(polygon_points, nr_points):
    """
    Testing point containment check for polygons and plotting the result.
    """
    
    polygon = shapely.geometry.Polygon(shapely.geometry.LineString(polygon_points))
    points = np.random.uniform(0,10,(nr_points, 2))
    
    # Determining in and out points
    t_start = time.time()
    in_area = shapely.vectorized.contains(polygon, *points.T)
    t = time.time() - t_start
    print(f'{round(t,15)} seconds')
    print(f'{1/t} rate')

    # Plotting polygon and points only if manageable nr_points
    if nr_points < 1000:
        plt.figure(figsize=(7,7))
        plt.plot(*polygon.exterior.xy, linewidth=2, color='black')
        
        for xy, isin in zip(points, in_area):
            if isin:
                plt.scatter(*xy, s=100, color='green', alpha=0.4)
            else:
                plt.scatter(*xy, s=100, color='red', alpha=0.4)
        plt.show()







def sensor_test():
    """
    Creating a bot with random orientation and sensors that rotate relative
    to the linked bot. Then, assigns a value to each sensor based on the
    proximity of the closest point within sensor range.
    """

    # Defining bot (standard orientation to the east)
    bot_pos = np.array((5,5))
    size = 0.8
    polygon_points = [
        bot_pos + (size, 0), 
        bot_pos + (0, size/3), 
        bot_pos - (size/3, 0), 
        bot_pos - (0, size/3)]
    bot_polygon = Polygon(LineString(polygon_points))

    # Defining radial sensors
    sensor_range = 3
    sensor_angles = [
        (-1/8 * pi, 1/8 * pi),
        (1/8 * pi, 1/2 * pi),
        (1/2 * pi, pi),
        (pi, 3/2 * pi),
        (3/2 * pi, 15/8 * pi)]
    get_point = lambda angle: bot_pos + (
        (sensor_range * cos(angle), sensor_range * sin(angle)))
    sensor_points = [
        [bot_pos, get_point(a[0]), get_point(a[1])] for a in sensor_angles]
    sensor_polygons = MultiPolygon([
        Polygon(LineString(points)) for points in sensor_points])
    
    # Bot and sensor rotation
    rotation = np.random.randint(0,360)
    bot_polygon = rotate(bot_polygon, rotation, origin=tuple(bot_pos))
    sensor_polygons = rotate(sensor_polygons, rotation, origin=tuple(bot_pos))

    # Creating random food points and assigning sensor values
    food_pos = np.random.uniform(1,9,(10,2))
    sensor_values = np.zeros(len(sensor_angles))
    for i, pol in enumerate(sensor_polygons):
        in_area = shapely.vectorized.contains(pol, *food_pos.T)
        if in_area.any():
            min_dist = cdist(bot_pos.reshape(1,2), food_pos[in_area]).min()
            sensor_values[i] =  1 - min_dist / sensor_range
    
    # Plotting
    cmap = plt.get_cmap('Blues')
    colors = cmap(sensor_values)
    fig, ax = plt.subplots(figsize=(7,7))
    
    # Creating patches for plotting sensors
    patches = []
    for pol in sensor_polygons:
        points = np.array(pol.exterior.coords.xy).T
        patches.append(matplotlib.patches.Polygon(points))
    sensor_collection = matplotlib.collections.PatchCollection(patches)
    ax.add_collection(sensor_collection)
    sensor_collection.set_color(colors)
    sensor_collection.set_alpha(0.35)
    sensor_collection.set_edgecolor('black')
    sensor_collection.set_linewidth(1)
    
    # Plotting food
    for p in food_pos:
        ax.scatter(*p, s=50, color='navy', alpha=0.4)
    
    # Plotting bot
    points = np.array(bot_polygon.exterior.coords.xy).T
    bot_patch = matplotlib.patches.Polygon(points)
    ax.add_patch(bot_patch)
    bot_patch.set_color('white')
    bot_patch.set_edgecolor('black')
    bot_patch.set_linewidth(2)
    
    # Finalizing plot
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    plt.show()



#sensor_test()
    
polygon_points = np.array(Point(5, 5).buffer(2.0).exterior.xy).T
polygon_point_contains(polygon_points, 500)

