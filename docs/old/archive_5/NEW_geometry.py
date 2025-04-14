import numpy as np
from math import pi, sin, cos
import numba
from numba import jit, boolean, float64, int64, boolean
import math
        

@jit(numba.types.Tuple((int64, float64[:,:,:], float64[:,:], float64[:,:]))(
        float64[:,:]), nopython=True)
def get_edge_data(pts):
    """
    Get the number of polygon edges, and for each edge an array of point pairs, 
    vectors and domains. 
    """
    n = len(pts)
    idx = list(range(n))
    pairs_idx = zip(idx, idx[1:] + [idx[0]])
    points = np.ones((n,2,2))
    vectors = np.ones((n,2))
    domains = np.ones((n,4))
    for i, j in pairs_idx:
        points[i][0][0] = pts[i][0]
        points[i][0][1] = pts[i][1]
        points[i][1][0] = pts[j][0]
        points[i][1][1] = pts[j][1]
        vectors[i][0] = pts[j][0] - pts[i][0]
        vectors[i][1] = pts[j][1] - pts[i][1]
        domains[i][0] = min(pts[i][0], pts[j][0])
        domains[i][1] = max(pts[i][0], pts[j][0])
        domains[i][2] = min(pts[i][1], pts[j][1])
        domains[i][3] = max(pts[i][1], pts[j][1])
    return n, points, vectors, domains


@jit(float64[:,:](float64[:,:], float64, float64), nopython=True)
def translate(points, x, y):
    """
    Translate the geometry points in a given (x,y) direction.
    """
    for i in range(len(points)):
        points[i][0] = points[i][0] + x
        points[i][1] = points[i][1] + y
    return points


@jit(float64[:,:](float64[:,:], float64, float64, float64), nopython=True)
def rotate(points, delta_angle, center_x, center_y):
    """ 
    Rotate the geometry by delta_angle radians around a center point.
    """
    cosang, sinang = np.cos(-delta_angle), np.sin(-delta_angle)
    for i in range(len(points)):
        tx = points[i][0] - center_x
        ty = points[i][1] - center_y
        points[i][0] = center_x + (tx * cosang + ty * sinang)
        points[i][1] = center_y + (-tx * sinang + ty * cosang)
    return points


@jit(boolean(float64[:], float64[:]), nopython=True)
def overlaps_domain(d1, d2):
    """
    Cheap function that returns True if there is some overlap in
    domain d1 and domain d2. The domains are given as (xmin, xmax, ymin, ymax).
    """
    return not (
        d1[1] < d2[0] or d2[1] < d1[0] or d1[3] < d2[2] or d2[3] < d1[2])


@jit(float64[:](
    float64[:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:]), 
    nopython=True)
def intersect_line_line(pts1, pts2, vec1, vec2, dom1, dom2):
    """
    Find the intersection point of two line segments, and return None if no
    intersection exists. Based on PyEuclid module.
    """
    nan = np.array([np.nan, np.nan])
    if not overlaps_domain(dom1, dom2):
        return nan
    d = vec2[1] * vec1[0] - vec2[0] * vec1[1]
    if d == 0:
        return nan
    dy = pts1[0][1] - pts2[0][1]
    dx = pts1[0][0] - pts2[0][0]
    ua = (vec2[0] * dy - vec2[1] * dx) / d
    if not (0 <= ua <= 1):
        return nan
    ub = (vec1[0] * dy - vec1[1] * dx) / d
    if not (0 <= ub <= 1):
        return nan
    return pts1[0] + ua * vec1
    

@jit(float64[:,:](
    float64[:,:], float64[:,:], float64[:], float64[:], float64[:]), 
    nopython=True)
def intersect_line_pol(pts_line, pts_pol, dom_line, dom_pol, vec_line):
    """
    Find the intersection points between a line and a polygon exterior. Return 
    the (n,x,y) coordinates of intersection points as a numpy array.
    """
    
    # Cheap domain check
    if not overlaps_domain(dom_line, dom_pol):
        return np.array([[np.nan],[np.nan]])
    
    # Get edge node pair indices and initialize intersections
    n_edges, pts_edges, vec_edges, dom_edges = get_edge_data(pts_pol)
    intersections = np.ones((n_edges, 2))
    
    # Loop over edges to find intersections with line
    for i in range(n_edges):
        intr = intersect_line_line(
            pts_line, pts_edges[i], 
            vec_line, vec_edges[i], 
            dom_line, dom_edges[i])
        intersections[i][0] = intr[0]
        intersections[i][1] = intr[1]
    return intersections


@jit(float64[:,:,:](float64[:,:], float64[:,:], float64[:], float64[:]), 
     nopython=True)
def intersect_pol_pol(pts1, pts2, dom1, dom2):
    """
    TODO
    """
    # Cheap domain check
    nan = np.array([[[np.nan, np.nan]]])
    if not overlaps_domain(dom1, dom2):
        return nan
    
    # Get polygon edges and initialize intersections array
    n_e1, pts_e1, vec_e1, dom_e1 = get_edge_data(pts1)
    n_e2, pts_e2, vec_e2, dom_e2 = get_edge_data(pts2)
    intersections = np.ones((n_e1, n_e2, 2))
    
    # Loop over edges for both polygons and check edge intersections
    for i in range(n_e1):
        for j in range(n_e2):
            inter = intersect_line_line(
                pts_e1[i], pts_e2[j], 
                vec_e1[i], vec_e2[j], 
                dom_e1[i], dom_e2[j])
            intersections[i,j,0] = inter[0]
            intersections[i,j,1] = inter[1]
    return intersections
    



@jit(boolean(float64[:,:], float64[:,:], float64[:], float64[:]), 
     nopython=True)
def intersect_pol_pol_bool(pts1, pts2, dom1, dom2):
    """
    TODO
    """
    # Cheap domain check
    if not overlaps_domain(dom1, dom2):
        return False
    
    # Get polygon edges 
    n_e1, pts_e1, vec_e1, dom_e1 = get_edge_data(pts1)
    n_e2, pts_e2, vec_e2, dom_e2 = get_edge_data(pts2)
    
    # Loop over edges for both polygons and check edge intersections
    for i in range(n_e1):
        for j in range(n_e2):
            inter = intersect_line_line(
                pts_e1[i], pts_e2[j], 
                vec_e1[i], vec_e2[j], 
                dom_e1[i], dom_e2[j])
            if np.isnan(inter).sum() > 0:
                return True
    return False


@jit(boolean(float64[:,:], float64[:], float64[:]), nopython=True)
def contains_point(pts_pol, dom_pol, xy):
    """
    Returns true if a polygon contains the point xy. Sums up the
    angles of vector pairs from point xy to consecutive pairs of vertices 
    from the polygon exterior. If the angles sum up to 2 pi, the point is
    contained. Source:
    https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
    """
    # Cheap domain check
    dom_xy = np.array((xy[0], xy[0], xy[1], xy[1]))
    if not overlaps_domain(dom_pol, dom_xy):
        return False
    
    # Get polygon edges and loop over edges
    vec = xy - pts_pol
    norm = (vec[:,0]**2 + vec[:,1]**2)**0.5
    rad_sum = 0
    
    # Get edge pairs
    n = len(pts_pol)
    idx = list(range(n))
    pairs_idx = zip(idx, idx[1:] + [idx[0]])
    
    for i, j in pairs_idx:
        inner = vec[i][0] * vec[j][0] + vec[i][1] * vec[j][1]
        norms = norm[i] * norm[j]
        
        # If norms equals zero, point lies on polygon vertex
        if norms == 0:
            return False
        
        # Compute angle in radians and add to rad_sum
        ratio = inner / norms
        if ratio < -1:
            ratio = -1
        elif ratio > 1:
            ratio = 1
        rad_sum += np.arccos(ratio)
    return 1.999999999999*math.pi < rad_sum < 2.000000000001*math.pi


class GeometryBase:

    def __init__(self, points):
        """
        Points must be an array or list with shape (n,2). The last point in 
        the list is linked to the first one to create a closed loop.
        """
        self.set_points(points)

    def __repr__(self):
        shape_name = self.__class__.__name__
        str_points  = ','.join([f'({x},{y})' for (x,y) in self.points])
        return f'{shape_name} [{str_points}]'

    def set_points(self, points):
        """
        Set or adjust the geometry points.
        """
        self.points = np.array(points, dtype=float)
        x, y = self.points.T
        self.domain = np.array((min(x), max(x), min(y), max(y)))

    def copy(self):
        """
        Return a copy of the geometry instance.
        """
        return self.__class__(self.points)

    def center(self):
        """
        Returns the average (x,y) of all polygon points.
        """
        return self.points.mean(axis=0)
   
    def radius(self, xy):
        """
        Returns the radius of a circle centered around a point that 
        completely surrounds the geometry.
        """
        distances = np.linalg.norm(xy - self.points, axis=1)
        return max(distances)
   
    def translate(self, delta_xy):
        """
        Translate the polygon points in a given (x,y) direction.
        """
        self.set_points(translate(self.points, *delta_xy))
    
    def rotate(self, delta_angle, center):
        """ 
        Rotate the geometry by delta_angle radians around a center point.
        """
        self.set_points(rotate(self.points, delta_angle, *center))
    
    def overlaps_domain(self, other_geom):
        """
        Returns True if there is some overlap in self.domain and other_geom.domain.
        """
        return overlaps_domain(self.domain, other_geom.domain)



class Line(GeometryBase):
    
    def set_points(self, points):
        super().set_points(points)
        self.vec = self.points[1] - self.points[0]
        
    def intersect_line(self, other_line):
        """
        Find the intersection points between self and another line.
        """
        intersection = intersect_line_line(
            self.points, other_line.points, self.vec, other_line.vec,
            self.domain, other_line.domain)
        return intersection[~np.isnan(intersection).any()].flatten()
    
    def intersect_pol(self, polygon):
        """
        Find the intersection points between self and a polygon exterior.
        """
        intersections = intersect_line_pol(
            self.points, polygon.points, self.domain, polygon.domain, self.vec)
        return intersections[~np.isnan(intersections).any(axis=1)]
        


class Polygon(GeometryBase):
    
    def set_points(self, points):
        super().set_points(points)

    def intersect_line(self, line):
        """
        Find the intersection points between polygon exterior and a line.
        """
        intersections = intersect_line_pol(
            line.points, self.points, line.domain, self.domain, line.vec)
        return intersections[~np.isnan(intersections).any(axis=1)]

    def intersect_pol(self, other_polygon):
        """
        Find the intersection points between self and other polygon exteriors.
        """
        intersections = intersect_pol_pol(
            self.points, other_polygon.points, self.domain, 
            other_polygon.domain)
        intersections = intersections.reshape(-1,2)
        return intersections[~np.isnan(intersections).any(axis=1).flatten()]
    
    def intersect_pol_bool(self, other_polygon):
        """
        Returns True if at least one intersection between polygons exists.
        """
        return intersect_pol_pol_bool(
            self.points, other_polygon.points, self.domain, 
            other_polygon.domain)
    
    def contains_point(self, xy):
        """
        Returns true if polygon contains the point (x,y). 
        """
        return contains_point(self.points, self.domain, xy)









