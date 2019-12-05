import numpy as np
import scipy.spatial as sps
from scipy.sparse import csc_matrix
from shapely.geometry import Point, MultiPoint, LineString
from shapely.ops import unary_union

class GridSolver:
    def __init__(self):
        pass

    def solve(self, env):
        '''
        This function solves the problem given certain environment
        '''
        pass

    def report_solution(self):
        '''
        This function report the solution in form of a chain of positions and time stamp
        '''
        pass

class SampleGraphSolver:
    '''
    This solver generate random samples and connect them together into a graph with constraint check
    '''
    def __init__(self, sample_num=100):
        self._sample_num = sample_num
        self._samples = None
        self._connections = None
        self._solution = None # store solution for a certain case

    def _generate_mesh(self, env):
        # generate nodes, first node is the goal
        points = env._generate_points(self._sample_num - 2)
        points = np.concatenate(([[env._end.x, env._end.y], [env._start.x, env._start.y]], points))
        
        while True:
            dist_list = np.array([Point(xy[0], xy[1]).distance(env.obstacles) for xy in points])
            collision_mask = dist_list < 1e-5
            collision_count = np.sum(collision_mask)
            if collision_count == 0:
                break

            # resample
            points[collision_mask] = env._generate_points(collision_count)
        self._samples = MultiPoint(points)

        # generate triangles
        tesselation = sps.Delaunay(points)
        triangles = tesselation.simplices.copy()
        edges = set()
        for tri in triangles:
            sortnodes = np.sort(tri)
            edges.add((sortnodes[0], sortnodes[1]))
            edges.add((sortnodes[1], sortnodes[2]))
            edges.add((sortnodes[0], sortnodes[2]))

        line_list = []
        obstacle_union = unary_union(env.obstacles)
        for n1, n2 in edges:
            line = LineString([self._samples[n1], self._samples[n2]])
            if line.intersection(obstacle_union).is_empty:
                line_list.append((n1, n2))
                line_list.append((n2, n1))
        self._connections = line_list

    def solve(self, env, steps=50):
        '''
        Steps that the algorithm runs to find the value function
        '''
        GOAL_REWARD = 1000

        self._generate_mesh(env)

        dist_list = [self._samples.geoms[n1].distance(self._samples.geoms[n2]) for n1, n2 in self._connections]
        adj_matrix = csc_matrix((dist_list, zip(*self._connections)), shape=(self._sample_num, self._sample_num))

        values = np.zeros(self._sample_num)
        values[0] = GOAL_REWARD
        best_actions = np.empty((steps, self._sample_num))
        for i in range(steps):
            pass # TODO: update value matrix 

    def report_solution(self):
        pass

    def render(self, ax):
        ax.scatter([p.x for p in self._samples], [p.y for p in self._samples])
        for n1, n2 in self._connections:
            ax.plot([self._samples[n1].x, self._samples[n2].x], [self._samples[n1].y, self._samples[n2].y], c='black')

class CellSolver:
    '''
    This solver generate cells and find a path between the cells
    '''
    def __init__(self):
        pass

    def solve(self, env):
        pass
