import numpy as np
from matplotlib import pyplot as plt, patches as patches
from shapely.geometry import box, Point, MultiPolygon

class DrivingEnv:
    def __init__(self, obstacle_num=10, random_seed=None):
        self._start = Point(1,1) # start point
        self._end = Point(9,9) # end point
        self._area = [0,0,10,10] # drivable area x0, y0, x1, y1

        self._random_state = np.random.RandomState(random_seed)
        self._obstacles = self._generate_obstacles(obstacle_num)

    def _generate_points(self, count=1):
        x = self._random_state.random_sample(count)*(self._area[2] - self._area[0]) + self._area[0]
        y = self._random_state.random_sample(count)*(self._area[3] - self._area[1]) + self._area[1]

        return np.array([x, y]).T

    def _generate_obstacles(self, num=10):
        # FIXME: currently no feasibility check
        obs = []
        max_ratio = 0.4
        while num > 0:
            ob_xy = self._generate_points()[0]
            ob_wl = [
                self._random_state.random_sample()*(self._area[2] - self._area[0]) * max_ratio,
                self._random_state.random_sample()*(self._area[3] - self._area[1]) * max_ratio
            ]
            ob = box(ob_xy[0] - ob_wl[0]/2., ob_xy[1] - ob_wl[1]/2., ob_xy[0] + ob_wl[0]/2., ob_xy[1] + ob_wl[1]/2.)
            if not ob.within(self._start) and not ob.within(self._end):
                obs.append(ob)
                num -= 1
        
        return MultiPolygon(obs)

    @property
    def obstacles(self):
        return self._obstacles

    def render(self, ax):
        '''
        Draw the enviroment into matplotlib figure ax.
        '''
        for ob in self._obstacles.geoms:
            poly_vis = patches.Polygon(np.array(ob.exterior.xy).T,
                linewidth=1, edgecolor='r', facecolor='#913D88AA')
            ax.add_patch(poly_vis)

        init = patches.Circle([self._start.x, self._start.y], radius=.1, fc='red', ec='black')
        goal = patches.Circle([self._end.x, self._end.y], radius=.1, fc='green', ec='black')
        ax.add_patch(init)
        ax.add_patch(goal)

        ax.set_xlim([self._area[0], self._area[2]])
        ax.set_ylim([self._area[1], self._area[3]])

    def solution_cost(self, solution):
        raise NotImplementedError()
