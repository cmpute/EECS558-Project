import numpy as np
from matplotlib import pyplot as plt, patches as patches

class BoxObstacle:
    '''
    This class represent an box-shaped obstacle with center (x,y) and dimensions (length_x, length_y)
    '''
    def __init__(self, center, dimension):
        self._center = center # [x, y]
        self._dim = dimension # [length_x, length_y]

    def draw(self, ax):
        rect = patches.Rectangle(self._center, self._dim[0], self._dim[1],
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    def check_with_point(self, xy):
        return (abs(xy[0] - self._center[0]) <= self._dim[0]) and (abs(xy[1] - self._center[1]) <= self._dim[1])

class DrivingEnv:
    def __init__(self, obstacle_num=10):
        self._start = [0,0] # start point
        self._end = [10,10] # end point
        self._area = [0,0,10,10] # drivable area x0, y0, x1, y1
        self._obstacles = self._generate_obstacles(obstacle_num)

    def _generate_obstacles(self, num=10):
        obs = []
        max_ratio = 0.4
        while num > 0:
            ob = BoxObstacle([
                np.random.random()*(self._area[2] - self._area[0]) + self._area[0],
                np.random.random()*(self._area[3] - self._area[1]) + self._area[1],
            ], [
                np.random.random()*(self._area[2] - self._area[0]) * max_ratio,
                np.random.random()*(self._area[3] - self._area[1]) * max_ratio
            ])
            if not ob.check_with_point(self._end) and not ob.check_with_point(self._start):
                obs.append(ob)
                num -= 1
        
        return obs

    @property
    def obstacles(self):
        return self._obstacles

    def render(self, ax):
        for ob in self._obstacles:
            ob.draw(ax)
        ax.set_xlim([self._area[0], self._area[2]])
        ax.set_ylim([self._area[1], self._area[3]])
