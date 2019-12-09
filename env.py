import numpy as np
from matplotlib import pyplot as plt, patches as patches
from shapely.geometry import box, Point, MultiPolygon
from solvers import default_settings
from easydict import EasyDict as edict

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
            if not self._start.within(ob) and not self._end.within(ob):
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

    def simulate(self, solver, ax=None, max_steps=1000, **settings):
        # default settings
        configs = edict(default_settings)
        configs.motion_noise = 0.02 # proportional to moving distance
        configs.observation_noise = 0.02 # noise for ego location observation
        configs.obstacle_noise = 0.1 # noise for reward process
        configs.update(settings)

        states = [np.array([self._start.x, self._start.y])]
        end = np.array([self._end.x, self._end.y])
        cost = 0.0
        for step in range(max_steps):
            obsrv = states[-1] + self._random_state.normal(scale=configs.observation_noise, size=2)
            action = np.asarray(solver.action(obsrv, step))
            new_state = states[-1] + action + self._random_state.normal(scale=configs.motion_noise*np.linalg.norm(action - states[-1]), size=2)

            # calculate cost
            cost -= np.linalg.norm(action) * configs.time_weight
            obs_distance = max(0.0, self._obstacles.distance(Point(new_state)) + self._random_state.normal(scale=configs.obstacle_noise))
            if obs_distance < 1e-3:
                cost -= 1000
            else:
                cost -= 1 / obs_distance * configs.safety_weight

            # update state and plot
            if ax:
                ax.plot([states[-1][0], new_state[0]], [states[-1][1], new_state[1]], lw=4, c="blue")
            states.append(new_state)

            if np.linalg.norm(states[-1] - end) < configs.goal_dist_thres:
                print("Navigate succeed")
                break

        if np.linalg.norm(states[-1] - end) >= configs.goal_dist_thres:
            print("!!! Navigation failed !!!")

        print("Final cost: %f" % cost)
        return states, cost
