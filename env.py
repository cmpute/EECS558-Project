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

    def simulate(self, solver, ax=None, arrive_thres=0.5, motion_noise=0.01, observation_noise=0.1, obstacle_noise=0.1, time_weight=1, safety_weight=1):
        '''
        arrive_thres: we consider the vehicle to be arrived at the target if the ddistance bewteen 
        motion_noise: proportional to moving distance
        observation_noise: noise for ego location observation
        obstacle_noise: noise for reward process
        ax: if provided with matplotlib ax, we will plot the transient into the figure
        '''

        max_steps = 100
        states = [np.array([self._start.x, self._start.y])]
        end = np.array([self._end.x, self._end.y])
        cost = 0.0
        for step in range(max_steps):
            obsrv = states[-1] + np.random.normal(scale=observation_noise, size=2)
            action = np.asarray(solver.action(obsrv, step))
            new_state = states[-1] + action + np.random.normal(scale=motion_noise*np.linalg.norm(action - states[-1]), size=2)

            # calculate cost
            cost += np.linalg.norm(action) * time_weight
            cost += self._obstacles.distance(Point(new_state)) * safety_weight
            # TODO: add cost of crash and cost randomness

            # update state and plot
            ax.plot([states[-1][0], new_state[0]], [states[-1][1], new_state[1]], lw=4, c="blue")
            states.append(new_state)

            if np.linalg.norm(states[-1] - end) < arrive_thres:
                print("Navigate succeed")
                break

        if np.linalg.norm(states[-1] - end) >= arrive_thres:
            print("!!! Navigation failed !!!")

        print("Final cost: %f" % cost)
        return states, cost
