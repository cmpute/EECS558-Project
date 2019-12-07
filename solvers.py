import numpy as np
import scipy.spatial as sps
from scipy.sparse import csc_matrix
from shapely.geometry import Point, MultiPoint, LineString, mapping
from shapely.ops import unary_union, nearest_points
from tqdm import trange
from easydict import EasyDict as edict
from matplotlib import patches as patches

class BaseSolver:
    '''
    Solver base, see below for what you need to implement
    '''
    def __init__(self):
        pass

    def solve(self, env, max_steps=50):
        '''
        This function solves the problem given certain environment
        '''
        raise NotImplementedError("Derived class should implement this function")

    def report_solution(self):
        '''
        This function report the solution in form of a chain of positions
        '''
        raise NotImplementedError("Derived class should implement this function")

    def action(self, state, step):
        '''
        After solving a environment, generate action for given state based on solution strategy.
        The action return the movement to next target point.
        '''
        raise NotImplementedError("Derived class should implement this function")

    def render(self, ax):
        '''
        Render debug elements onto a figure, this is for debugging
        '''
        pass

default_settings = dict(
    goal_reward=10000,
    goal_dist_thres=0.1,
    collision_cost=-1000,
    safety_weight=1,
    time_weight=1,
)

class GridSolver(BaseSolver):
    def __init__(self, grid_size=10):
        self._grid_size = grid_size
        self._grid = None
        self._grid_ticks_x = None
        self._grid_ticks_y = None
        self._grid_length_x = 0
        self._grid_length_y = 0
        self._start_xy = None
        self._end_xy = None

    def solve(self, env, max_steps=200, **settings):
        configs = edict(default_settings)
        configs.update(settings)

        print("Preparing mesh...")
        self._grid = np.full((self._grid_size, self._grid_size), False)
        self._grid_ticks_x = np.linspace(env._area[0], env._area[2], self._grid_size + 1)
        self._grid_ticks_y = np.linspace(env._area[1], env._area[3], self._grid_size + 1)
        self._grid_length_x = (env._area[2] - env._area[0]) / float(self._grid_size)
        self._grid_length_y = (env._area[3] - env._area[1]) / float(self._grid_size)
        self._start_xy = [np.searchsorted(self._grid_ticks_x, env._start.x)-1, np.searchsorted(self._grid_ticks_y, env._start.y)-1]
        self._end_xy = [np.searchsorted(self._grid_ticks_x, env._end.x)-1, np.searchsorted(self._grid_ticks_y, env._end.y)-1]

        print("Preparing matrices")
        for ob in env.obstacles:
            minx, miny, maxx, maxy = ob.bounds
            iminx = np.clip(np.searchsorted(self._grid_ticks_x, minx), 1, self._grid_size) - 1
            iminy = np.clip(np.searchsorted(self._grid_ticks_y, miny), 1, self._grid_size) - 1
            imaxx = np.clip(np.searchsorted(self._grid_ticks_x, maxx), 1, self._grid_size)
            imaxy = np.clip(np.searchsorted(self._grid_ticks_y, maxy), 1, self._grid_size)
            self._grid[iminx:imaxx, iminy:imaxy] = True

        goal_array = np.zeros((self._grid_size, self._grid_size))
        goal_array[self._end_xy[0], self._end_xy[1]] = configs.goal_reward
        safety_cost = np.array([[1/(env.obstacles.distance(Point(
                self._grid_ticks_x[j] + self._grid_length_x/2,
                self._grid_ticks_y[i] + self._grid_length_y/2)) + 1e-8)
            for i in range(self._grid_size)]
            for j in range(self._grid_size)]
        ) * configs.safety_weight

        values = np.full((self._grid_size + 2, self._grid_size + 2), -np.inf) # two more rows and columns as sentinels
        values[1:-1, 1:-1] = goal_array
        best_actions = []
        for _ in trange(max_steps, desc="Backward induction..."):
            # actions are up(0), right(1), down(2), left(3)
            values[1:-1, 2:][self._grid] = -np.inf # you cannot go to blocked area
            best_n2 = np.argmax([ # center block
                values[1:-1, 2:], # up
                values[2:, 1:-1], # right
                values[1:-1, :-2], # down
                values[:-2, 1:-1], # left
            ], axis=0)
            best_actions.append(best_n2)

            new_values = np.full((self._grid_size + 2, self._grid_size + 2), -np.inf)
            new_values[1:-1, 1:-1] = -(safety_cost + configs.time_weight)
            new_values[1:-1, 1:-1][best_n2 == 0] += values[1:-1, 2:][best_n2 == 0]
            new_values[1:-1, 1:-1][best_n2 == 1] += values[2:, 1:-1][best_n2 == 1]
            new_values[1:-1, 1:-1][best_n2 == 2] += values[1:-1, :-2][best_n2 == 2]
            new_values[1:-1, 1:-1][best_n2 == 3] += values[:-2, 1:-1][best_n2 == 3]
            values = new_values

        self._solution = np.array(list(reversed(best_actions)))
        if not np.all(values[1:-1, 1:-1][self._grid] >= 0):
            print("!!! No feasible solution found in given steps !!!")

    def report_solution(self, start_xy=None):
        loc = start_xy or self._start_xy
        loc_list = []
        for sol_t in self._solution:
            action = sol_t[loc[0], loc[1]]
            if action == 0:
                loc[1] += 1
            elif action == 1:
                loc[0] += 1
            elif action == 2:
                loc[1] -= 1
            elif action == 3:
                loc[0] -= 1

            loc_list.append(list(loc))
            if loc == self._end_xy:
                break
        return loc_list

    def action(self, state, step):
        # TODO!!: there're problems on edge of the grid
        ix = np.clip(np.searchsorted(self._grid_ticks_x, state[0]), 1, self._grid_size) - 1
        iy = np.clip(np.searchsorted(self._grid_ticks_y, state[1]), 1, self._grid_size) - 1

        step = step % len(self._solution) # XXX: what to do if need more steps
        action = self._solution[0][ix, iy]
        if action == 0:
            return 0, self._grid_length_y
        elif action == 1:
            return self._grid_length_x, 0
        elif action == 2:
            return 0, -self._grid_length_y
        elif action == 3:
            return -self._grid_length_x, 0

    def render(self, ax):
        # draw grids
        for x in self._grid_ticks_x:
            ax.axvline(x, ls='--')
        for y in self._grid_ticks_y:
            ax.axhline(y, ls='--')

        # draw grid values
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                if self._grid[i,j]:
                    rect = patches.Rectangle((self._grid_ticks_x[i], self._grid_ticks_y[j]),
                        self._grid_length_x, self._grid_length_y, color="#f1a20888")
                    ax.add_patch(rect)

        # draw solution
        solution = self.report_solution()
        for i in range(len(solution) - 1):
            x1 = self._grid_ticks_x[solution[i][0]] + self._grid_length_x/2
            y1 = self._grid_ticks_y[solution[i][1]] + self._grid_length_y/2
            x2 = self._grid_ticks_x[solution[i+1][0]] + self._grid_length_x/2
            y2 = self._grid_ticks_y[solution[i+1][1]] + self._grid_length_y/2
            ax.plot([x1, x2], [y1, y2], lw=4, c='green')

class SampleGraphSolver(BaseSolver):
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
        self._connections = line_list

    def solve(self, env, max_steps=50, early_stop=True, **settings):
        configs = edict(default_settings)
        configs.update(settings)

        print("Preparing mesh...")
        self._generate_mesh(env)

        print("Preparing matrices...")
        dist_list = [configs.time_weight * self._samples.geoms[n1].distance(self._samples.geoms[n2]) for n1, n2 in self._connections] * 2
        connection_list = self._connections + [(n2, n1) for n1, n2 in self._connections]
        adj_matrix = csc_matrix((dist_list, zip(*connection_list)), shape=(self._sample_num, self._sample_num))
        # point_array = np.array(mapping(self._samples)['coordinates'])
        safety_cost = np.array([1/env.obstacles.distance(p) for p in self._samples]) * configs.safety_weight
        goal_array = np.zeros(self._sample_num)
        goal_array[0] = configs.goal_reward # In backward induction, we require exact arrival

        print("Connectivity check...")
        stack = [1] # start from intial point
        connect_flag = np.full(self._sample_num, False)
        while len(stack) > 0:
            node = stack.pop()
            for next_node in adj_matrix[node].nonzero()[1]:
                if not connect_flag[next_node]:
                    stack.append(next_node)
                    connect_flag[next_node] = True
        if not connect_flag[0]:
            raise RuntimeError("Initial point and end point is not connected! Consider add more samples")

        # backward induction
        values = np.copy(goal_array)
        best_actions = []
        # XXX: should this be max step limit or converge condition?
        for _ in trange(max_steps, desc="Backward induction..."):
            new_values = np.empty(self._sample_num)
            new_actions = np.empty(self._sample_num, dtype=int)
            for n1 in range(self._sample_num):
                mask = adj_matrix[n1].nonzero()[1]
                if len(mask) == 0:
                    continue

                rewards = values[mask]
                rewards += goal_array[n1] # goal reward
                rewards -= safety_cost[n1] # safety cost
                rewards -= adj_matrix[n1, mask].toarray().ravel() # distance cost
                
                best_n2 = np.argmax(rewards)
                new_actions[n1] = mask[best_n2] # store in forward direction
                new_values[n1] = rewards[best_n2]

            values = new_values
            best_actions.append(new_actions)
            if np.all(new_values[connect_flag] >= 0):
                if early_stop:
                    print("Early stopped")
                    break

        self._solution = np.array(list(reversed(best_actions)))
        if not np.all(new_values[connect_flag] >= 0):
            print("!!! No feasible solution found in given steps !!!")

    def report_solution(self, start_sample_index=1):
        node_list = [start_sample_index]
        for row in self._solution:
            node_list.append(row[node_list[-1]])
            if node_list[-1] == 0:
                break
        return node_list

    def render(self, ax):
        ax.scatter([p.x for p in self._samples], [p.y for p in self._samples])

        solution = self.report_solution()
        solution_set = set()
        for i in range(len(solution) - 1):
            solution_set.add((solution[i], solution[i+1]))

        for n1, n2 in self._connections:
            if (n1, n2) in solution_set or (n2, n1) in solution_set:
                color = "green"
                lwidth = 4
            else:
                color = "black"
                lwidth = 1
            ax.plot([self._samples[n1].x, self._samples[n2].x], [self._samples[n1].y, self._samples[n2].y], lw=lwidth, c=color)

    def action(self, state, step):
        dist = np.array([(state[0] - p.x, state[1] - p.y) for p in self._samples])
        nearest = np.argmin(np.linalg.norm(dist, axis=1))
        step = step % len(self._solution) # XXX: what to do if need more steps
        target = self._samples[self._solution[step, nearest]]
        return target.x - state[0], target.y - state[1]

class CellSolver(BaseSolver):
    '''
    This solver generate cells and find a path between the cells
    '''
    def __init__(self):
        pass
