import numpy as np
import scipy.spatial as sps
from scipy.sparse import csc_matrix
from shapely.geometry import Point, MultiPoint, LineString, mapping
from shapely.ops import unary_union, nearest_points
from tqdm import trange

class BaseSolver:
    '''
    Solver base, see below for what you need to implement
    '''
    def __init__(self):
        pass

    def solve(self, env):
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

class GridSolver(BaseSolver):
    def __init__(self):
        pass

    def solve(self, env):
        pass

    def report_solution(self):
        pass

    def action(self, state, step):
        pass

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

    def solve(self, env, max_steps=50, goal_reward=1000, safety_weight=1, time_weight=1):

        print("Preparing mesh...")
        self._generate_mesh(env)

        print("Preparing matrices...")
        dist_list = [time_weight * self._samples.geoms[n1].distance(self._samples.geoms[n2]) for n1, n2 in self._connections] * 2
        connection_list = self._connections + [(n2, n1) for n1, n2 in self._connections]
        adj_matrix = csc_matrix((dist_list, zip(*connection_list)), shape=(self._sample_num, self._sample_num))
        # point_array = np.array(mapping(self._samples)['coordinates'])
        safety_reward = np.array([env.obstacles.distance(p) for p in self._samples]) * safety_weight
        goal_array = np.zeros(self._sample_num)
        goal_array[0] = goal_reward

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
        for _ in trange(max_steps, desc="Backward induction..."):
            new_values = np.empty(self._sample_num)
            new_actions = np.empty(self._sample_num, dtype=int)
            for n1 in range(self._sample_num):
                mask = adj_matrix[n1].nonzero()[1]
                if len(mask) == 0:
                    continue

                rewards = values[mask]
                rewards += goal_array[mask] # goal reward
                rewards += safety_reward[mask] # safety reward
                rewards -= adj_matrix[n1, mask].toarray().ravel() # distance cost
                
                best_n2 = np.argmax(rewards)
                new_actions[n1] = mask[best_n2] # store in forward direction
                new_values[n1] = rewards[best_n2]

            values = new_values
            best_actions.append(new_actions)
            if np.all(new_values[connect_flag] >= goal_reward):
                print("Early stopped")
                break

        self._solution = np.array(list(reversed(best_actions)))
        if not np.all(new_values[connect_flag] >= goal_reward):
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
