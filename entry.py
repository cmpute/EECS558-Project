from matplotlib import pyplot as plt

from env import DrivingEnv
from solvers import GridSolver, SampleGraphSolver

env = DrivingEnv(15, random_seed=1234)
solver = GridSolver(21)
# solver = SampleGraphSolver(800)
solver.solve(env, max_steps=500, safety_weight=10, goal_dist_thres=0.01)

fig, ax = plt.subplots(1)
env.render(ax)
solver.render(ax)
env.simulate(solver, ax)
plt.show()
