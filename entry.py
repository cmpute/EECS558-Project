from matplotlib import pyplot as plt

from env import DrivingEnv
from solvers import SampleGraphSolver

env = DrivingEnv(15, random_seed=1)
solver = SampleGraphSolver(1000)
solver.solve(env, safety_weight=20)
# print(solver.report_solution())

fig, ax = plt.subplots(1)
env.render(ax)
solver.render(ax)
plt.show()
