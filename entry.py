from matplotlib import pyplot as plt

from env import DrivingEnv
from solvers import GridSolver, SampleGraphSolver

env = DrivingEnv(15, random_seed=1234)
# solver = GridSolver(50)
solver = SampleGraphSolver(800)
solver.solve(env, max_steps=200, early_stop=False, safety_weight=100, safety_type='tanh')

fig, ax = plt.subplots(1)
env.render(ax)
solver.render(ax)
env.simulate(solver, ax)
plt.show()
