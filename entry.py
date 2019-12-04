from env import DrivingEnv

from matplotlib import pyplot as plt

env = DrivingEnv(random_seed=1234)
fig, ax = plt.subplots(1)
env.render(ax)
plt.show()
