from env import DrivingEnv

from matplotlib import pyplot as plt

env = DrivingEnv()
fig, ax = plt.subplots(1)
env.render(ax)
plt.show()
