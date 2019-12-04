<!-- This document is for write down some thoughts or results that can be used for writing the report later -->

# Problem Definiton

Our target is to derive a path planning solver for a vehicle. To make the problem simpler without loss of generality, we define the vehicle as a point mass and navigate in 2D space. 3D navigation can be reduced to 2D easily, and the vehicle size can be considered by adding buffer area around the obstacles.

The problem for planning is usually defined as (The piano mover's problem):

1. Given a world W (in out case W=R^2)
2. Given a semi-algebraic obstacle region O \subset W
3. Given a semi-algebraic robot defined in W
4. Given a configuration space C, the set of all possible transformations taht may be applied to the robot(vehicle). Divided into C_obstacle and C_free
5. Qi \in C_free as initial configuration
6. Qg \in C_free as goal configuration
7. We want a algorithm to compute a (continuous) path \tau: [0,1]->C_free, s.t. tau(0) = q_I and tau(1) = q_G

Then the simplified stochastic control problem in our case is:

X: state of vehicle, usually (x, y)
U: action of vehicle, go to certain direction by certain distance. (Here we don't consider velocity continuity at first)
Y: observation of vehicle state with noise (localization noise)
C: cost, usually contains: 1. Action cost (like energy consumption?) 2. Time cost (constant for each time stamp) 3. Safety cost (distance to obstacles) 4. Terminal cost (obtained if we navigate to target successfully).

The cost structure and weights could be further modified and we will discuss about the planning performance with difference cost setup. For the timestep, the problem is obviouly a infinite horizon one, but since the distance between initial point and goal is limited, a good solution should also have limited time horizon, so we can limit the problem in certain time horizon w.r.t speed profile.

The major problems here we will met in vehicle path planning are:
1. Infinite state space and action space
2. Motion and observation uncertainty
3. Dynamic obstacles

In our project, we only focus on the first problem and explore the ways to solve planning by dynamic programming with some simplification.

# Grid solver

First we will try a very brief idea, where state space is rasterized into grid and action space is just walking along the grids.

# Sample graph solver

In this method, we first generate feasible samples randomly from C_free and connect them together into graph. Then state space is just the graph nodes and action space is move to adjacent node.

# Cell solver

In this method, we split the feasible area into cells and generate a route along the cells

# Approximate value function

In this method, we will try to solve the problem without discretization, but use approximation of value funtions.
