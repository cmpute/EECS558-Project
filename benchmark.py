import numpy as np
from matplotlib import pyplot as plt

from env import DrivingEnv
from solvers import GridSolver, SampleGraphSolver

def time_compare(env, min_sample=10, max_sample=50, count=10):
    sample_count = np.linspace(min_sample, max_sample, count).astype(int)
    grid_times = []
    graph_times = []
    for size in sample_count:
        solver = GridSolver(size)
        grid_times.append(solver.solve(env, max_steps=500))

        solver = SampleGraphSolver(size*size)
        graph_times.append(solver.solve(env, max_steps=500))

    plt.figure()
    plt.semilogy(sample_count, grid_times, label="Grid-based")
    plt.semilogy(sample_count, graph_times, label="Graph-based")
    plt.xlabel("Equivalent sample size")
    plt.ylabel("Running time (s)")
    plt.legend()
    plt.show()

def grid_size_reward_compare(env, min_sample=10, max_sample=50, count=10, repeat=5):
    size_list = np.linspace(min_sample, max_sample, count).astype(int)
    cost_list = []
    for size in size_list:
        cost_cases = []
        for _ in range(repeat):
            solver = GridSolver(size)
            solver.solve(env, max_steps=500, early_stop=False)
            states, cost = env.simulate(solver)
            cost_cases.append(cost)
        cost_list.append(cost_cases)

    plt.figure()
    plt.plot(size_list, np.mean(cost_list, axis=1))
    plt.show()

def grid_with_different_safety_cost(env, cost_type="linear"):
    def render_graph(solver, ax):
        solution = solver.report_solution()
        solution_set = set()
        for i in range(len(solution) - 1):
            solution_set.add((solution[i], solution[i+1]))

        for n1, n2 in solver._connections:
            if (n1, n2) in solution_set or (n2, n1) in solution_set:
                color = "#1A090D"
                lwidth = 5
            else:
                color = "#4A139488"
                lwidth = 1
            ax.plot([solver._samples[n1].x, solver._samples[n2].x], [solver._samples[n1].y, solver._samples[n2].y], lw=lwidth, c=color)

        ax.scatter([p.x for p in solver._samples], [p.y for p in solver._samples], c=solver._safety_cost_cache)

    solver = SampleGraphSolver(800)
    solver.solve(env, max_steps=200, safety_weight=100, safety_type=cost_type)

    fig, ax = plt.subplots(1)
    env.render(ax)
    render_graph(solver, ax)
    plt.title("Graph-based solution with %s cost" % cost_type)
    plt.show()

env = DrivingEnv(15, random_seed=1234)
grid_with_different_safety_cost(env, cost_type="tanh")
