import numpy as np
from matplotlib import pyplot as plt

from env import DrivingEnv
from solvers import GridSolver, SampleGraphSolver

def time_compare(seed=1234, min_sample=10, max_sample=50, count=10):
    sample_count = np.linspace(min_sample, max_sample, count).astype(int)
    grid_times = []
    graph_times = []
    for size in sample_count:
        env = DrivingEnv(15, random_seed=seed)
        solver = GridSolver(size)
        grid_times.append(solver.solve(env, max_steps=500))

        env = DrivingEnv(15, random_seed=seed)
        solver = SampleGraphSolver(size*size)
        graph_times.append(solver.solve(env, max_steps=500))

    plt.figure()
    plt.semilogy(sample_count, grid_times, label="Grid-based")
    plt.semilogy(sample_count, graph_times, label="Graph-based")
    plt.xlabel("Equivalent sample size")
    plt.ylabel("Running time (s)")
    plt.legend()
    plt.show()

def grid_size_reward_compare(seed=1234, min_sample=10, max_sample=50, count=10, repeat=5):
    env = DrivingEnv(15, random_seed=seed)
    size_list = np.linspace(min_sample, max_sample, count).astype(int)
    cost_list = []
    for size in size_list:
        cost_cases = []
        for _ in range(repeat):
            solver = SampleGraphSolver(size*size)
            solver.solve(env, max_steps=200, early_stop=False)
            states, cost = env.simulate(solver)
            cost_cases.append(cost)
        cost_list.append(cost_cases)

    plt.figure()
    plt.plot(size_list, np.mean(cost_list, axis=1))
    plt.xlabel("Graph size")
    plt.ylabel("Time and safety cost")
    plt.title("Graph based policy performance versus graph size")
    plt.show()

def grid_with_different_safety_cost(cost_type="linear"):
    env = DrivingEnv(15, random_seed=1234)
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

def graph_with_different_weight(seed=1234, ratio_count=7):
    ratios = np.logspace(-3, 3, ratio_count)

    fig, ax = plt.subplots(1)
    DrivingEnv(15, random_seed=seed).render(ax)

    handles = [None] * ratio_count
    for rid, ratio in enumerate(ratios):
        coeff = np.sqrt(ratio)
        env = DrivingEnv(15, random_seed=seed)
        solver = SampleGraphSolver(800)
        solver.solve(env, max_steps=100, early_stop=False, safety_weight=coeff, time_weight=1/coeff, safety_type="linear")

        solution = solver.report_solution()
        solution_set = set()
        for i in range(len(solution) - 1):
            solution_set.add((solution[i], solution[i+1]))

        for n1, n2 in solver._connections:
            if (n1, n2) in solution_set or (n2, n1) in solution_set:
                lwidth, color = 4, "C%d" % rid
                handles[rid], = ax.plot([solver._samples[n1].x, solver._samples[n2].x], [solver._samples[n1].y, solver._samples[n2].y], lw=lwidth, c=color)

    # fig.legend(handles, ["safety/time=%f" % ratio for ratio in ratios], loc=1)
    plt.title("Difference path under different weights")
    plt.show()
        
graph_with_different_weight()
