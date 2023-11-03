import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import display


def draw_population(individuals, fitnesses, generation,
                    function, left, right, fig_handles=None):
    if generation == 1:
        # if this is the first time we create this plot, we need to initialize it first
        fig, ax = plt.subplots()

        # generate x values from 0 to 1
        x = np.linspace(left, right, np.round((right - left) * 100).astype(int))

        # draw the function itself
        ax.plot(x, function(x), color='black')

        # create a scatter plot
        scatter = ax.scatter(individuals, fitnesses)

        plt.tight_layout()
    else:
        fig, ax, scatter = fig_handles

        # update the current plot
        scatter.set_offsets(np.column_stack((individuals, fitnesses)))

        # redraw the whole plot
        fig.canvas.flush_events()

    # customize the plot as needed
    ax.set_xlabel('Individuals')
    ax.set_ylabel('Fitnesses')
    ax.set_title(f'Generation {generation}')

    # pause briefly to see the update (you can adjust the duration)
    time.sleep(0.1)

    display(fig)

    return fig, ax, scatter


def draw_fitness_stats(fitness_bests, fitness_avgs, generation, current_max, fig_handles=None):
    if generation == 1:
        # if this is the first time we create this plot, we need to initialize it first
        fig, ax = plt.subplots()

        # create a scatter plot
        scatter_best = ax.scatter(1, fitness_bests[0], color='red', label='Best fitness')
        scatter_avg = ax.scatter(1, fitness_avgs[0], color='blue', label='Average fitness')

        plt.tight_layout()
    else:
        fig, ax, scatter_best, scatter_avg = fig_handles

        # update the current plot
        ax.set_xlim(1 - 0.1, generation + 0.1)
        ax.set_ylim(-0.1, current_max + 0.1)

        scatter_best.set_offsets(np.column_stack((np.arange(generation) + 1, fitness_bests[:generation])))
        scatter_avg.set_offsets(np.column_stack((np.arange(generation) + 1, fitness_avgs[:generation])))

        # redraw the whole plot
        fig.canvas.flush_events()

    # customize the plot as needed
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()
    ax.set_title(f'Fitness stats for generation {generation}')

    # pause briefly to see the update (you can adjust the duration)
    time.sleep(0.1)

    display(fig)

    return fig, ax, scatter_best, scatter_avg


def draw_population_and_fitness_stats(individuals, fitnesses,
                                      fitness_bests, fitness_avgs, generation,
                                      current_max,
                                      function, left, right,
                                      fig_handles=None):
    if generation == 1:
        # if this is the first time we create this plot, we need to initialize it first
        fig, (ax_pop, ax_stat) = plt.subplots(1, 2, figsize=(10, 4))

        # generate x values from 0 to 1
        x = np.linspace(left, right, np.round((right - left) * 100).astype(int))

        # draw the function itself
        ax_pop.plot(x, function(x), color='black')

        # create scatter plots
        scatter_pop = ax_pop.scatter(individuals, fitnesses)
        scatter_best = ax_stat.scatter(1, fitness_bests[0], color='red', label='Best fitness')
        scatter_avg = ax_stat.scatter(1, fitness_avgs[0], color='blue', label='Average fitness')

        plt.tight_layout()
    else:
        fig, (ax_pop, ax_stat), scatter_pop, scatter_best, scatter_avg = fig_handles

        # update the current plot
        ax_stat.set_xlim(1 - 0.1, generation + 0.1)
        ax_stat.set_ylim(-0.1, current_max + 0.1)

        scatter_pop.set_offsets(np.column_stack((individuals, fitnesses)))
        scatter_best.set_offsets(np.column_stack((np.arange(generation) + 1, fitness_bests[:generation])))
        scatter_avg.set_offsets(np.column_stack((np.arange(generation) + 1, fitness_avgs[:generation])))

        # redraw the whole plot
        fig.canvas.flush_events()

    # customize the plot as needed
    ax_pop.set_xlabel('Individuals')
    ax_pop.set_ylabel('Fitnesses')
    ax_pop.set_title(f'Generation {generation}')

    ax_stat.set_xlabel('Generation')
    ax_stat.set_ylabel('Fitness')
    ax_stat.legend()
    ax_stat.set_title(f'Fitness stats for generation {generation}')

    # pause briefly to see the update (you can adjust the duration)
    time.sleep(0.1)

    display(fig)

    return fig, (ax_pop, ax_stat), scatter_pop, scatter_best, scatter_avg


def report_ea_progress(fitness_best, fitness_worst, fitness_avg, individual_best,
                       generation, elapsed_time,
                       norm_diff_to_optimum=None):
    print('GENERATION {}'.format(generation))

    minutes = np.floor(elapsed_time / 60)
    seconds = elapsed_time - minutes * 60
    print('Time elapsed: {} minutes(s) and {} second(s).'.format(minutes, seconds))

    print('Average fitness value --- {:.8f}'.format(fitness_avg))
    print('Best fitness value --- {:.8f}'.format(fitness_best))
    print('Worst fitness value --- {:.8f}'.format(fitness_worst))

    if norm_diff_to_optimum is not None:
        print('Norm of the difference with the optimum value --- {:.3f}'.format(norm_diff_to_optimum))

    print('Best individual --- ', ["{:.8f} ".format(x) for x in individual_best])

    print('-'*15)


def report_average_ea_progress(ea_function, T=10, random_seed=123,
                               print_iteration_number=False, do_minimize=True):
    best_fitnesses = np.zeros(T)
    generation_finals = np.zeros(T)
    successes = np.zeros(T)
    function_evals = np.zeros(T)

    np.random.seed(random_seed)

    for i in range(T):
        if print_iteration_number:
            print(f"Iteration number {i+1}")

        _, _, current_best_fitnesses,\
        generation_finals[i], successes[i], function_evals[i] = ea_function()
        if do_minimize:
            best_fitnesses[i] = np.min(current_best_fitnesses)
        else:
            best_fitnesses[i] = np.max(current_best_fitnesses)

    print(f"Total number of successes: {np.sum(successes)}")

    print("Mean function evaluations: {:.3f}".format(np.mean(function_evals)))
    print("Std of function evaluations: {:.3f}".format(np.std(function_evals, ddof=1)))

    print("Mean final generation: {:.3f}".format(np.mean(generation_finals)))
    print("Std of final generations: {:.3f}".format(np.std(generation_finals, ddof=1)))

    print("Mean of best fitnesses: {:.8f}".format(np.mean(best_fitnesses)))
    if do_minimize:
        print("The best fitness ever: {:.8f}".format(np.min(best_fitnesses)))
    else:
        print("The best fitness ever: {:.8f}".format(np.max(best_fitnesses)))
    print("")
