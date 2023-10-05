import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import display


def draw_population(individuals, fitnesses, generation, fig_handles=None):
    if generation == 1:
        # if this is the first time we create this plot, we need to initialize it first
        fig, ax = plt.subplots()

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
                                      current_max, fig_handles=None):
    if generation == 1:
        # if this is the first time we create this plot, we need to initialize it first
        fig, (ax_pop, ax_stat) = plt.subplots(1, 2, figsize=(10, 4))

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


def report_ga_progress(fitness_best, fitness_worst, fitness_avg, generation, elapsed_time,
                       norm_diff_to_optimum=None):
    print('GENERATION {}'.format(generation))

    minutes = np.floor(elapsed_time / 60)
    seconds = elapsed_time - minutes * 60
    print('Time elapsed: {} minutes(s) and {} second(s).'.format(minutes, seconds))

    print('Average fitness value --- {:.3f}'.format(fitness_avg))
    print('Best fitness value --- {:.3f}'.format(fitness_best))
    print('Worst fitness value --- {:.3f}'.format(fitness_worst))

    if norm_diff_to_optimum is not None:
        print('Norm of the difference with the optimum value --- {:.3f'.format(norm_diff_to_optimum))

    print('-'*15)