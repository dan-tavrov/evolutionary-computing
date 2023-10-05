import time

import numpy as np
from IPython.core.display import clear_output

from ea_operators import initialize, phenotypes, selection, crossover, mutation
from ea_operators.fitness import calculate_fitness, scale_fitness
from ea_utilities import objective_functions, visualize, save


# function that implements a GA for parameter optimization:
# real individuals are represented as binary strings
# crossover can be either n-point or uniform
# mutation is bit-flip
# various selection algorithms supported
def simple_ga(objective_function,
              mu,
              nvar,  # number of parameters of the objective function
              prob_crossover, prob_mutation,
              a=-5.12, b=5.12,  # boundaries of an interval to perform search on
              bits_num=32,  # number of bits per one parameter
              crossover_function = crossover.crossover_one_point,
              generations_count=1000,  # maximum number of iterations
              optimum_value=0,  # optimum value of a function, if known in advance
              precision=1e-8,  # precision for the final solution
              do_minimize=True,  # if we need to convert a maximization problem into a minimization one
              do_draw_stats=False,  # whether to draw best and average fitnesses
              do_print=True,  # whether to print results of each generation
              do_save=False,  # whether to save each generation in a separate file
              do_scale=True,  # whether to perform fitness scaling
              suppress_output=False):  # whether to report the final information
    population = initialize.initialize_real(mu, nvar, bits_num)
    stat_fig = None

    fitness_bests = np.zeros(generations_count)
    fitness_worsts = np.zeros(generations_count)
    fitness_avgs = np.zeros(generations_count)

    current_best_fitness = 0
    current_best_population = None
    current_best_individuals = None
    current_best_fitnesses = None
    generation_final = generations_count

    # start recording the time
    start_time = time.time()

    success = False
    for i in range(generations_count):
        # calculate fitness values
        (individuals, fitnesses) = \
            calculate_fitness(
                population,
                lambda x: phenotypes.phenotype_real(
                    x, a, b, nvar, bits_num
                ),
                objective_function
            )

        # compute statistics
        if do_minimize:
            fitness_bests[i], fitness_worsts[i], fitness_avgs[i] = \
                np.min(fitnesses), np.max(fitnesses), np.mean(fitnesses)
        else:
            fitness_bests[i], fitness_worsts[i], fitness_avgs[i] = \
                np.max(fitnesses), np.min(fitnesses), np.mean(fitnesses)

        norm_diff_to_optimum = np.linalg.norm(fitness_bests[i] - optimum_value)

        if (do_minimize and (current_best_fitness < fitness_bests[i]))\
                or \
           ((not do_minimize) and (current_best_fitness > fitness_bests[i])):
            current_best_fitness = fitness_bests[i]
            current_best_population = population
            current_best_individuals = individuals
            current_best_fitnesses = fitnesses

        if do_draw_stats:
            stat_fig = visualize.draw_fitness_stats(
                fitness_bests, fitness_avgs, i + 1, current_best_fitness,
                stat_fig
            )

        # print info as needed
        if do_print:
            visualize.report_ga_progress(
                fitness_bests[i], fitness_worsts[i], fitness_avgs[i], i + 1,
                time.time() - start_time,
                norm_diff_to_optimum
            )

        if do_draw_stats:
            # clear current figures
            clear_output(wait=True)

        # dump the population and its fitness values into a file
        if do_save:
            save.save_to_csv(population, individuals, fitnesses, i + 1)

        # if the solution has already been found, exit the program
        if norm_diff_to_optimum <= precision:
            if not suppress_output:
                print('Finished because the perfect solution has been found!')
            generation_final = i + 1
            success = True
            break

        # do evolutionary operators as such
        # convert fitnesses to "all non-negative and waiting to be maximized"
        if do_minimize:
            fitnesses = current_best_fitness - fitnesses
        else:
            fitnesses = fitnesses - current_best_fitness
        fitnesses = np.where(fitnesses < 0, 0, fitnesses)

        # scale fitnesses if needed
        if do_scale:
            fitnesses = scale_fitness(fitnesses)

        # select parents to mate
        parent_indices = selection.selection_roulette_wheel(fitnesses)

        # perform the crossover
        child1, child2 = crossover_function(
            population[parent_indices[0::2]],
            population[parent_indices[1::2]],
            prob_crossover=prob_crossover
        )

        # mutate the offspring
        child1 = mutation.mutation_bit_flip(
            child1, prob_mutation=prob_mutation
        )

        child2 = mutation.mutation_bit_flip(
            child2, prob_mutation=prob_mutation
        )

        population[0::2, ] = child1.copy()
        population[1::2, ] = child2.copy()

    if not suppress_output:
        print(f'The final generation was {generation_final}.')

        # save the best results
        save.save_to_csv(current_best_population,
                         current_best_individuals,
                         current_best_fitnesses,
                         generation_final)

    return current_best_population, current_best_individuals, current_best_fitnesses, generation_final, success
