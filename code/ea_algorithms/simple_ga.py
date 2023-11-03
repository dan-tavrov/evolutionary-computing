import time

import numpy as np
from IPython.core.display import clear_output

from ea_operators import initialize, phenotypes, selection, crossover, mutation
from ea_operators.fitness import compute_fitness, scale_fitness
from ea_utilities import objective_functions, visualize, save


# function that implements a simple GA:
# integer individuals are represented as binary strings
# objective function is x^n
# crossover is one-point
# mutation is bit-flip
# selection is fitness-proportionate (can be with linear scaling)
def simple_ga(mu, chromosome_length, n, prob_crossover, prob_mutation,
              generations_count=1000,  # maximum number of iterations
              do_draw_population=False,  # whether to draw the whole population
              do_draw_stats=False,  # whether to draw best and average fitnesses
              do_print=True,  # whether to print results of each generation
              do_save=False,  # whether to save each generation in a separate file
              do_scale=False,  # whether to perform fitness scaling
              suppress_output=False):  # whether to report the final information
    population = initialize.initialize_simple_ga(mu, chromosome_length)
    population_fig = None
    stat_fig = None
    both_figs = None

    function_evals = 0

    # fitness function
    objective_function = lambda x: \
        objective_functions.simple_objective_function(
            x, n, 2 ** chromosome_length - 1
        )

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
            compute_fitness(
                population, phenotypes.phenotype_simple_ga,
                objective_function
            )

        function_evals += mu

        # compute statistics
        fitness_bests[i], fitness_worsts[i], fitness_avgs[i], fitness_best_index = \
            np.max(fitnesses), np.min(fitnesses), np.mean(fitnesses), np.argmax(fitnesses)

        if fitness_bests[i] > current_best_fitness:
            current_best_fitness = fitness_bests[i]
            current_best_population = population
            current_best_individuals = individuals
            current_best_fitnesses = fitnesses

        if do_draw_population and do_draw_stats:
            # draw population and fitness stats
            both_figs = visualize.draw_population_and_fitness_stats(
                individuals, fitnesses, fitness_bests, fitness_avgs,
                i + 1, current_best_fitness,
                objective_function, 0, 2**chromosome_length - 1,
                both_figs
            )
        else:
            # draw population only
            if do_draw_population:
                population_fig = visualize.draw_population(
                    individuals, fitnesses, i + 1,
                    objective_function, 0, 2**chromosome_length - 1,
                    population_fig
                )

            # draw fitness stats only
            if do_draw_stats:
                stat_fig = visualize.draw_fitness_stats(
                    fitness_bests, fitness_avgs, i + 1, current_best_fitness,
                    stat_fig
                )

        # print info as needed
        if do_print:
            visualize.report_ea_progress(
                fitness_bests[i], fitness_worsts[i], fitness_avgs[i],
                individuals[fitness_best_index],
                i + 1,
                time.time() - start_time
            )

        if do_draw_population or do_draw_stats:
            # clear current figures
            clear_output(wait=True)

        # dump the population and its fitness values into a file
        if do_save:
            save.save_to_csv(population, fitnesses, i + 1)

        # if the maximum fitness is already 1, exit the program
        if np.allclose(fitness_bests[i], 1, atol=1e-9):
            if not suppress_output:
                print('Finished because the perfect solution has been found!')
            generation_final = i + 1
            success = True
            break

        # do evolutionary operators as such
        # scale fitnesses if needed
        if do_scale:
            fitnesses = scale_fitness(fitnesses)

        # select parents to mate
        parent_indices = selection.selection_roulette_wheel(fitnesses)

        # perform the crossover
        child1, child2 = crossover.crossover_one_point(
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
                         current_best_fitnesses,
                         generation_final)

    return current_best_population, current_best_individuals, current_best_fitnesses,\
           generation_final, success, function_evals
