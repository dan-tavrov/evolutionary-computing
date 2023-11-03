import random
import time

import numpy as np
from IPython.core.display import clear_output

from ea_operators import initialize, phenotypes, selection, crossover, mutation
from ea_operators.fitness import compute_fitness, scale_fitness, derate_fitness
from ea_utilities import visualize, save


# function that implements a GA for parameter optimization:
# real individuals are represented as binary strings
# crossover can be either n-point or uniform
# mutation is bit-flip
# various selection algorithms supported
def ga(objective_function,
       mu,
       nvar,  # number of parameters of the objective function
       prob_crossover, prob_mutation,
       a=-5.12, b=5.12,  # boundaries of an interval to perform search on
       bits_num=32,  # number of bits per one parameter
       crossover_function=crossover.crossover_one_point,  # tuple where the first
       # element is the crossover function name, and the second element is a DICTIONARY
       # with additional (named) arguments specific to a function
       # e.g., for n-point crossover, this could look like
       # (crossover.crossover_n_point, {"n": 3})
       selection_function=selection.selection_roulette_wheel,  # the same but for parent selection
       # in particular, setting (selection.selection_tournament, {"q": 5})
       # will give tournament selection with q = 5
       children_count=np.inf,  # how many individuals to replace in each generation
       replace_worst=False,  # whether replacement should be deterministic (all worst) or stochastic
       generations_count=1000,  # maximum number of iterations
       optimum_value=0,  # optimum value of a function, if known in advance
       precision=1e-8,  # precision for the final solution
       sigma_share=10,  # niche radius for sharing technique
       do_minimize=True,  # if we need to convert a maximization problem into a minimization one
       do_draw_population=False,  # whether to draw the whole population
       do_draw_stats=False,  # whether to draw best and average fitnesses
       do_print=True,  # whether to print results of each generation
       do_save=False,  # whether to save each generation in a separate file
       do_scale=True,  # whether to perform fitness scaling
       do_derate=False,  # whether to derate fitnesses using sharing
       suppress_output=False):  # whether to report the final information
    if mu % 2 != 0:
        # make sure our population is even-numbered
        mu -= 1
        if mu <= 0:
            return

    if children_count > mu:
        children_count = mu

    if children_count % 2 != 0:
        # make sure that we always select an even number of parents
        # otherwise would be hard to create pairs
        children_count -= 1
        if children_count <= 0:
            children_count = 2

    all_indices = list(range(mu))

    if isinstance(crossover_function, tuple) and len(crossover_function) > 1:
        crossover_function_name, crossover_function_args_dict = crossover_function
        crossover_function_args_dict["prob_crossover"] = prob_crossover
    else:
        crossover_function_name = crossover_function
        crossover_function_args_dict = {"prob_crossover": prob_crossover}

    if isinstance(selection_function, tuple) and len(selection_function) > 1:
        selection_function_name, selection_function_args_dict = selection_function
        selection_function_args_dict["children_count"] = children_count
    else:
        selection_function_name = selection_function
        selection_function_args_dict = {"children_count": children_count}

    function_evals = 0

    population = initialize.initialize_real(mu, nvar, bits_num)
    population_fig = None
    stat_fig = None

    fitness_bests = np.zeros(generations_count)
    fitness_worsts = np.zeros(generations_count)
    fitness_avgs = np.zeros(generations_count)

    if do_minimize:
        current_best_fitness = np.inf
        current_extreme_value = 0
    else:
        current_best_fitness = 0
        current_extreme_value = np.inf

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
                population,
                lambda x: phenotypes.phenotype_real(
                    x, a, b, nvar, bits_num
                ),
                objective_function
            )

        if i == 0:
            # we always evaluate all individuals in the first generation
            function_evals += mu
        else:
            # of course in this code, for other generations, we still evaluate everyone
            # doing otherwise would introduce unnecessary overhead
            # but if function evaluation is really costly, we really evaluate only new children
            function_evals += children_count

        # compute statistics
        if do_minimize:
            fitness_bests[i], fitness_worsts[i], fitness_avgs[i], fitness_best_index = \
                np.min(fitnesses), np.max(fitnesses), np.mean(fitnesses), np.argmin(fitnesses)
        else:
            fitness_bests[i], fitness_worsts[i], fitness_avgs[i], fitness_best_index = \
                np.max(fitnesses), np.min(fitnesses), np.mean(fitnesses), np.argmax(fitnesses)

        norm_diff_to_optimum = np.linalg.norm(fitness_bests[i] - optimum_value)

        if (do_minimize and (fitness_bests[i] < current_best_fitness))\
                or \
           ((not do_minimize) and (fitness_bests[i] > current_best_fitness)):
            current_best_fitness = fitness_bests[i]
            current_best_population = population
            current_best_individuals = individuals
            current_best_fitnesses = fitnesses

        if (do_minimize and (fitness_worsts[i] > current_extreme_value)) \
                or \
           ((not do_minimize) and (fitness_worsts[i] < current_extreme_value)):
            current_extreme_value = fitness_worsts[i]

        # draw population only
        if do_draw_population and nvar == 1:
            population_fig = visualize.draw_population(
                individuals, fitnesses, i + 1,
                objective_function, a, b,
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
                time.time() - start_time,
                norm_diff_to_optimum
            )

        if do_draw_population or do_draw_stats:
            # clear current figures
            clear_output(wait=True)

        # dump the population and its fitness values into a file
        if do_save:
            save.save_to_csv(population, fitnesses, i + 1)

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
            fitnesses = current_extreme_value - fitnesses
        else:
            fitnesses = fitnesses - current_extreme_value
        fitnesses = np.where(fitnesses < 0, 0, fitnesses)

        # derate fitnesses if needed
        if do_derate:
            fitnesses = derate_fitness(fitnesses, population, sigma_share)

        # scale fitnesses if needed
        if do_scale:
            fitnesses = scale_fitness(fitnesses)

        # select parents to mate
        # we select children_count parents
        parent_indices = selection_function_name(
            fitnesses,
            **selection_function_args_dict
        )

        # perform the crossover
        child1, child2 = crossover_function_name(
            population[parent_indices[0::2]],
            population[parent_indices[1::2]],
            **crossover_function_args_dict
        )

        # mutate the offspring
        child1 = mutation.mutation_bit_flip(
            child1, prob_mutation=prob_mutation
        )
        child2 = mutation.mutation_bit_flip(
            child2, prob_mutation=prob_mutation
        )

        # replace (some of the) old individuals with the new offspring
        if replace_worst:
            indices_to_replace = np.argsort(fitnesses)  # now the worst individuals are the first ones
            population[indices_to_replace[0:children_count], ] = \
                np.concatenate((child1, child2), axis=0)
        else:
            random.shuffle(all_indices)
            population[all_indices[0:children_count], ] = \
                np.concatenate((child1, child2), axis=0)

    if not suppress_output:
        print(f'The final generation was {generation_final}.')

        # save the best results
        save.save_to_csv(current_best_population,
                         current_best_fitnesses,
                         generation_final)

    return current_best_population, current_best_individuals, current_best_fitnesses,\
           generation_final, success, function_evals
