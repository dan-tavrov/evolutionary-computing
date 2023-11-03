import time

import numpy as np
from IPython.core.display import clear_output

from ea_operators import initialize, crossover, mutation
from ea_utilities import visualize, save


# function that implements an ES for parameter optimization
# note that we only consider minimization problems -- for simplicity
def es(objective_function,
       mu=10,  # population size
       children_count=100,  # how many individuals to replace in each generation
       nvar=1,  # number of parameters of the objective function
       recombination_function=crossover.recombination_intermediate,  # recombination function
       selection_strategy=0,  # 0 if "comma", 1 if "plus"
       many_sigmas=True,  # whether we use separate sigmas for each object variable
       generations_count=1000,  # maximum number of iterations
       starting_point=0,  # for initialization
       sigma_initial=1,  # initial sigma for mutation
       optimum_value=0,  # optimum value of a function, if known in advance
       precision=1e-8,  # precision for the final solution
       sigma_lower_bound=1e-9,  # lower bound for sigmas
       a=-np.inf, b=np.inf,  # constraints
       tau=1, tau_prime=1,  # constants of proportionality for mutating sigmas
       do_draw_stats=False,  # whether to draw best and average fitnesses
       do_print=True,  # whether to print results of each generation
       do_save=False,  # whether to save each generation in a separate file
       suppress_output=False):  # whether to report the final information
    if mu % 2 != 0:
        # make sure our population is even-numbered
        mu -= 1
        if mu <= 0:
            return

    if many_sigmas:
        population = initialize.initialize_es_many_sigmas(mu, nvar, starting_point,
                                                          sigma_lower_bound, sigma_initial=sigma_initial,
                                                          a=a, b=b)
    else:
        population = initialize.initialize_es_one_sigma(mu, nvar, starting_point,
                                                        sigma_lower_bound, sigma_initial=sigma_initial,
                                                        a=a, b=b)

    stat_fig = None

    fitness_bests = np.zeros(generations_count)
    fitness_worsts = np.zeros(generations_count)
    fitness_avgs = np.zeros(generations_count)

    current_best_fitness = np.inf
    current_best_population = None
    current_best_fitnesses = None
    generation_final = generations_count

    # start recording the time
    start_time = time.time()

    # calculate fitness values for the first population
    fitnesses = objective_function(population[:, :nvar])
    function_evals = mu

    success = False
    for i in range(generations_count):
        # compute statistics
        fitness_bests[i], fitness_worsts[i], fitness_avgs[i], fitness_best_index = \
            np.min(fitnesses), np.max(fitnesses), np.mean(fitnesses), np.argmin(fitnesses)

        norm_diff_to_optimum = np.linalg.norm(fitness_bests[i] - optimum_value)

        if fitness_bests[i] < current_best_fitness:
            current_best_fitness = fitness_bests[i]
            current_best_population = population
            current_best_fitnesses = fitnesses

        # draw fitness stats only
        if do_draw_stats:
            stat_fig = visualize.draw_fitness_stats(
                fitness_bests, fitness_avgs, i + 1, current_best_fitness,
                stat_fig
            )

            # clear current figures
            clear_output(wait=True)

        # print info as needed
        if do_print:
            visualize.report_ea_progress(
                fitness_bests[i], fitness_worsts[i], fitness_avgs[i],
                population[fitness_best_index, :nvar],
                i + 1,
                time.time() - start_time,
                norm_diff_to_optimum
            )

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

        # create children
        children = np.tile(recombination_function(population, nvar), (children_count, 1))

        # mutation
        if many_sigmas:
            children = mutation.mutation_es_many_sigmas(children, nvar, sigma_lower_bound,
                                                        a, b, tau, tau_prime)
        else:
            children = mutation.mutation_es_one_sigma(children, nvar, sigma_lower_bound,
                                                      a, b, tau)

        # compute fitnesses of these children
        fitnesses_children = objective_function(children[:, :nvar])
        function_evals += children_count

        if selection_strategy == 0:
            # if we have "comma" strategy, we select only among new children
            indices = np.argsort(fitnesses_children)
            population = children[indices[:mu], :].copy()
            fitnesses = fitnesses_children[indices[:mu]]
        else:
            # if we have "plus" strategy, we select among children and old parents
            fitnesses_total = np.concatenate((fitnesses, fitnesses_children))
            population_total = np.vstack((population, children))

            indices = np.argsort(fitnesses_total)
            population = population_total[indices[:mu], :].copy()
            fitnesses = fitnesses_total[indices[:mu]]

    if not suppress_output:
        print(f'The final generation was {generation_final}.')

        # save the best results
        save.save_to_csv(current_best_population,
                         current_best_fitnesses,
                         generation_final)

    return current_best_population, current_best_population[:, :nvar], current_best_fitnesses,\
           generation_final, success, function_evals
