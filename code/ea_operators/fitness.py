import numpy as np


# just a simple wrapper function that get the phenotypes for a given population
# and calculates fitness values using a given objective function
def compute_fitness(population, phenotypes_function, objective_function):
    individuals = phenotypes_function(population)

    # the function returns BOTH the individual phenotypes and the corresponding fitness values
    return individuals, objective_function(individuals)


# another simple wrapper function, but which also performs (linear) fitness scaling
def scale_fitness(fitnesses, c_m=2):
    fitness_max, fitness_min, fitness_avg = \
        np.max(fitnesses), np.min(fitnesses), np.mean(fitnesses)

    if np.allclose(fitness_max, fitness_avg):
        # a highly unlikely event but still
        return fitnesses

    if fitness_min < (c_m * fitness_avg - fitness_max) / (c_m - 1):
        # if the minimum point will be mapped onto a negative number,
        # calculate coefficients a and b so that the minimum point gets
        # mapped onto zero
        a = fitness_avg / (fitness_avg - fitness_min)
        b = -fitness_min * a
    else:
        # if all fitness values can be mapped onto positive numbers, proceed
        # with the usual formula

        # calculate coefficients a and b
        temp = fitness_avg / (fitness_max - fitness_avg)
        a = (c_m - 1) * temp
        b = (fitness_max - c_m * fitness_avg) * temp

    return a*fitnesses + b


# a function to derate fitnesses, as used in EAs that implement sharing
def derate_fitness(fitnesses, population, sigma_share):
    # get the population size
    mu = len(fitnesses)

    # get chromosome size
    l = population.shape[1]

    shares = np.zeros(mu)

    # calculate the sharing function for all individuals
    for i in range(mu):
        for j in range(mu):
            dist = l - np.sum(population[i, ] == population[j, ])
            share = 1 - dist/sigma_share

            if share >= 0:
                shares[i] += share

    # derate fitness
    return fitnesses / shares
