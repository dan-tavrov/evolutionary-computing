import numpy as np
from ea_operators import mutation


def initialize_simple_ga(mu, chromosome_length):
    return np.round(np.random.rand(mu, chromosome_length))


def initialize_real(mu, nvar, bits_num):
    # here nvar is the number of parameters to be coded
    # bits_num is the number of bits per one parameter
    return initialize_simple_ga(mu, nvar*bits_num)


def initialize_es_one_sigma(mu, nvar, starting_point, sigma_lower_bound,
                            a=-np.inf, b=np.inf, sigma_initial=3):
    # we initialize a population for the ES with 1 sigma parameter
    # we mutate a starting point mu times
    population = np.zeros((mu, nvar + 1))

    # set all object variables equal to the starting point
    population[:, :nvar] = starting_point

    # set all initial sigmas
    population[:, -1] = sigma_initial

    # mutate each individual
    return mutation.mutation_es_one_sigma(population, nvar, sigma_lower_bound, a=a, b=b)


def initialize_es_many_sigmas(mu, nvar, starting_point, sigma_lower_bound,
                              a=-np.inf, b=np.inf, sigma_initial=1):
    # we initialize a population for the ES with nvar sigma parameters, one for each dimension
    # we mutate a starting point mu times
    population = np.zeros((mu, 2*nvar))

    # set all object variables equal to the starting point
    population[:, :nvar] = starting_point

    # set all initial sigmas
    population[:, nvar:] = sigma_initial

    # mutate each individual
    return mutation.mutation_es_many_sigmas(population, nvar, sigma_lower_bound, a=a, b=b)
