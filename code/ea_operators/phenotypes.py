import numpy as np


def phenotype_simple_ga(population):
    # for a simple GA, each individual is assumed to be a binary representation of an integer

    # create a sequence of powers of two, in the correct order
    # (1 being the rightmost, and 2^{n - 1} being the leftmost)
    powers = 2**np.arange(population.shape[1])[::-1]

    # obtain the result as a (fast, hence @) dot product
    return population @ powers


def phenotype_real(population, a, b, nvar, bits_num):
    # for a general GA, each individual represents a real number on the interval [a, b] (up to a certain precision)
    # here bits_num is the number of bits used for coding one parameter

    # create a sequence of powers of two, in the correct order
    # (1 being the rightmost, and 2^{n - 1} being the leftmost)
    powers = 2 ** np.arange(bits_num)[::-1]

    # reshape the individuals matrix
    population_reshaped = population.reshape(-1, nvar, bits_num)

    # convert each binary number to a decimal
    return np.apply_along_axis(lambda x: (x @ powers) / (2**bits_num - 1) * (b - a) + a,
                               axis=2, arr=population_reshaped)
