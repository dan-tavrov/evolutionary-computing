import numpy as np


# function that implements the bit-flip mutation for a simple GA
def mutation_bit_flip(population, prob_mutation, return_meta_data=False):
    output = population.copy()

    # create a matrix of random numbers
    randoms = np.random.rand(*population.shape)

    # flip the bits where the generated numbers are less than the probability
    indices = np.where(randoms <= prob_mutation)
    output[indices] = 1 - population[indices]

    if return_meta_data:
        return output, indices
    else:
        return output


# function that implements mutation for an ES with one strategic parameter
def mutation_es_one_sigma(population, nvar, sigma_lower_bound,
                          a=-np.inf, b=np.inf, tau=1):
    mu = population.shape[0]

    # mutate the sigmas first
    population[:, -1] = population[:, -1] * np.exp(np.random.normal(0, tau / np.sqrt(nvar), mu))

    # check if sigmas fall below the acceptable level
    population[population[:, -1] < sigma_lower_bound, -1] = sigma_lower_bound

    # mutate objective variables
    population[:, :nvar] = population[:, :nvar] + np.diag(population[:, -1]) @ np.random.randn(mu, nvar)

    # impose constraints
    population[:, :nvar][population[:, :nvar] < a] = a
    population[:, :nvar][population[:, :nvar] > b] = b

    return population


# function that implements mutation for an ES with many strategic parameters
def mutation_es_many_sigmas(population, nvar, sigma_lower_bound,
                            a=-np.inf, b=np.inf, tau=1, tau_prime=1):
    mu = population.shape[0]

    # mutate the sigmas first
    common_rands = np.random.randn(mu, 1) * tau_prime / np.sqrt(2*nvar)
    population[:, nvar:] = population[:, nvar:] * \
                        np.exp(common_rands + np.random.randn(mu, nvar) * tau / np.sqrt(2*np.sqrt(nvar)))

    # check if sigmas fall below the acceptable level
    population[:, nvar:][population[:, nvar:] < sigma_lower_bound] = sigma_lower_bound

    # mutate objective variables
    population[:, :nvar] = population[:, :nvar] + population[:, nvar:] * np.random.randn(mu, nvar)

    # impose constraints
    population[:, :nvar][population[:, :nvar] < a] = a
    population[:, :nvar][population[:, :nvar] > b] = b

    return population
