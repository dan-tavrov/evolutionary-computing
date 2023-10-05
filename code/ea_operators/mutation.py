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
