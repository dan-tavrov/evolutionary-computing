import numpy as np

# this function takes fitness values as input and returns INDICES of selected individuals
def selection_roulette_wheel(fitnesses, children_count=0):
    # lambda stands for the number of children to create

    # get the population size
    mu = len(fitnesses)

    if children_count == 0:
        # basically if we omitted this argument, we need to generate mu offspring
        children_count = mu

    if np.sum(fitnesses) == 0:
        # obviously, in this case we need uniform distribution
        probabilities = np.ones(mu) / mu
    else:
        # obtain selection probabilities
        probabilities = fitnesses / np.sum(fitnesses)

    # create the roulette
    probabilities_cum = np.cumsum(probabilities)

    # generate a random number
    r = np.random.rand(1) / children_count

    # find the index on the roulette where the next "arrow" is located
    parent_indices = np.zeros(children_count, dtype='int64')
    for i in range(children_count):
        # in a loop, until all individuals are selected

        b = np.where(probabilities_cum >= r)[0]
        if len(b) != 0:
            parent_indices[i] = b[0]  # retrieve the number itself, not the array
        else:
            parent_indices[i] = mu

        # proceed to the next "arrow"
        r = r + 1 / children_count

    return np.random.permutation(parent_indices)
