import numpy as np


# function that implements the one-point crossover for a simple GA
def crossover_one_point(parents1, parents2, prob_crossover,
                        return_meta_data=False):
    if len(parents1.shape) == 1:
        parents1 = np.asmatrix(parents1)
        parents2 = np.asmatrix(parents2)

    # initialize the results
    children1 = parents1.copy()
    children2 = parents2.copy()

    (mu, chromosome_length) = parents1.shape

    # we "flip a coin" to see if we need to perform crossover
    # for each parent pair we generate a random number and see if it exceeds probability
    rands = np.random.rand(mu)
    indices = np.where(rands <= prob_crossover)[0]

    # generate crossover points
    crossover_points = np.random.randint(chromosome_length - 1, size=len(indices))

    # exchange tails of the chromosomes
    for i, index in enumerate(indices):
        tail = np.arange(crossover_points[i] + 1, chromosome_length)
        children1[index, tail] = parents2[index, tail].copy()
        children2[index, tail] = parents1[index, tail].copy()

    if return_meta_data:
        return children1, children2, crossover_points, indices
    else:
        return children1, children2


# function that implements the n-point crossover for a GA
def crossover_n_point(parents1, parents2, prob_crossover,
                      return_meta_data=False, n=1):
    if len(parents1.shape) == 1:
        parents1 = np.asmatrix(parents1)
        parents2 = np.asmatrix(parents2)

    # initialize the results
    children1 = parents1.copy()
    children2 = parents2.copy()

    (mu, chromosome_length) = parents1.shape

    # we "flip a coin" to see if we need to perform crossover
    # for each parent pair we generate a random number and see if it exceeds probability
    rands = np.random.rand(mu)
    indices = np.where(rands <= prob_crossover)[0]

    # generate crossover points
    crossover_points = np.sort(
        np.random.randint(chromosome_length - 1, size=(len(indices), n)),
        axis=1
    )

    # append last index to do everything in one loop
    crossover_points_full = np.c_[crossover_points, chromosome_length*np.ones((len(indices), 1))].astype(int)

    for i in range(len(indices)):
        index = indices[i]
        start_idx = 0

        for j, point in enumerate(crossover_points_full[i, ]):
            end_idx = point + 1

            if j % 2 == 1:
                children1[index, start_idx:end_idx] = parents1[index, start_idx:end_idx].copy()
                children2[index, start_idx:end_idx] = parents2[index, start_idx:end_idx].copy()
            else:
                children1[index, start_idx:end_idx] = parents2[index, start_idx:end_idx].copy()
                children2[index, start_idx:end_idx] = parents1[index, start_idx:end_idx].copy()

            start_idx = end_idx

    if return_meta_data:
        return children1, children2, crossover_points, indices
    else:
        return children1, children2


# function that implements the uniform crossover for a GA
def crossover_uniform(parents1, parents2, prob_crossover,
                      return_meta_data=False):
    if len(parents1.shape) == 1:
        parents1 = np.asmatrix(parents1)
        parents2 = np.asmatrix(parents2)

    # initialize the results
    children1 = parents1.copy()
    children2 = parents2.copy()

    (mu, chromosome_length) = parents1.shape

    # we "flip a coin" to see if we need to perform crossover
    # for each parent pair we generate a random number and see if it exceeds probability
    rands = np.random.rand(mu)
    indices = np.where(rands <= prob_crossover)[0]

    # create a matrix of random numbers
    randoms = np.zeros_like(parents1)
    randoms[indices, :] = np.random.rand(len(indices), chromosome_length)

    children1[indices, :] = np.where(randoms[indices, :] > 0.5, parents2[indices, :], parents1[indices, :])
    children2[indices, :] = np.where(randoms[indices, :] > 0.5, parents1[indices, :], parents2[indices, :])

    if return_meta_data:
        return children1, children2, randoms, indices
    else:
        return children1, children2
