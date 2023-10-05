import numpy as np


def save_to_csv(population, individuals, fitnesses, generation):
    chromosome_length = population.shape[1]
    data_to_save = np.column_stack((population, individuals, fitnesses))
    np.savetxt('population_{}.csv'.format(generation),
               data_to_save,
               fmt='%f',
               delimiter=',',
               header=','*chromosome_length + 'Phenotype,Fitness',
               comments='')
