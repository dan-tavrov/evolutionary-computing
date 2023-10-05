import numpy as np


def initialize_simple_ga(mu, chromosome_length):
    return np.round(np.random.rand(mu, chromosome_length))


def initialize_real(mu, nvar, bits_num):
    # here nvar is the number of parameters to be coded
    # bits_num is the number of bits per one parameter
    return initialize_simple_ga(mu, nvar*bits_num)
