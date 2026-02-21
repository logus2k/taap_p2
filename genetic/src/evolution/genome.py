import numpy as np

from evolution.genetic_algorithm import compute_genome_size


def create_genome(input_size, hidden1_size, hidden2_size, output_size):
    genome_size = compute_genome_size(input_size, hidden1_size, hidden2_size, output_size)
    genome = np.random.uniform(-1, 1, genome_size)
    return genome
