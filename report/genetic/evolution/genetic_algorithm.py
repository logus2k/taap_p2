import numpy as np
import random

from genetic.evolution.genome import create_genome
from genetic.evolution.parallel_evaluator import ParallelEvaluator
from genetic.evolution.utils import compute_genome_size


class GeneticAlgorithm:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size,
                 population_size, mutation_rate, render, max_workers=None,
                 generations=5000, mutation_rate_min=0.01):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.population_size = population_size
        self.mutation_rate_start = mutation_rate
        self.mutation_rate_min = mutation_rate_min
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.genome_length = compute_genome_size(input_size, hidden1_size, hidden2_size, output_size)
        self.population = self.initialize_population()
        self.render = render
        self.evaluator = ParallelEvaluator(max_workers)

    def initialize_population(self):
        return [create_genome(self.input_size, self.hidden1_size, self.hidden2_size, self.output_size)
                for _ in range(self.population_size)]

    def update_mutation_rate(self, generation):
        progress = generation / self.generations
        self.mutation_rate = self.mutation_rate_start + (self.mutation_rate_min - self.mutation_rate_start) * progress

    def mutate(self, genome):
        mutation = np.clip(np.random.randn(len(genome)) * self.mutation_rate, -0.1, 0.1)
        return genome + mutation

    def evaluate_population(self, generation):
        self.update_mutation_rate(generation)
        seeds = [random.randint(0, 1000) for _ in range(3)]
        return self.evaluator.evaluate(
            self.population, self.input_size, self.hidden1_size,
            self.hidden2_size, self.output_size, generation, seeds, self.render
        )

    def select_parents(self, sorted_population):
        num_parents = max(2, int(self.population_size * 0.2))
        parents = [genome for genome, _ in sorted_population[:num_parents]]
        return parents

    def crossover(self, parent1, parent2):
        child = np.array([np.random.choice([g1, g2])
                          for g1, g2 in zip(parent1, parent2)])
        return child

    def next_generation(self, sorted_population):
        parents = self.select_parents(sorted_population)
        new_population = []

        elites = [genome for genome, _ in sorted_population[:3]]
        new_population.extend(elites)

        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
