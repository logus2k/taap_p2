import os
from concurrent.futures import ProcessPoolExecutor
from environment.lunarlander_runner import evaluate_genome


def _evaluate_single(args):
    genome, input_size, hidden1_size, hidden2_size, output_size, index, generation, seeds, render = args
    fitness = evaluate_genome(genome, input_size, hidden1_size, hidden2_size, output_size, index, generation, seeds, render)
    return genome, fitness


class ParallelEvaluator:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or os.cpu_count()

    def evaluate(self, genomes, input_size, hidden1_size, hidden2_size, output_size, generation, seeds, render):
        tasks = [
            (genome, input_size, hidden1_size, hidden2_size, output_size, i, generation, seeds, render)
            for i, genome in enumerate(genomes)
        ]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_evaluate_single, tasks))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
