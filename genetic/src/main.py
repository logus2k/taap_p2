import matplotlib.pyplot as plt
import argparse
import json
import os

from evolution.genetic_algorithm import GeneticAlgorithm

def main():
    
    parser = argparse.ArgumentParser(prog="Neuroevolution LunarLander-v3")
    
    parser.add_argument("-r", "--render", action="store_true", help="Renders all genomes gifs and stores them in /assets/gifs/.")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Max parallel workers (default: CPU count)")
    arg = parser.parse_args()

    input_size = 8
    hidden1_size = 10
    hidden2_size = 10
    output_size = 4

    population_size = 50
    mutation_rate = 0.05
    generations = 5000

    render = True if arg.render else False

    os.makedirs("assets/data", exist_ok=True)
    os.makedirs("assets/figures", exist_ok=True)
    os.makedirs("assets/gifs", exist_ok=True)

    ga = GeneticAlgorithm(
        input_size, 
        hidden1_size, 
        hidden2_size, 
        output_size,
        population_size, 
        mutation_rate, 
        render, 
        arg.workers,
        generations=generations
    )

    best_fitness_history = []
    avg_fitness_history = []
    worst_fitness_history = []

    for gen in range(generations):
        sorted_population = ga.evaluate_population(gen)
        fitness_values = [fitness for _, fitness in sorted_population]
        best_fitness = fitness_values[0]
        worst_fitness = fitness_values[-1]
        avg_fitness = sum(fitness_values) / len(fitness_values)

        print(f"Generation {gen + 1} | Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f} | Worst: {worst_fitness:.2f}")

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        worst_fitness_history.append(worst_fitness)

        ga.next_generation(sorted_population)

    with open("assets/data/fitness_stats.json", "w") as f:
        json.dump({
            "best": best_fitness_history,
            "avg": avg_fitness_history,
            "worst": worst_fitness_history
        }, f)

    plt.plot(best_fitness_history, label="Best Fitness")
    plt.plot(avg_fitness_history, label="Average Fitness")
    plt.plot(worst_fitness_history, label="Worst Fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(bbox_to_anchor=(0.99, 1.12), ncol=3)

    plt.savefig("assets/figures/fitness_plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
