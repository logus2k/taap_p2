# Genetic Algorithm Implementation

This document describes the complete implementation of a neuroevolution system that evolves neural networks to solve the LunarLander-v3 environment. The codebase is organized into four main modules plus an entry point:

```
src/
  evolution/
    utils.py                 - Shared utility (genome size calculation)
    genome.py                - Genome creation
    neural_network.py        - Neural network (phenotype)
    genetic_algorithm.py     - Core GA loop
    parallel_evaluator.py    - Parallel genome evaluation
  environment/
    lunarlander_runner.py    - Gymnasium environment interface
  main.py                   - Entry point
```

The flow is: **create genomes -> decode into neural networks -> evaluate in environment -> select, crossover, mutate -> repeat**.

## 1. The Genome (`genome.py`)

A genome is a flat NumPy array of floating-point numbers. Each number will become a weight or bias in the neural network. The genome encodes the entire network in a single vector.

```python
from evolution.utils import compute_genome_size

def create_genome(input_size, hidden1_size, hidden2_size, output_size):
    genome_size = compute_genome_size(input_size, hidden1_size, hidden2_size, output_size)
    genome = np.random.uniform(-1, 1, genome_size)
    return genome
```

**Key concepts:**

- `np.random.uniform(-1, 1, genome_size)` creates random values between -1 and 1. This is the initial "DNA" of each individual.
- The genome size depends on the network architecture. For our network (8 inputs, 10+10 hidden, 4 outputs), the genome has 284 values.

### Genome size calculation (`utils.py`)

```python
def compute_genome_size(input_size, hidden1_size, hidden2_size, output_size):
    return (input_size * hidden1_size + hidden1_size +
            hidden1_size * hidden2_size + hidden2_size +
            hidden2_size * output_size + output_size)
```

This counts every weight and bias in the network:

| Component | Count | For our architecture |
|-----------|-------|---------------------|
| Weights: input -> hidden1 | input_size * hidden1_size | 8 * 10 = 80 |
| Biases: hidden1 | hidden1_size | 10 |
| Weights: hidden1 -> hidden2 | hidden1_size * hidden2_size | 10 * 10 = 100 |
| Biases: hidden2 | hidden2_size | 10 |
| Weights: hidden2 -> output | hidden2_size * output_size | 10 * 4 = 40 |
| Biases: output | output_size | 4 |
| **Total** | | **244** |

Every one of these 244 numbers is a gene in the genome. Evolution operates on this flat array without knowing which genes are weights and which are biases.

## 2. The Neural Network (`neural_network.py`)

The neural network is the "phenotype" - the expressed form of the genome. It takes the flat genome array and reshapes it into weight matrices and bias vectors.

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, genome):
        idx = 0

        self.w1 = genome[idx:idx + input_size * hidden1_size].reshape((input_size, hidden1_size))
        idx += input_size * hidden1_size

        self.b1 = genome[idx:idx + hidden1_size]
        idx += hidden1_size

        self.w2 = genome[idx:idx + hidden1_size * hidden2_size].reshape((hidden1_size, hidden2_size))
        idx += hidden1_size * hidden2_size

        self.b2 = genome[idx:idx + hidden2_size]
        idx += hidden2_size

        self.w3 = genome[idx:idx + hidden2_size * output_size].reshape((hidden2_size, output_size))
        idx += hidden2_size * output_size

        self.b3 = genome[idx:idx + output_size]
```

**How the unpacking works:**

The genome is a flat array like `[0.3, -0.7, 0.1, 0.9, ...]`. The constructor walks through it sequentially with an index (`idx`), slicing out chunks and reshaping them:

1. First 80 values become the 8x10 weight matrix `w1`
2. Next 10 values become the bias vector `b1`
3. Next 100 values become the 10x10 weight matrix `w2`
4. Next 10 values become the bias vector `b2`
5. Next 40 values become the 10x4 weight matrix `w3`
6. Last 4 values become the bias vector `b3`

### Forward pass

```python
def forward(self, x):
    x = np.array(x)
    x = np.tanh(np.dot(x, self.w1) + self.b1)
    x = np.tanh(np.dot(x, self.w2) + self.b2)
    output = np.dot(x, self.w3) + self.b3
    return output
```

This is a standard feedforward neural network with:

- **Input**: 8 values from the environment (x, y, velocity_x, velocity_y, angle, angular_velocity, left_leg_contact, right_leg_contact)
- **Hidden layer 1**: `tanh(input * W1 + b1)` produces 10 values
- **Hidden layer 2**: `tanh(hidden1 * W2 + b2)` produces 10 values
- **Output**: `hidden2 * W3 + b3` produces 4 values (one per action)

The `tanh` activation squashes values to the range [-1, 1]. The output layer has no activation function (linear), so the raw scores determine the action. The action with the highest score is selected via `np.argmax(output)`.

**Important**: This is pure NumPy, no PyTorch, no autograd, no gradient computation. The network is just a mathematical function that maps 8 inputs to 4 outputs. The GA never needs to differentiate through it.

## 3. Environment Evaluation (`lunarlander_runner.py`)

This module runs a genome through the LunarLander environment and returns its fitness.

```python
def evaluate_genome(genome, input_size, hidden1_size, hidden2_size, output_size,
                    n_genome, generation, seeds, render=False):
    total_rewards = []

    env = gym.make("LunarLander-v3", render_mode="rgb_array" if (...) else None)
    nn = NeuralNetwork(input_size, hidden1_size, hidden2_size, output_size, genome)

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total_reward = 0
        done = False

        while not done:
            output = nn.forward(obs)
            action = np.argmax(output)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            if total_reward < -200:
                total_reward = -300
                break

        total_rewards.append(total_reward)

    env.close()
    return sum(total_rewards) / len(total_rewards)
```

**Step by step:**

1. Create the Gymnasium environment and decode the genome into a neural network
2. For each evaluation seed:
   - Reset the environment with that seed (different initial conditions)
   - Run a loop: observe state -> forward pass -> pick action -> step environment
   - Accumulate reward until the episode ends (landing, crash, or timeout)
   - If reward drops below -200, give a penalty of -300 and stop early (this genome is clearly bad, no point continuing)
3. Return the average reward across all seeds

**Multiple seeds**: Each genome is evaluated on 3 different random seeds per generation. This reduces noise - a genome that scores well on one lucky scenario but fails on others will get a mediocre average, preventing lucky genomes from dominating the population.

**Early termination penalty**: The `-300` penalty is harsher than just letting the episode continue. Without it, a bad genome that crashes early at -200 gets a better score than one that struggles longer and reaches -250. The penalty ensures consistently bad genomes are punished equally.

## 4. Parallel Evaluation (`parallel_evaluator.py`)

Since each genome's evaluation is independent, we can run them in parallel across CPU cores.

```python
def _evaluate_single(args):
    genome, input_size, hidden1_size, hidden2_size, output_size, index, generation, seeds, render = args
    fitness = evaluate_genome(genome, input_size, hidden1_size, hidden2_size,
                              output_size, index, generation, seeds, render)
    return genome, fitness


class ParallelEvaluator:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or os.cpu_count()

    def evaluate(self, genomes, input_size, hidden1_size, hidden2_size,
                 output_size, generation, seeds, render):
        tasks = [
            (genome, input_size, hidden1_size, hidden2_size, output_size,
             i, generation, seeds, render)
            for i, genome in enumerate(genomes)
        ]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_evaluate_single, tasks))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
```

**Key concepts:**

- `_evaluate_single` is a top-level function (not a method) because Python's `multiprocessing` on Windows uses `spawn`, which requires all worker functions to be picklable. Top-level functions satisfy this.
- `ProcessPoolExecutor` distributes the 50 genome evaluations across available CPU cores. With 20 workers and 50 genomes, each worker handles 2-3 evaluations.
- Results are sorted by fitness (highest first) before returning, so the caller always receives the population in rank order.
- The speedup is near-linear with core count since evaluations are completely independent (no shared state).

## 5. The Genetic Algorithm (`genetic_algorithm.py`)

This is the core module that orchestrates evolution. It manages the population and implements selection, crossover, and mutation.

### Initialization

```python
class GeneticAlgorithm:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size,
                 population_size, mutation_rate, render, max_workers=None,
                 generations=5000, mutation_rate_min=0.01):
        # ... store parameters ...
        self.mutation_rate_start = mutation_rate      # 0.05
        self.mutation_rate_min = mutation_rate_min    # 0.01
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = self.initialize_population()
        self.evaluator = ParallelEvaluator(max_workers)
```

The population starts as 50 random genomes. The mutation rate starts at 0.05 and will decay to 0.01 over the course of training.

### Adaptive Mutation Rate

```python
def update_mutation_rate(self, generation):
    progress = generation / self.generations
    self.mutation_rate = self.mutation_rate_start + \
        (self.mutation_rate_min - self.mutation_rate_start) * progress
```

This is a linear interpolation:
- At generation 0: `progress = 0`, so `mutation_rate = 0.05`
- At generation 2500: `progress = 0.5`, so `mutation_rate = 0.03`
- At generation 5000: `progress = 1.0`, so `mutation_rate = 0.01`

**Why decay?** Early in training, the population is random and needs large mutations to explore the search space. Later, the population has converged to good solutions and large mutations would knock them off course. Decaying the rate allows the algorithm to transition from exploration to refinement.

Without decay, the fixed mutation rate of 0.05 caused population collapses even at generation 4000+, where the average fitness would suddenly drop from 270 to 16 between generations.

### Evaluation

```python
def evaluate_population(self, generation):
    self.update_mutation_rate(generation)
    seeds = [random.randint(0, 1000) for _ in range(3)]
    return self.evaluator.evaluate(
        self.population, self.input_size, self.hidden1_size,
        self.hidden2_size, self.output_size, generation, seeds, self.render
    )
```

Each generation uses 3 fresh random seeds. This means the evaluation conditions change every generation - no genome can memorize a specific scenario. The seeds are shared across all 50 genomes in a generation, so their fitness scores are directly comparable.

### Selection

```python
def select_parents(self, sorted_population):
    num_parents = max(2, int(self.population_size * 0.2))
    parents = [genome for genome, _ in sorted_population[:num_parents]]
    return parents
```

**Truncation selection**: Only the top 20% (10 out of 50) survive to become parents. This creates strong selection pressure - the bottom 80% of genomes are discarded regardless of their absolute fitness.

The `max(2, ...)` guard ensures at least 2 parents exist, which is the minimum needed for crossover.

### Crossover

```python
def crossover(self, parent1, parent2):
    child = np.array([np.random.choice([g1, g2])
                      for g1, g2 in zip(parent1, parent2)])
    return child
```

**Uniform crossover**: For each gene (weight/bias) in the genome, the child randomly inherits from either parent with 50/50 probability. If parent1 has `[0.3, -0.7, 0.1]` and parent2 has `[0.8, 0.2, -0.5]`, the child might be `[0.3, 0.2, 0.1]` (took gene 1 from parent1, gene 2 from parent2, gene 3 from parent1).

This is the simplest form of crossover. It can disrupt correlated weight patterns (e.g., if a specific combination of weights in layer 1 works together), but in practice it performed well enough for this problem.

### Mutation

```python
def mutate(self, genome):
    mutation = np.clip(np.random.randn(len(genome)) * self.mutation_rate, -0.1, 0.1)
    return genome + mutation
```

**Gaussian mutation with clipping:**

1. `np.random.randn(len(genome))` generates one random value per gene from a standard normal distribution (mean=0, std=1)
2. Multiply by `self.mutation_rate` (0.05 early, 0.01 late) to scale the perturbation
3. `np.clip(..., -0.1, 0.1)` caps the maximum change to any single gene at +/-0.1
4. Add the mutation to the existing genome

This means every gene in every child is perturbed slightly. The clipping prevents any single gene from changing too drastically in one generation.

### Creating the Next Generation

```python
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
```

**Step by step:**

1. **Elitism**: The top 3 genomes are copied directly into the next generation without any modification. This guarantees that the best solutions are never lost. Without elitism, a lucky mutation could destroy the best genome and the population would regress.

2. **Fill remaining slots**: The remaining 47 slots are filled by:
   - Selecting 2 different parents (`random.sample` guarantees distinct parents, unlike `random.choices` which could pick the same parent twice)
   - Creating a child through crossover
   - Mutating the child
   - Adding it to the new population

3. **Replace**: The entire old population is discarded and replaced with the new one.

The balance is: 3 elites (6%) provide stability, 47 new children (94%) provide exploration. This ratio was tuned experimentally.

## 6. Entry Point (`main.py`)

The main script ties everything together:

```python
def main():
    parser = argparse.ArgumentParser(prog="Neuroevolution LunarLander-v3")
    parser.add_argument("-r", "--render", action="store_true")
    parser.add_argument("-w", "--workers", type=int, default=None)
    arg = parser.parse_args()

    # Network architecture
    input_size = 8       # LunarLander observation space
    hidden1_size = 10
    hidden2_size = 10
    output_size = 4      # LunarLander action space

    # GA hyperparameters
    population_size = 50
    mutation_rate = 0.05
    generations = 5000

    ga = GeneticAlgorithm(input_size, hidden1_size, hidden2_size, output_size,
                          population_size, mutation_rate, render, arg.workers,
                          generations=generations)

    for gen in range(generations):
        sorted_population = ga.evaluate_population(gen)
        # ... logging ...
        ga.next_generation(sorted_population)

    # ... save results and plot ...
```

The training loop is simple: evaluate, log, evolve, repeat. All complexity lives in the modules.

## 7. The Complete Training Pipeline

Here is how a single generation flows through the system:

```
Generation N
  |
  v
evaluate_population()
  |-- update_mutation_rate(N)           # Decay mutation based on progress
  |-- generate 3 random seeds           # New eval conditions each generation
  |-- ParallelEvaluator.evaluate()      # Distribute across CPU cores
  |     |
  |     |-- For each of 50 genomes (in parallel):
  |     |     |-- NeuralNetwork(genome)           # Decode genome
  |     |     |-- For each of 3 seeds:
  |     |     |     |-- env.reset(seed)           # New scenario
  |     |     |     |-- Loop: forward() -> argmax -> env.step()
  |     |     |     |-- Accumulate reward
  |     |     |-- Average reward across 3 seeds   # Fitness score
  |     |
  |     |-- Sort by fitness (descending)
  |
  v
next_generation(sorted_population)
  |-- select_parents()                  # Top 20% (10 genomes)
  |-- Copy top 3 as elites             # Preserved unchanged
  |-- For remaining 47 slots:
  |     |-- sample 2 distinct parents
  |     |-- crossover() -> child
  |     |-- mutate() -> mutated child
  |-- Replace population
  |
  v
Generation N+1
```

## 8. Hyperparameter Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Population size | 50 | Number of genomes per generation |
| Mutation rate (start) | 0.05 | Initial perturbation strength |
| Mutation rate (end) | 0.01 | Final perturbation strength |
| Mutation clip | +/- 0.1 | Max change per gene per generation |
| Elitism | Top 3 | Genomes preserved unchanged |
| Parent selection | Top 20% | Genomes eligible for reproduction |
| Crossover | Uniform (gene-wise) | How parents combine |
| Eval seeds per generation | 3 | Episodes per genome per generation |
| Generations | 5000 | Total evolutionary cycles |
| Network architecture | 8-10-10-4 | Input-hidden1-hidden2-output |
| Hidden activation | tanh | Squashes to [-1, 1] |
| Output activation | None (linear) | Raw scores for action selection |
| Genome size | 244 | Total weights + biases |
| Max workers | os.cpu_count() | Parallel evaluation processes |

## 9. Key Design Decisions and Their Effects

### Multi-seed evaluation (3 seeds per generation)
**Problem solved**: With 1 seed, a genome could score 300 by luck on an easy scenario, get selected as a parent, and pass on mediocre genes. The population would chase noise rather than genuine ability.
**Effect**: More robust selection. A genome must perform well across 3 different initial conditions to rank highly. This tripled evaluation time but dramatically improved convergence stability.

### Early termination penalty (-300)
**Problem solved**: Without it, a genome that crashes immediately at -200 gets a better fitness than one that tries hard but accumulates -250 over a longer episode. This rewarded giving up over trying.
**Effect**: All clearly failing genomes receive the same harsh penalty, creating a cleaner fitness landscape for selection.

### Elitism (top 3)
**Problem solved**: With only 1 elite (the original code), the second and third best genomes were discarded every generation. If the best genome happened to score poorly on the next generation's random seeds, the population could lose all its best solutions simultaneously.
**Effect**: More stable baseline. The population always retains its three best-known solutions, providing a safety net against regression.

### Adaptive mutation decay (0.05 to 0.01)
**Problem solved**: Fixed mutation at 0.05 caused population collapses even at generation 4000+. The average fitness would swing from 270 to 16 between consecutive generations because children were being perturbed too aggressively relative to their parents.
**Effect**: The most impactful single change. Eliminated late-generation collapses and improved final evaluation success rate from 81.7% to 100%.

### Distinct parent selection (random.sample vs random.choices)
**Problem solved**: `random.choices` could select the same genome as both parents. Crossover between identical parents produces an identical child, which then just gets mutated. This wastes a population slot on what is essentially a random walk from a single parent rather than a genuine recombination of two different solutions.
**Effect**: Every child is a true combination of two different parents, improving genetic diversity.

### Parallel evaluation (ProcessPoolExecutor)
**Problem solved**: Sequential evaluation of 50 genomes, each running 3 episodes, was slow. With 3 seeds, training time tripled compared to the original single-seed setup.
**Effect**: Near-linear speedup with core count. On a 24-core machine with 20 workers, a full 5000-generation run completes in approximately 17-30 minutes depending on the seed.
