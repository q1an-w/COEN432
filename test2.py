import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from deap import tools, base, creator

# Constants
POPULATION_SIZE = 1000
GENERATIONS = 100
MAX_MUTATION_RATE = 0.8
MIN_MUTATION_RATE = 0.2
FITNESS_THRESHOLD = 70  # The fitness score at which to switch crossover strategies
ELITE_PERCENTAGE = 0.1  # Percentage of elite individuals to carry over
RANDOM_PERCENTAGE = 0.15  # Percentage of random individuals to introduce

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Optimized input file reader using pandas


def read_input(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None)
    tiles = []
    for row in df.values:
        for tile in row:
            tile_digits = [int(digit) for digit in f"{tile:04d}"]
            tiles.append(tile_digits)
    return np.array(tiles).reshape(64, 4)

# Initialize the population with unique arrangements of tiles


def initialize_population(tiles):
    population = []
    for _ in range(POPULATION_SIZE):
        arrangement = np.random.permutation(tiles)  # Faster random shuffle
        grid_arrangement = arrangement.reshape(8, 8, 4)
        population.append(creator.Individual(grid_arrangement))
    return population

# Counts correct edges since max # good edges is 112


def fitness(individual):
    puzzle = np.array(individual).reshape(8, 8, 4)
    right_edges = puzzle[:, :-1, 1] != puzzle[:, 1:, 3]
    bottom_edges = puzzle[:-1, :, 2] != puzzle[1:, :, 0]
    total_mismatches = np.sum(right_edges) + np.sum(bottom_edges)
    fitness_score = 112 - total_mismatches
    return fitness_score,

# Select the best candidates using numpy for faster


def selection(population):
    fitness_scores = np.array([fitness(puzzle)[0] for puzzle in population])
    best_indices = np.argsort(
        fitness_scores)[-POPULATION_SIZE // 2:]  # Select top 50%
    return [population[i] for i in best_indices]

# Generate a random individual


def generate_random_individual(tiles):
    arrangement = np.random.permutation(tiles)
    grid_arrangement = arrangement.reshape(8, 8, 4)
    return creator.Individual(grid_arrangement)

# Two-point crossover ensuring valid tiles


def two_point_crossover(parent1, parent2):
    crossover_point1 = random.randint(1, 6)
    crossover_point2 = random.randint(crossover_point1 + 1, 7)

    # Create child arrays with swapped sections
    child1 = np.vstack(
        (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.vstack(
        (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))

    # Ensure children contain only valid tiles from the input
    return creator.Individual(child1), creator.Individual(child2)

# Uniform crossover ensuring valid tiles


def uniform_crossover(parent1, parent2):
    mask = np.random.rand(8, 8) > 0.5
    child1 = np.where(mask[:, :, None], parent1, parent2)
    child2 = np.where(mask[:, :, None], parent2, parent1)

    return creator.Individual(child1), creator.Individual(child2)

# Rotate a tile


def rotate_tile(tile, rotations):
    rotation_mapping = {
        0: tile,                          # No rotation
        1: [tile[3], tile[0], tile[1], tile[2]],  # 1 clockwise rotation
        2: [tile[2], tile[3], tile[0], tile[1]],  # 2 clockwise rotations
        3: [tile[1], tile[2], tile[3], tile[0]],  # 3 clockwise rotations
    }
    return rotation_mapping[rotations]


def swap_tiles(puzzle, idx1, idx2):
    # Create a copy of the puzzle to avoid modifying the original
    new_puzzle = puzzle.copy()

    # Swap tiles using NumPy's advanced indexing
    new_puzzle[idx1], new_puzzle[idx2] = new_puzzle[idx2].copy(
    ), new_puzzle[idx1].copy()

    return new_puzzle

# Mutate a candidate solution ensuring valid tiles


def mutate(puzzle, tiles, fitness_score):
    if fitness_score < 112:  # Only scale if fitness is less than the max
        mutation_rate = MAX_MUTATION_RATE * (1 - (fitness_score / 112))
        mutation_rate = max(mutation_rate, MIN_MUTATION_RATE)
    else:
        mutation_rate = MIN_MUTATION_RATE  # Apply minimum rate if fitness is maximized

    if np.random.rand() < mutation_rate:
        num_mutations = 3 if fitness_score < FITNESS_THRESHOLD else 1
        for _ in range(num_mutations):
            action = random.choice(['swap', 'rotate'])
            idx1 = np.random.randint(0, 8, size=2)

            if action == 'swap':
                idx2 = np.random.randint(0, 8, size=2)
                # Swap tiles
                puzzle = swap_tiles(puzzle, tuple(idx1), tuple(idx2))
            elif action == 'rotate':
                # Rotate a tile at idx1
                rotations = np.random.randint(1, 4)  # Rotate 1 to 3 times
                tile_to_rotate = puzzle[tuple(idx1)].copy()
                rotated_tile = rotate_tile(tile_to_rotate, rotations)
                puzzle[tuple(idx1)] = rotated_tile

    return puzzle

# Run the genetic algorithm


def run_genetic_algorithm(tiles):
    population = initialize_population(tiles)
    best_solution = None
    best_fitness = -1

    start_time = time.time()  # Record the start time

    for generation in range(GENERATIONS):
        # Selection
        population = selection(population)

        # Preserve elite individuals
        elite_count = int(ELITE_PERCENTAGE * POPULATION_SIZE)
        elites = population[-elite_count:]  # Keep top elite_count individuals

        # Generate random individuals
        random_count = int(RANDOM_PERCENTAGE * POPULATION_SIZE)
        random_individuals = [generate_random_individual(
            tiles) for _ in range(random_count)]

        # Crossover phase
        # Start new population with elites and randoms
        new_population = list(elites) + random_individuals
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)

            # Switch to uniform crossover based on fitness score
            if fitness(parent1)[0] < FITNESS_THRESHOLD and fitness(parent2)[0] < FITNESS_THRESHOLD:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1, child2 = uniform_crossover(parent1, parent2)

            # Mutate and append to new population
            child1 = mutate(child1, tiles, fitness(child1)[0])
            child2 = mutate(child2, tiles, fitness(child2)[0])
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        # Track best solution
        for puzzle in population:
            score = fitness(puzzle)[0]
            if score > best_fitness:
                best_fitness = score
                best_solution = puzzle

        # Log the current best fitness score
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Generation {generation + 1}/{GENERATIONS}: Best Fitness (score) = {best_fitness}")

    end_time = time.time()  # Record the end time
    total_run_time = end_time - start_time

    # Log total run time
    print(
        f"Total Run Time: {total_run_time:.2f} seconds | Mismatches: {112 - best_fitness}")

    return best_solution


# Write the output to a file
team_info = "Qian Yi Wang (40211303) Philip Carlsson-Coulombe (40208572)"


def write_output(file_path, solution):
    with open(file_path, 'w') as file:
        file.write(f"{team_info}\n")
        for row in solution:
            line = ' '.join([''.join(map(str, tile)) for tile in row])
            file.write(line + '\n')


# Main execution
if __name__ == "__main__":
    input_file = "Ass1Input.txt"
    output_file = "Ass1Output.txt"

    tiles = read_input(input_file)
    best_solution = run_genetic_algorithm(tiles)
