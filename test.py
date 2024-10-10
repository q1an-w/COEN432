import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from deap import tools, base, creator

# Constants
POPULATION_SIZE = 1000
GENERATIONS = 100
MAX_MUTATION_RATE = 0.95
MIN_MUTATION_RATE = 0.55
FITNESS_THRESHOLD = 100  # Fitness score at which to stop crossovers

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

# Rotate a tile by a given number of 90-degree rotations


def rotate_tile(tile, rotations):
    return np.roll(tile, -rotations)

# Initialize the population with unique arrangements of tiles


def initialize_population(tiles):
    population = []
    for _ in range(POPULATION_SIZE):
        arrangement = np.random.permutation(tiles)  # Random shuffle of tiles
        rotated_arrangement = np.array(
            [rotate_tile(tile, np.random.randint(0, 4)) for tile in arrangement])
        grid_arrangement = rotated_arrangement.reshape(8, 8, 4)
        population.append(creator.Individual(grid_arrangement))
    return population

# Fitness function based on edge mismatches


def fitness(individual):
    puzzle = np.array(individual).reshape(8, 8, 4)
    right_edges = puzzle[:, :-1, 1] != puzzle[:, 1:, 3]
    bottom_edges = puzzle[:-1, :, 2] != puzzle[1:, :, 0]
    total_mismatches = np.sum(right_edges) + np.sum(bottom_edges)
    fitness_score = 112 - total_mismatches
    return fitness_score,

# Select the best candidates using efficient numpy operations


def selection(population):
    fitness_scores = np.array([fitness(puzzle)[0] for puzzle in population])
    best_indices = np.argsort(
        fitness_scores)[-POPULATION_SIZE // 2:]  # Select top 50%
    return [population[i] for i in best_indices]

# Mutate a candidate solution by performing independent tile swaps and rotations


def mutate(puzzle, original_tiles, fitness_score):
    # Create a deep copy of the puzzle to maintain original tile integrity
    puzzle = np.copy(puzzle)

    mutation_rate = MAX_MUTATION_RATE * \
        (1 - (fitness_score / 112)) + MIN_MUTATION_RATE
    mutation_rate = min(mutation_rate, MAX_MUTATION_RATE)

    # Randomly determine the number of mutations to perform (between 2 and 5)
    num_mutations = np.random.randint(2, 6)  # Choose between 2 and 5 mutations

    for _ in range(num_mutations):
        if np.random.rand() < mutation_rate:
            # Random tile swap
            idx1 = np.random.randint(0, 8, size=2)
            idx2 = np.random.randint(0, 8, size=2)

            if not np.array_equal(idx1, idx2):
                # Perform swap within the puzzle (tiles are swapped directly)
                puzzle[idx1[0], idx1[1]], puzzle[idx2[0], idx2[1]
                                                 ] = puzzle[idx2[0], idx2[1]], puzzle[idx1[0], idx1[1]]

        if np.random.rand() < mutation_rate:
            # Random tile rotation within the individual puzzle
            idx1 = np.random.randint(0, 8, size=2)
            rotations1 = np.random.randint(1, 4)
            puzzle[idx1[0], idx1[1]] = rotate_tile(
                puzzle[idx1[0], idx1[1]], rotations1)

    return puzzle

# Run the genetic algorithm


def run_genetic_algorithm(tiles):
    population = initialize_population(tiles)
    best_solution = None
    best_fitness = -1

    start_time = time.time()  # Record the start time

    for generation in range(GENERATIONS):
        population = selection(population)

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = parent1.copy(), parent2.copy()

            # Mutate the children in place
            child1 = mutate(child1, tiles, fitness(child1)[0])
            child2 = mutate(child2, tiles, fitness(child2)[0])

            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        for puzzle in population:
            score = fitness(puzzle)[0]
            if score > best_fitness:
                best_fitness = score
                best_solution = puzzle

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Generation {generation + 1}/{GENERATIONS}: Best Fitness (score) = {best_fitness}")

    end_time = time.time()  # Record the end time
    total_run_time = end_time - start_time

    print(
        f"Total Run Time: {total_run_time:.2f} seconds | Mismatches: {112-best_fitness}")

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
    write_output(output_file, best_solution)

    print("Output written to", output_file)
