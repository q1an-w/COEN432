import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from deap import tools, base, creator

# Constants
POPULATION_SIZE = 1000
GENERATIONS = 100
INITIAL_MUTATION_RATE = 0.8  # High mutation rate for diversity
FINAL_MUTATION_RATE = 0.1  # Low mutation rate for convergence
ELITISM_SIZE = 10  # Number of top individuals to carry over to the next generation

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


def initialize_population(tiles):
    population = []
    unique_set = set()  # Use a set to track unique configurations as tuples

    while len(population) < POPULATION_SIZE:
        arrangement = np.random.permutation(tiles)  # Faster random shuffle
        grid_arrangement = arrangement.reshape(8, 8, 4)

        # Convert grid arrangement to a tuple of tuples for uniqueness check
        individual_tuple = tuple(
            map(tuple, grid_arrangement.reshape(-1, 4).tolist()))

        # Ensure uniqueness before adding
        if individual_tuple not in unique_set:
            unique_set.add(individual_tuple)  # Add to set
            population.append(creator.Individual(
                grid_arrangement))  # Append the individual
    return population

# Ensure that a child maintains all unique tiles


def ensure_unique_tiles(grid_arrangement):
    flattened = grid_arrangement.reshape(-1, 4)
    unique_tiles = np.unique(flattened, axis=0)
    while unique_tiles.shape[0] < 64:
        unique_tiles = np.vstack((unique_tiles, random.choice(flattened)))
    return unique_tiles.reshape(8, 8, 4)

# Optimized fitness function using vectorized operations


# def fitness1(puzzle):
    score = 0
    edge_weight = 1.0  # Emphasizing edge matches
    mismatch_penalty = -0.1  # Moderate penalty for mismatches
    correct_position_bonus = 1.0  # Reward for correct position
    contiguous_bonus = 1.5  # Reward for contiguous matches

    # Evaluate right edge matches
    right_edges = puzzle[:, :-1, 1] == puzzle[:, 1:, 3]
    right_matches = np.sum(right_edges) * edge_weight
    right_mismatches = (
        right_edges.shape[0] - np.sum(right_edges)) * mismatch_penalty

    # Evaluate bottom edge matches
    bottom_edges = puzzle[:-1, :, 2] == puzzle[1:, :, 0]
    bottom_matches = np.sum(bottom_edges) * edge_weight
    bottom_mismatches = (
        bottom_edges.shape[0] - np.sum(bottom_edges)) * mismatch_penalty

    # Evaluate contiguous matches
    contiguous_matches = np.sum(
        right_edges[:-1, :] & bottom_edges[:, :-1]) * contiguous_bonus

    # Count correct position but wrong orientation
    correct_orientation = (puzzle[:, :, 1] == puzzle[:, :, 3]) | (
        puzzle[:, :, 2] == puzzle[:, :, 0])
    correct_position_count = np.sum(correct_orientation)

    # Calculate total score
    score += (right_matches + bottom_matches + right_mismatches + bottom_mismatches +
              contiguous_matches + correct_position_count * correct_position_bonus)

    return score


def fitness(puzzle):
    total_mismatches = 0

    # Evaluate right edge matches
    # True if there is a mismatch
    right_edges = puzzle[:, :-1, 1] != puzzle[:, 1:, 3]
    total_mismatches += np.sum(right_edges)

    # Evaluate bottom edge matches
    # True if there is a mismatch
    bottom_edges = puzzle[:-1, :, 2] != puzzle[1:, :, 0]
    total_mismatches += np.sum(bottom_edges)

    # Calculate fitness score as 112 - total_mismatches
    fitness_score = 112 - total_mismatches

    return fitness_score

# Stochastic Universal Sampling for selection


def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selected = []

    for _ in range(POPULATION_SIZE // 2):
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, individual in enumerate(population):
            current += fitness_values[i]
            if current >= pick:
                selected.append(individual)
                break

    return selected

# Crossover while ensuring tile uniqueness


def two_point_crossover(parent1, parent2):
    child1_tiles = np.empty_like(parent1)
    child2_tiles = np.empty_like(parent2)

    crossover_point1 = random.randint(1, 6)
    crossover_point2 = random.randint(crossover_point1 + 1, 7)

    for i in range(8):
        for j in range(8):
            if (i < crossover_point1 or i >= crossover_point2):
                child1_tiles[i, j] = parent1[i, j]
            else:
                child1_tiles[i, j] = parent2[i, j]

    for i in range(8):
        for j in range(8):
            if (i < crossover_point1 or i >= crossover_point2):
                child2_tiles[i, j] = parent2[i, j]
            else:
                child2_tiles[i, j] = parent1[i, j]

    child1_tiles = ensure_unique_tiles(child1_tiles)
    child2_tiles = ensure_unique_tiles(child2_tiles)

    return creator.Individual(child1_tiles), creator.Individual(child2_tiles)

# Rotate tile to simulate orientation changes


def rotate_tile(tile):
    # Rotate right, can be adjusted for different orientations
    return np.roll(tile, shift=1)

# Mutate a candidate solution while ensuring uniqueness


def mutate(puzzle, generation):
    if generation < 0.75 * GENERATIONS:  # Early generations focus on diversity
        mutation_rate = INITIAL_MUTATION_RATE
        if np.random.rand() < mutation_rate:
            # Randomly choose two positions
            idx = np.random.randint(0, 8, size=(2, 2))
            puzzle[idx[0][0], idx[0][1]], puzzle[idx[1][0], idx[1][1]] = puzzle[idx[1]
                                                                                [0], idx[1][1]].copy(), puzzle[idx[0][0], idx[0][1]].copy()
            puzzle = ensure_unique_tiles(puzzle)
    else:  # Later generations focus on orientation adjustment
        mutation_rate = FINAL_MUTATION_RATE
        if np.random.rand() < mutation_rate:
            for i in range(8):
                for j in range(8):
                    if np.random.rand() < mutation_rate:  # Randomly rotate some tiles
                        puzzle[i, j] = rotate_tile(puzzle[i, j])

    return puzzle

# Run the genetic algorithm


def run_genetic_algorithm(tiles):
    population = initialize_population(tiles)
    best_solution = None
    best_score = -1

    start_time = time.time()

    for generation in range(GENERATIONS):
        # Calculate fitness for the entire population once
        fitness_values = [fitness(ind) for ind in population]

        # Selection
        population = selection(population, fitness_values)

        # Carry over the best individuals
        elite_individuals = sorted(population, key=fitness, reverse=True)[
            :ELITISM_SIZE]
        new_population = elite_individuals[:]

        # Crossover phase
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = two_point_crossover(parent1, parent2)

            child1 = mutate(child1, generation)
            child2 = mutate(child2, generation)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        # Track best solution
        for puzzle in population:
            score = fitness(puzzle)
            if score > best_score:
                best_score = score
                best_solution = puzzle

        # Log the best fitness score for the current generation with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"[{timestamp}] Generation {generation + 1}/{GENERATIONS}: Best Fitness (score) = {best_score}")

    end_time = time.time()
    total_run_time = end_time - start_time
    print(f"Total Run Time: {total_run_time:.2f} seconds")

    return best_solution


# Write the output to a file
team_info = "TeamName TeamID1 TeamID2"


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
