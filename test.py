import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from deap import tools, base, creator

# Constants
POPULATION_SIZE = 1000
GENERATIONS = 100
MAX_MUTATION_RATE = 0.9
MIN_MUTATION_RATE = 0.1
FITNESS_THRESHOLD = 60

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Optimized input file reader using pandas


def read_input(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None)
    tiles = [list(map(int, f"{tile:04d}")) for tile in df.values.flatten()]
    return np.array(tiles).reshape(64, 4)

# Precompute all rotations for each tile


def precompute_tile_rotations(tiles):
    tile_rotations = {}
    for tile in tiles:
        rotations = [tuple(np.roll(tile, -i)) for i in range(4)]
        tile_rotations[tuple(tile)] = rotations
    return tile_rotations

# Ensure that the individual consists of valid tiles from the input tiles


def create_valid_individual(individual, valid_tiles):
    # Set of all valid rotations for faster lookup
    valid_tiles_set = set(sum(valid_tiles.values(), []))

    for i in range(8):
        for j in range(8):
            tile = tuple(individual[i][j])
            if tile not in valid_tiles_set:
                # Find the closest valid tile
                individual[i][j] = min(valid_tiles_set, key=lambda t: np.sum(
                    np.abs(np.array(tile) - np.array(t))))
    return individual

# Initialize the population with unique arrangements of tiles


def initialize_population(tiles):
    population = []
    for _ in range(POPULATION_SIZE):
        arrangement = np.random.permutation(tiles).reshape(8, 8, 4)
        population.append(creator.Individual(arrangement))
    return population

# Fitness function based on edge mismatches


def fitness(individual):
    right_edges = individual[:, :-1, 1] != individual[:, 1:, 3]
    bottom_edges = individual[:-1, :, 2] != individual[1:, :, 0]
    total_mismatches = np.sum(right_edges) + np.sum(bottom_edges)
    return 112 - total_mismatches,

# Select the best candidates using numpy operations


def selection(population):
    return tools.selBest(population, k=POPULATION_SIZE // 2)

# Two-point crossover ensuring valid tiles


def two_point_crossover(parent1, parent2, valid_tiles):
    crossover_point1, crossover_point2 = sorted(random.sample(range(1, 7), 2))
    child1 = np.vstack(
        (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.vstack(
        (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
    return (creator.Individual(create_valid_individual(child1, valid_tiles)),
            creator.Individual(create_valid_individual(child2, valid_tiles)))

# Uniform crossover ensuring valid tiles


def uniform_crossover(parent1, parent2, valid_tiles):
    mask = np.random.rand(8, 8) > 0.5
    child1, child2 = np.where(mask[:, :, None], parent1, parent2), np.where(
        mask[:, :, None], parent2, parent1)
    return (creator.Individual(create_valid_individual(child1, valid_tiles)),
            creator.Individual(create_valid_individual(child2, valid_tiles)))

# Mutate a candidate solution ensuring valid tiles


def mutate(puzzle, tiles, fitness_score):
    mutation_rate = max(MAX_MUTATION_RATE *
                        (1 - fitness_score / 112), MIN_MUTATION_RATE)
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.random.randint(0, 8, size=(2, 2))
        valid_tile1, valid_tile2 = random.choice(tiles), random.choice(tiles)
        puzzle[idx1[0], idx1[1]], puzzle[idx2[0],
                                         idx2[1]] = valid_tile1, valid_tile2
    return puzzle

# Run the genetic algorithm


def run_genetic_algorithm(tiles, tile_rotations):
    population = initialize_population(tiles)
    best_solution, best_fitness = None, -1
    start_time = time.time()

    for generation in range(GENERATIONS):
        population = selection(population)
        new_population = []

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)
            crossover_func = (two_point_crossover if fitness(parent1)[
                              0] < FITNESS_THRESHOLD else uniform_crossover)
            child1, child2 = crossover_func(parent1, parent2, tile_rotations)
            child1, child2 = mutate(child1, tiles, fitness(child1)[0]), mutate(
                child2, tiles, fitness(child2)[0])
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        for puzzle in population:
            score = fitness(puzzle)[0]
            if score > best_fitness:
                best_fitness, best_solution = score, puzzle

        print(
            f"[{datetime.now()}] Generation {generation + 1}/{GENERATIONS}: Best Fitness = {best_fitness}")

    print(
        f"Total Run Time: {time.time() - start_time:.2f} seconds | Mismatches: {112 - best_fitness}")
    return best_solution


# Write the output to a file
team_info = "Qian Yi Wang (40211303) Philip Carlsson-Coulombe (40208572)"


def write_output(file_path, solution):
    with open(file_path, 'w') as file:
        file.write(f"{team_info}\n")
        for row in solution:
            file.write(' '.join([''.join(map(str, tile))
                       for tile in row]) + '\n')


# Main execution
if __name__ == "__main__":
    input_file, output_file = "Ass1Input.txt", "Ass1Output.txt"
    tiles = read_input(input_file)
    tile_rotations = precompute_tile_rotations(tiles)
    best_solution = run_genetic_algorithm(tiles, tile_rotations)
    write_output(output_file, best_solution)
    print("Output written to", output_file)
