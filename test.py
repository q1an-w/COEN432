import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from deap import tools, base, creator

# Constants
POPULATION_SIZE = 1000
GENERATIONS = 100
INITIAL_MUTATION_RATE = 0.8
FINAL_MUTATION_RATE = 0.1
ELITISM_SIZE = 10
TOURNAMENT_SIZE = 5
CONSERVATIVE_FITNESS_THRESHOLD = 0.90

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

fitness_cache = {}


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
    unique_set = set()
    while len(population) < POPULATION_SIZE:
        arrangement = np.random.permutation(tiles)
        grid_arrangement = arrangement.reshape(8, 8, 4)
        individual_tuple = tuple(
            map(tuple, grid_arrangement.reshape(-1, 4).tolist()))
        if individual_tuple not in unique_set:
            unique_set.add(individual_tuple)
            population.append(creator.Individual(grid_arrangement.flatten()))
    return population


def all_rotations(tile):
    return [np.roll(tile, shift) for shift in range(4)]


def ensure_unique_tiles(grid_arrangement, original_tiles):
    unique_tiles_set = set()
    flattened = grid_arrangement.reshape(-1, 4)

    original_rotations_set = {
        tuple(rot) for tile in original_tiles for rot in all_rotations(tile)}

    for tile in flattened:
        for rotation in all_rotations(tile):
            if tuple(rotation) in original_rotations_set:
                unique_tiles_set.add(tuple(rotation))
                break

    while len(unique_tiles_set) < 64:
        new_tile = random.choice(original_tiles)
        for rotation in all_rotations(new_tile):
            unique_tiles_set.add(tuple(rotation))
            if len(unique_tiles_set) >= 64:
                break

    unique_tiles = np.array(list(unique_tiles_set))

    if unique_tiles.shape[0] < 64:
        raise ValueError("Insufficient unique tiles.")

    return unique_tiles.reshape(8, 8, 4)


def fitness(individual):
    ind_tuple = tuple(individual)
    if ind_tuple in fitness_cache:
        return fitness_cache[ind_tuple]

    puzzle = np.array(individual).reshape(8, 8, 4)
    right_edges = puzzle[:, :-1, 1] != puzzle[:, 1:, 3]
    bottom_edges = puzzle[:-1, :, 2] != puzzle[1:, :, 0]

    total_mismatches = np.sum(right_edges) + np.sum(bottom_edges)
    fitness_score = 112 - total_mismatches

    fitness_cache[ind_tuple] = fitness_score,
    return fitness_score,


def rank_selection(population, fitness_values):
    sorted_indices = np.argsort(fitness_values)[
        ::-1]  # Sort in descending order
    rank_probs = np.arange(len(population)) / \
        len(population)  # Rank-based probabilities
    selected_indices = np.random.choice(sorted_indices, size=len(
        population) // 2, p=rank_probs / np.sum(rank_probs))
    return [population[i] for i in selected_indices]


def uniform_crossover(parent1, parent2, original_tiles):
    parent1_tiles = np.array(parent1).reshape(8, 8, 4)
    parent2_tiles = np.array(parent2).reshape(8, 8, 4)

    mask = np.random.rand(8, 8) < 0.5  # Create a random mask for crossover
    child1_tiles = np.where(mask[..., None], parent1_tiles, parent2_tiles)
    child2_tiles = np.where(mask[..., None], parent2_tiles, parent1_tiles)

    child1_tiles = ensure_unique_tiles(child1_tiles, original_tiles)
    child2_tiles = ensure_unique_tiles(child2_tiles, original_tiles)

    return creator.Individual(child1_tiles.flatten()), creator.Individual(child2_tiles.flatten())


def two_point_crossover(parent1, parent2, original_tiles):
    parent1_tiles = np.array(parent1).reshape(8, 8, 4)
    parent2_tiles = np.array(parent2).reshape(8, 8, 4)

    pt1, pt2 = np.random.randint(0, 8, size=2)
    if pt1 > pt2:
        pt1, pt2 = pt2, pt1

    child1_tiles = np.copy(parent1_tiles)
    child2_tiles = np.copy(parent2_tiles)

    child1_tiles[pt1:pt2, :, :] = parent2_tiles[pt1:pt2, :, :]
    child2_tiles[pt1:pt2, :, :] = parent1_tiles[pt1:pt2, :, :]

    child1_tiles = ensure_unique_tiles(child1_tiles, original_tiles)
    child2_tiles = ensure_unique_tiles(child2_tiles, original_tiles)

    return creator.Individual(child1_tiles.flatten()), creator.Individual(child2_tiles.flatten())


def adaptive_mutate(puzzle, best_score, tiles):
    puzzle = np.array(puzzle).reshape(8, 8, 4)

    mutation_rate = FINAL_MUTATION_RATE if best_score >= 112 * \
        CONSERVATIVE_FITNESS_THRESHOLD else INITIAL_MUTATION_RATE

    if np.random.rand() < mutation_rate:
        idx = np.random.randint(0, 8, size=(2, 2))
        puzzle[idx[0][0], idx[0][1]], puzzle[idx[1][0], idx[1][1]] = puzzle[idx[1]
                                                                            [0], idx[1][1]].copy(), puzzle[idx[0][0], idx[0][1]].copy()
        puzzle = ensure_unique_tiles(puzzle, tiles)

    return puzzle.flatten()


def run_genetic_algorithm(tiles):
    population = initialize_population(tiles)
    best_solution = None
    best_score = -1

    start_time = time.time()

    for generation in range(GENERATIONS):
        # Sequential fitness evaluation
        fitness_values = [fitness(ind)[0] for ind in population]

        # Using rank selection instead of tournament
        population = rank_selection(population, fitness_values)

        elite_individuals = sorted(population, key=lambda ind: fitness(ind)[
                                   0], reverse=True)[:ELITISM_SIZE]
        new_population = elite_individuals[:]

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)

            # Choose crossover method based on best score
            child1, child2 = two_point_crossover(parent1, parent2, tiles) if best_score >= 112 * \
                CONSERVATIVE_FITNESS_THRESHOLD else uniform_crossover(parent1, parent2, tiles)

            child1 = adaptive_mutate(child1, best_score, tiles)
            child2 = adaptive_mutate(child2, best_score, tiles)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        for puzzle in population:
            score = fitness(puzzle)[0]
            if score > best_score:
                best_score = score
                best_solution = puzzle

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"[{timestamp}] Generation {generation + 1}/{GENERATIONS}: Best Fitness (score) = {best_score}")

    end_time = time.time()
    total_run_time = end_time - start_time
    print(
        f"Total Run Time: {total_run_time:.2f} seconds  |  # Mismatches = {112-best_score}")

    return best_solution.reshape(8, 8, 4)


team_info = "TeamName TeamID1 TeamID2"


def write_output(file_path, solution):
    with open(file_path, 'w') as file:
        file.write(f"{team_info}\n")
        for row in solution:
            line = ' '.join([''.join(map(str, tile)) for tile in row])
            file.write(line + '\n')


if __name__ == "__main__":
    input_file = "Ass1Input.txt"
    output_file = "Ass1Output.txt"

    tiles = read_input(input_file)
    best_solution = run_genetic_algorithm(tiles)
    write_output(output_file, best_solution)

    print("Output written to", output_file)
