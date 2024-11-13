import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from deap import tools, base, creator
from scipy.spatial.distance import hamming
import uuid

# Default Population Size and Generations
POPULATION_SIZE = 1000
GENERATIONS = 100

# Constants
MAX_MUTATION_RATE = 0.73
MIN_MUTATION_RATE = 0.15
FITNESS_THRESHOLD = 51
ELITE_PERCENTAGE = 0.125
BASE_RANDOM_PERCENTAGE = 0.125
STAGNATION_LIMITS = [2, 5, 8, 100]
RANDOM_INCREMENT = [0.05, 0.175, 0.25, 0.625]
MAX_MUTATION_RATE_BONUS = 0.4
STAGNATION_BONUS_SCALING = 0.05
MIN_HAMMING_DISTANCE = 100  # 185
MAX_HAMMING_DISTANCE = 300  # 215

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Reads input file


def read_input(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None)
    tiles = [list(map(int, f"{tile:04d}"))
             for row in df.values for tile in row]
    return np.array(tiles).reshape(64, 4)

# Generate a random hash for the individual using UUID


def generate_random_hash():
    return str(uuid.uuid4())

# Calculate the Hamming distance between two individuals


def hamming_distance(ind1, ind2):
    return hamming(ind1.flatten(), ind2.flatten()) * len(ind1.flatten())

# Initializes population given the input tiles


def initialize_population(tiles):
    population = []
    hamming_distances_map = {}

    start_time = time.time()

    while len(population) < POPULATION_SIZE:
        arrangement = np.random.permutation(tiles).reshape(8, 8, 4)
        new_individual = creator.Individual(arrangement)
        new_hash = generate_random_hash()

        if new_hash not in hamming_distances_map:
            if all(MIN_HAMMING_DISTANCE <= hamming_distance(new_individual, ind) <= MAX_HAMMING_DISTANCE for ind in population):
                population.append(new_individual)
                hamming_distances_map[new_hash] = new_individual

    end_time = time.time()
    print(f"Population initialized in {end_time - start_time:.2f} seconds")
    return population

# Counts correct edges since max # good edges is 112


def fitness(individual):
    puzzle = np.array(individual).reshape(8, 8, 4)
    right_edges = puzzle[:, :-1, 1] != puzzle[:, 1:, 3]
    bottom_edges = puzzle[:-1, :, 2] != puzzle[1:, :, 0]
    mismatches = np.sum(right_edges) + np.sum(bottom_edges)
    return 112 - mismatches,

# Select the best candidates using numpy for faster computation


def selection(population):
    fitness_scores = np.array([fitness(ind)[0] for ind in population])
    best_indices = np.argsort(fitness_scores)[-POPULATION_SIZE // 2:]
    return [population[i] for i in best_indices]

# Generate a random individual (introduce diversity)


def generate_random_individual(tiles):
    return creator.Individual(np.random.permutation(tiles).reshape(8, 8, 4))

# Two-point crossover


def two_point_crossover(parent1, parent2):
    crossover_point1 = random.randint(1, 6)
    crossover_point2 = random.randint(crossover_point1 + 1, 7)
    child1 = np.vstack(
        (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.vstack(
        (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
    return creator.Individual(child1), creator.Individual(child2)

# Uniform crossover


def uniform_crossover(parent1, parent2):
    mask = np.random.rand(8, 8) > 0.5
    child1 = np.where(mask[:, :, None], parent1, parent2)
    child2 = np.where(mask[:, :, None], parent2, parent1)
    return creator.Individual(child1), creator.Individual(child2)

# Rotate a tile


def rotate_tile(tile, rotations):
    return np.roll(tile, rotations)

# Swap tiles


def swap_tiles(puzzle, idx1, idx2):
    puzzle[idx1], puzzle[idx2] = puzzle[idx2].copy(), puzzle[idx1].copy()
    return puzzle

# Mutate a candidate solution


def mutate(puzzle, stagnation_counter, fitness_score):
    mutation_rate = max(MIN_MUTATION_RATE, MAX_MUTATION_RATE * (1 - fitness_score / 112) +
                        min(stagnation_counter * STAGNATION_BONUS_SCALING, MAX_MUTATION_RATE_BONUS))

    if np.random.rand() < mutation_rate:
        num_mutations = 3 if fitness_score < FITNESS_THRESHOLD else 1
        for _ in range(num_mutations):
            action = random.choice(['swap', 'rotate'])
            idx1 = tuple(np.random.randint(0, 8, size=2))

            if action == 'swap':
                idx2 = tuple(np.random.randint(0, 8, size=2))
                puzzle = swap_tiles(puzzle, idx1, idx2)
            elif action == 'rotate':
                puzzle[idx1] = rotate_tile(puzzle[idx1], random.randint(1, 3))

    return puzzle

# Run the genetic algorithm


def run_genetic_algorithm(tiles):
    population = initialize_population(tiles)
    best_solution, best_fitness = None, -1
    stagnation_counter = 0
    current_random_percentage = BASE_RANDOM_PERCENTAGE

    for generation in range(GENERATIONS):
        population = selection(population)
        elite_count = int(ELITE_PERCENTAGE * POPULATION_SIZE)
        elites = population[-elite_count:]
        random_individuals = [generate_random_individual(tiles) for _ in range(
            int(current_random_percentage * POPULATION_SIZE))]
        new_population = elites + random_individuals

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)
            if fitness(parent1)[0] < FITNESS_THRESHOLD and fitness(parent2)[0] < FITNESS_THRESHOLD:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1, child2 = uniform_crossover(parent1, parent2)

            child1 = mutate(child1, stagnation_counter, fitness(child1)[0])
            child2 = mutate(child2, stagnation_counter, fitness(child2)[0])
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        prev_best_score = best_fitness
        for puzzle in population:
            score = fitness(puzzle)[0]
            if score > best_fitness:
                best_fitness = score
                best_solution = puzzle
                prev_best_score = 0

        stagnation_counter = 0 if best_fitness > prev_best_score else stagnation_counter + 1

        if stagnation_counter >= STAGNATION_LIMITS[0]:
            current_random_percentage = BASE_RANDOM_PERCENTAGE + \
                RANDOM_INCREMENT[0]

        print(
            f"Generation {generation + 1}/{GENERATIONS}: Best Fitness = {best_fitness}, Stagnation = {stagnation_counter}")

    print(f"Final Best Fitness = {best_fitness}")
    return best_solution


# Write the output to a file
team_info = "Qian Yi Wang 40211303 Philip Carlsson-Coulombe 40208572"


def write_output(file_path, solution):
    with open(file_path, 'w') as file:
        file.write(f"{team_info}\n")
        for row in solution:
            file.write(' '.join([''.join(map(str, tile))
                       for tile in row]) + '\n')


# Main execution
if __name__ == "__main__":
    input_file = "Ass1Input.txt"
    output_file = "Ass1Output.txt"
    tiles = read_input(input_file)
    best_solution = run_genetic_algorithm(tiles)
    write_output(output_file, best_solution)
