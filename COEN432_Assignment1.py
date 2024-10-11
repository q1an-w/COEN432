# Qian Yi Wang (40211303) Philip Carlsson-Coulombe (40208572)
# Best output: 6 mismatches

import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from deap import tools, base, creator
from scipy.spatial.distance import hamming
import uuid

"""
The general strategy here is to first initialize a clamped "diverse" population using hamming distance as a heuristic for distance,
    Then use crossovers & mutations to converge towards a good solution

Selection is a basic tournamenet selecting the top 50% of candidates to perform crossover / mutation on
    at the end, the top 12.5% of indiivduals are retained to the next generation, as well as introducing 12.5 % of random individuals to preserve diversity

Crossover begins as a random 2-point crosssover, until the fitness passes 51, at which point it switches to uniform crossover

Mutation then occrus on the generations children, defined by a random chance that changes based off a few factors.
    firstly, the mutation rate diminishes as fitness increases to preserve good solutions.
    If the best fitness stagnates however, a bonus is applied to the mutation rate as the stagnation increases to try and break the cycle
    A mutation is first 3 operations (either a swap or a rotation), becomes 1 operation once best fitness passes a certain threshold
"""
# Default Population Size and Generations, can be modified by user input
POPULATION_SIZE = 1000
GENERATIONS = 100

# CONSTANTS
MAX_MUTATION_RATE = 0.73
MIN_MUTATION_RATE = 0.15
# The fitness score at which to switch crossover strategies (2-point crossover -> uniform crossover)
FITNESS_THRESHOLD = 51
# Percentage of elite individuals to carry over to next generation
ELITE_PERCENTAGE = 0.125
# Base percentage of random individuals to introduce to each new population (preserver diversity)
BASE_RANDOM_PERCENTAGE = 0.125


# Number of generations for each stagnation level
STAGNATION_LIMITS = [2, 5, 8]
RANDOM_INCREMENT = [0.05, 0.175, 0.25]  # Increment for each stagnation level

# Define a maximum cap for the mutation rate based on stagnation
MAX_MUTATION_RATE_BONUS = 0.4  # Maximum additional bonus to the mutation rate
STAGNATION_BONUS_SCALING = 0.05  # Scaling array for the stagnation counter bonus

# Minimum Hamming distance (set as a ratio of differing positions between arrays (max is 265))
MIN_HAMMING_DISTANCE = 185
# Maximum Hamming distance (set as a ratio of differing positions between arrays (max is 265))
MAX_HAMMING_DISTANCE = 215

# DEAP setup (we use DEAP to manage individual tile sets (ensures no extra / weird tile duplication))
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Reads Inputfile into Tile Set


def read_input(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None)
    tiles = []
    for row in df.values:
        for tile in row:
            tile_digits = [int(digit) for digit in f"{tile:04d}"]
            tiles.append(tile_digits)
    return np.array(tiles).reshape(64, 4)

# Generate a random hash for the individual using UUID


def generate_random_hash():
    return str(uuid.uuid4())


# Calculate the Hamming distance between two individuals using SciPy.


def hamming_distance(ind1, ind2):
    flat_ind1 = ind1.flatten()
    flat_ind2 = ind2.flatten()
    # hamming from scipy returns proportion of differing elements
    return hamming(flat_ind1, flat_ind2) * len(flat_ind1)


# Initializes population given the input tiles


def initialize_population(tiles):
    population = []
    hamming_distances_map = {}  # Dictionary to store hashes of individuals

    start_time = time.time()  # Record the start time

    while len(population) < POPULATION_SIZE:
        arrangement = np.random.permutation(tiles)
        grid_arrangement = arrangement.reshape(8, 8, 4)
        new_individual = creator.Individual(grid_arrangement)
        new_hash = generate_random_hash()
        # Check if the new individual has a sufficient Hamming distance from the existing population
        if new_hash not in hamming_distances_map:
            # Only check Hamming distances against individuals already in the population to avoid  O(n^3)
            if all(hamming_distance(new_individual, ind) > MIN_HAMMING_DISTANCE or hamming_distance(new_individual, ind) < MAX_HAMMING_DISTANCE for ind in population):
                population.append(new_individual)
                hamming_distances_map[new_hash] = new_individual

    end_time = time.time()  # Record the end time
    total_run_time = end_time - start_time
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"[{timestamp}] Finished Initializing Population (size {POPULATION_SIZE}) in {total_run_time:.2f} seconds")
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

# Generate a random individual (introduce diversity forcefully alter on) (avoid hills)


def generate_random_individual(tiles):
    arrangement = np.random.permutation(tiles)
    grid_arrangement = arrangement.reshape(8, 8, 4)
    return creator.Individual(grid_arrangement)


# Two-point crossover ensuring valid tiles during swap (making use of creator.Individual to enforce tiles are good (check using check validity python script to check input and output files are both "tile equal")
# use vstack here for 2 points since multi array combine

def two_point_crossover(parent1, parent2):
    crossover_point1 = random.randint(1, 6)
    crossover_point2 = random.randint(crossover_point1 + 1, 7)
    # Create child arrays with swapped sections1000
    child1 = np.vstack(
        (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.vstack(
        (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
    return creator.Individual(child1), creator.Individual(child2)


# Uniform crossover ensuring valid tiles
# use vstack for 1 point crossover
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

# Swap tiles


def swap_tiles(puzzle, idx1, idx2):
    new_puzzle = puzzle.copy()
    new_puzzle[idx1], new_puzzle[idx2] = new_puzzle[idx2].copy(
    ), new_puzzle[idx1].copy()
    return new_puzzle


# Mutate a candidate solution


def mutate(puzzle, stagnation_counter, fitness_score):
    # Calculate the base mutation rate
    if fitness_score < 112:  # Only scale if fitness is less than the max
        mutation_rate = MAX_MUTATION_RATE * (1 - (fitness_score / 112))
        mutation_rate = max(mutation_rate, MIN_MUTATION_RATE)
    else:
        mutation_rate = MIN_MUTATION_RATE  # Apply minimum rate if fitness is maximized

    # Calculate the mutation bonus based on stagnation counter
    mutation_bonus = min(stagnation_counter *
                         STAGNATION_BONUS_SCALING, MAX_MUTATION_RATE_BONUS)

    # Adjust the mutation rate by adding the mutation bonus
    mutation_rate += mutation_bonus
    # Cap the mutation rate
    mutation_rate = min(mutation_rate, MAX_MUTATION_RATE)

    if np.random.rand() < mutation_rate:
        # mutate a bit less agreesively after itness threshold
        num_mutations = 3 if fitness_score < FITNESS_THRESHOLD else 1
        for _ in range(num_mutations):
            action = random.choice(['swap', 'rotate'])
            idx1 = np.random.randint(0, 8, size=2)

            if action == 'swap':
                idx2 = np.random.randint(0, 8, size=2)
                puzzle = swap_tiles(puzzle, tuple(idx1), tuple(idx2))
            elif action == 'rotate':
                rotations = np.random.randint(1, 4)  # Rotate 1 to 3 times
                tile_to_rotate = puzzle[tuple(idx1)].copy()
                rotated_tile = rotate_tile(tile_to_rotate, rotations)
                puzzle[tuple(idx1)] = rotated_tile

    return puzzle


# Run the genetic algorithm


def run_genetic_algorithm(tiles):
    # INTIALIZATION
    start_time = time.time()  # Record the start time
    population = initialize_population(tiles)
    best_solution = None
    best_fitness = -1
    stagnation_counter = 0  # Counter for stagnation
    # Start with base random percentage
    current_random_percentage = BASE_RANDOM_PERCENTAGE

    for generation in range(GENERATIONS):
        # SELECTION
        population = selection(population)

        # PRESERVE ELITES %
        elite_count = int(ELITE_PERCENTAGE * POPULATION_SIZE)
        elites = population[-elite_count:]  # Keep top elite_count individuals

        # Generate random individuals based on current random percentage
        random_count = int(current_random_percentage * POPULATION_SIZE)
        random_individuals = [generate_random_individual(
            tiles) for _ in range(random_count)]

       # CROSSOVER
        # Start new population with elites and randoms
        new_population = list(elites) + random_individuals
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)

            # Switch to uniform crossover based on fitness score
            if fitness(parent1)[0] < FITNESS_THRESHOLD and fitness(parent2)[0] < FITNESS_THRESHOLD:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1, child2 = uniform_crossover(parent1, parent2)

            # MUTATION
            child1 = mutate(child1, stagnation_counter, fitness(child1)[0])
            child2 = mutate(child2, stagnation_counter, fitness(child2)[0])
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        # Track best solution
        prev_best_score = best_fitness
        for puzzle in population:
            score = fitness(puzzle)[0]
            if score > best_fitness:
                best_fitness = score
                best_solution = puzzle
                prev_best_score = 0  # Reset stagnation counter on fitness betterment

        if best_fitness > prev_best_score:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        # Adjust random individuals amount based on stagnation
        if stagnation_counter >= STAGNATION_LIMITS[0] and stagnation_counter < STAGNATION_LIMITS[1]:
            current_random_percentage = BASE_RANDOM_PERCENTAGE + \
                RANDOM_INCREMENT[0]
        elif stagnation_counter >= STAGNATION_LIMITS[1] and stagnation_counter < STAGNATION_LIMITS[2]:
            current_random_percentage = BASE_RANDOM_PERCENTAGE + \
                RANDOM_INCREMENT[1]
        elif stagnation_counter >= STAGNATION_LIMITS[2]:
            current_random_percentage = BASE_RANDOM_PERCENTAGE + \
                RANDOM_INCREMENT[2]

        # Log the current best fitness score
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Generation {generation + 1}/{GENERATIONS}: Best Fitness (score) = {best_fitness} | Stagnation : {stagnation_counter}")

    end_time = time.time()  # Record the end time
    total_run_time = end_time - start_time

    # Log total run time
    print(
        f"Total Run Time: {total_run_time:.2f} seconds | Mismatches: {112 - best_fitness}")

    return best_solution


# Write the output to a file
team_info = "Qian Yi Wang 40211303 Philip Carlsson-Coulombe 40208572"


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

    # Input validation for population amount
    while True:
        try:
            POPULATION_SIZE = int(
                input("Enter the population size (between 1 and 1000): "))
            if 1 <= POPULATION_SIZE <= 1000:
                break
            else:
                print("Please enter a population size between 1 and 1000.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # Input validation for number of generations
    while True:
        try:
            GENERATIONS = int(
                input("Enter the number of generations (between 1 and 100): "))
            if 1 <= GENERATIONS <= 100:
                break
            else:
                print("Please enter a number of generations between 1 and 100.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    tiles = read_input(input_file)
    best_solution = run_genetic_algorithm(tiles)
    write_output(output_file, best_solution)
