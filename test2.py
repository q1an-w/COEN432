import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from deap import tools, base, creator
from scipy.spatial.distance import hamming
import uuid

# Constants for the Genetic Algorithm
POPULATION_SIZE = 1000           # Number of individuals in the population
GENERATIONS = 100                # Number of generations to run the algorithm for
# Maximum mutation rate (how often mutations occur)
MAX_MUTATION_RATE = 0.73
MIN_MUTATION_RATE = 0.15         # Minimum mutation rate
FITNESS_THRESHOLD = 51           # Fitness threshold to switch crossover strategies
# Percentage of elite individuals to keep in each generation
ELITE_PERCENTAGE = 0.125
BASE_RANDOM_PERCENTAGE = 0.125   # Base percentage of randomly generated individuals

# Stagnation thresholds to increase mutation rates
STAGNATION_LIMITS = [2, 5, 8]
# Increase in random percentage with stagnation
RANDOM_INCREMENT = [0.05, 0.175, 0.25]

# Hamming distance constraints to maintain diversity in the population
MIN_HAMMING_DISTANCE = 185
MAX_HAMMING_DISTANCE = 215

# DEAP setup for creating individuals and defining fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Function to read puzzle input file using pandas for efficiency


def read_input(file_path):
    # Read input as a dataframe
    df = pd.read_csv(file_path, sep='\s+', header=None)
    tiles = []  # To store the tiles
    # Convert each row of the dataframe into individual tiles
    for row in df.values:
        for tile in row:
            # Convert tile into 4-digit array
            tile_digits = [int(digit) for digit in f"{tile:04d}"]
            tiles.append(tile_digits)
    # Reshape into 64 tiles, each with 4 sides
    return np.array(tiles).reshape(64, 4)

# Function to generate a random unique identifier (hash) for each individual


def generate_random_hash():
    return str(uuid.uuid4())  # UUID ensures uniqueness for hashing purposes

# Function to compute Hamming distance between two individuals (how different they are)


def hamming_distance(ind1, ind2):
    flat_ind1 = ind1.flatten()  # Flatten the grid into 1D array
    flat_ind2 = ind2.flatten()
    # Use scipy's hamming function to calculate the proportion of differing elements
    # Convert proportion to absolute distance
    var = hamming(flat_ind1, flat_ind2) * len(flat_ind1)
    return var

# Initialize population of individuals with diverse arrangements of tiles


def initialize_population(tiles):
    population = []
    hamming_distances_map = {}  # To store hashes and check for uniqueness

    start_time = time.time()  # Record start time for performance tracking

    # Populate until the population reaches the desired size
    while len(population) < POPULATION_SIZE:
        arrangement = np.random.permutation(
            tiles)  # Randomly permute the tiles
        grid_arrangement = arrangement.reshape(
            8, 8, 4)  # Reshape into 8x8 grid
        new_individual = creator.Individual(
            grid_arrangement)  # Create individual
        new_hash = generate_random_hash()  # Generate unique hash for this individual

        # Ensure sufficient Hamming distance from all other individuals
        if new_hash not in hamming_distances_map:
            if all(hamming_distance(new_individual, ind) > MIN_HAMMING_DISTANCE or
                   hamming_distance(new_individual, ind) < MAX_HAMMING_DISTANCE for ind in population):
                # Add individual to population
                population.append(new_individual)
                # Track individual in hash map
                hamming_distances_map[new_hash] = new_individual

    end_time = time.time()  # Record end time
    total_run_time = end_time - start_time  # Calculate total time taken

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Timestamp for logging
    print(f"[{timestamp}] Finished Initializing Population in {total_run_time:.2f} seconds")

    return population  # Return the initialized population

# Fitness function: counts the number of correct edges in the puzzle


def fitness(individual):
    # Reshape the individual into an 8x8 grid
    puzzle = np.array(individual).reshape(8, 8, 4)
    # Check mismatches on the right edges
    right_edges = puzzle[:, :-1, 1] != puzzle[:, 1:, 3]
    # Check mismatches on the bottom edges
    bottom_edges = puzzle[:-1, :, 2] != puzzle[1:, :, 0]
    total_mismatches = np.sum(right_edges) + \
        np.sum(bottom_edges)  # Total mismatches
    # 112 is the max number of correct edges
    fitness_score = 112 - total_mismatches
    return fitness_score,

# Selection function: selects the top 50% individuals based on fitness


def selection(population):
    # Calculate fitness for the population
    fitness_scores = np.array([fitness(puzzle)[0] for puzzle in population])
    best_indices = np.argsort(
        fitness_scores)[-POPULATION_SIZE // 2:]  # Select the top 50%
    # Return the selected population
    return [population[i] for i in best_indices]

# Generate a new random individual for population


def generate_random_individual(tiles):
    arrangement = np.random.permutation(tiles)  # Random permutation of tiles
    grid_arrangement = arrangement.reshape(8, 8, 4)  # Reshape into an 8x8 grid
    return creator.Individual(grid_arrangement)  # Return the new individual

# Two-point crossover between two parents


def two_point_crossover(parent1, parent2):
    crossover_point1 = random.randint(1, 6)  # First crossover point
    crossover_point2 = random.randint(
        crossover_point1 + 1, 7)  # Second crossover point

    # Create two children by swapping sections between parents
    child1 = np.vstack(
        (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.vstack(
        (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))

    return creator.Individual(child1), creator.Individual(child2)

# Uniform crossover between two parents


def uniform_crossover(parent1, parent2):
    mask = np.random.rand(8, 8) > 0.5  # Generate a random mask
    # Swap tiles between parents based on the mask to generate two children
    child1 = np.where(mask[:, :, None], parent1, parent2)
    child2 = np.where(mask[:, :, None], parent2, parent1)

    return creator.Individual(child1), creator.Individual(child2)

# Rotate a tile by a certain number of 90-degree rotations


def rotate_tile(tile, rotations):
    rotation_mapping = {
        0: tile,  # No rotation
        1: [tile[3], tile[0], tile[1], tile[2]],  # 1 clockwise rotation
        2: [tile[2], tile[3], tile[0], tile[1]],  # 2 clockwise rotations
        3: [tile[1], tile[2], tile[3], tile[0]],  # 3 clockwise rotations
    }
    return rotation_mapping[rotations]

# Swap two tiles in the puzzle


def swap_tiles(puzzle, idx1, idx2):
    new_puzzle = puzzle.copy()  # Create a copy to avoid modifying the original
    new_puzzle[idx1], new_puzzle[idx2] = new_puzzle[idx2].copy(
    ), new_puzzle[idx1].copy()  # Swap tiles
    return new_puzzle  # Return the modified puzzle


# Mutate a puzzle by swapping or rotating tiles
MAX_MUTATION_RATE_BONUS = 0.4  # Maximum bonus to mutation rate
# Scaling factor for mutation rate bonus based on stagnation
STAGNATION_BONUS_SCALING = 0.05


def mutate(puzzle, stagnation_counter, fitness_score):
    # Calculate base mutation rate: lower fitness leads to higher mutation rate
    if fitness_score < 112:
        mutation_rate = MAX_MUTATION_RATE * \
            (1 - (fitness_score / 112))  # Scale mutation rate by fitness
        # Ensure it doesn't go below the minimum
        mutation_rate = max(mutation_rate, MIN_MUTATION_RATE)
    else:
        mutation_rate = MIN_MUTATION_RATE  # Apply minimum rate if fitness is perfect

    # Add a bonus to the mutation rate based on how long the algorithm has stagnated
    mutation_bonus = min(stagnation_counter *
                         STAGNATION_BONUS_SCALING, MAX_MUTATION_RATE_BONUS)
    mutation_rate += mutation_bonus  # Add bonus to mutation rate
    # Ensure it doesn't exceed the max rate
    mutation_rate = min(mutation_rate, MAX_MUTATION_RATE)

    # Apply mutation based on the calculated rate
    if np.random.rand() < mutation_rate:
        # More mutations for low fitness
        num_mutations = 3 if fitness_score < FITNESS_THRESHOLD else 1
        for _ in range(num_mutations):
            idx1 = (random.randint(0, 7), random.randint(0, 7))  # Random tile
            # Another random tile
            idx2 = (random.randint(0, 7), random.randint(0, 7))

            if random.random() > 0.5:  # 50% chance of rotating the tile
                # Random number of 90-degree rotations
                rotations = random.randint(1, 3)
                puzzle[idx1] = rotate_tile(
                    puzzle[idx1], rotations)  # Rotate a tile
            else:  # Otherwise, swap two tiles
                puzzle = swap_tiles(puzzle, idx1, idx2)  # Swap tiles

    return creator.Individual(puzzle)  # Return the mutated puzzle

# Main Genetic Algorithm function to evolve the population


def run_genetic_algorithm(population, tiles):
    stagnation_counter = 0  # To track how many generations have had no improvement
    last_best_fitness = 0  # The best fitness score from the last generation

    # Loop over generations
    for gen in range(GENERATIONS):
        print(f"\nGeneration {gen + 1}/{GENERATIONS}")

        # Calculate fitness for all individuals
        fitness_scores = np.array([fitness(ind)[0] for ind in population])

        # Find the best individual in this generation
        best_fitness = np.max(fitness_scores)
        best_individual = population[np.argmax(fitness_scores)]

        # Handle stagnation by adjusting mutation rate if fitness doesn't improve
        if best_fitness > last_best_fitness:
            stagnation_counter = 0  # Reset counter if there's improvement
        else:
            stagnation_counter += 1  # Increment if no improvement

        last_best_fitness = best_fitness  # Update best fitness from last generation

        # Selection step: keep top 50% of the population
        elite_population = selection(population)

        # Randomly generate new individuals to maintain diversity
        random_percentage = BASE_RANDOM_PERCENTAGE  # Start with base random percentage
        # If stagnation has occurred, increase the number of random individuals
        for i, stagnation_limit in enumerate(STAGNATION_LIMITS):
            if stagnation_counter > stagnation_limit:
                # Increment random percentage based on stagnation
                random_percentage += RANDOM_INCREMENT[i]
        random_population = [generate_random_individual(
            tiles) for _ in range(int(random_percentage * POPULATION_SIZE))]

        # Perform crossover to create new offspring
        crossover_population = []
        crossover_type = "Uniform" if best_fitness < FITNESS_THRESHOLD else "Two-Point"
        for _ in range(POPULATION_SIZE - len(elite_population) - len(random_population)):
            # Randomly select two parents
            parent1, parent2 = random.sample(elite_population, 2)
            # Apply crossover based on current strategy
            if crossover_type == "Uniform":
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1, child2 = two_point_crossover(parent1, parent2)
            crossover_population.append(child1)
            crossover_population.append(child2)

        # Apply mutation to the crossover population
        mutated_population = []
        for child in crossover_population:
            mutated_population.append(
                mutate(child, stagnation_counter, fitness(child)[0]))

        # Form the new population by combining elite, random, and mutated individuals
        population = elite_population + random_population + \
            mutated_population[:POPULATION_SIZE -
                               len(elite_population) - len(random_population)]

        # Log progress
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Best Fitness: {best_fitness:.2f}, Mismatches: }, Stagnation: {stagnation_counter}")

    return best_individual, best_fitness


# Main execution block
if __name__ == "__main__":
    # Path to the input file containing tile data
    INPUT_FILE_PATH = "Ass1Input.txt"

    # Read and process input file to get the tiles
    tiles = read_input(INPUT_FILE_PATH)

    # Initialize population
    population = initialize_population(tiles)

    # Run the genetic algorithm to evolve the solution
    best_solution, best_fitness = run_genetic_algorithm(population, tiles)

    # Output the final best solution
    print("\nFinal Best Solution:")
    print(best_solution)
    print(f"Final Best Fitness: {best_fitness:.2f}")
