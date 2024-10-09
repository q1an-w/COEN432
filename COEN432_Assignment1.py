import random
import numpy as np
import pandas as pd
import time  # Import the time module
from datetime import datetime  # For logging timestamps
from deap import tools, base, creator

# Constants
POPULATION_SIZE = 1000
GENERATIONS = 100
MUTATION_RATE = 0.8

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Optimized input file reader using pandas


def read_input(file_path):
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file_path, sep='\s+', header=None)

    # Ensure each tile's digits are split and form a consistent (64, 4) shape
    tiles = []
    for row in df.values:
        for tile in row:
            # Pad the tile to ensure 4 digits
            tile_digits = [int(digit) for digit in f"{tile:04d}"]
            tiles.append(tile_digits)

    # Convert to NumPy array with shape (64, 4)
    return np.array(tiles).reshape(64, 4)

# Initialize the population with unique arrangements of tiles


def initialize_population(tiles):
    population = []
    for _ in range(POPULATION_SIZE):
        arrangement = np.random.permutation(tiles)  # Faster random shuffle
        grid_arrangement = arrangement.reshape(8, 8, 4)
        population.append(creator.Individual(grid_arrangement))
    return population


def fitness(individual):
    ind_tuple = tuple(individual)
    puzzle = np.array(individual).reshape(8, 8, 4)
    right_edges = puzzle[:, :-1, 1] != puzzle[:, 1:, 3]
    bottom_edges = puzzle[:-1, :, 2] != puzzle[1:, :, 0]

    total_mismatches = np.sum(right_edges) + np.sum(bottom_edges)
    fitness_score = 112 - total_mismatches
    return fitness_score,

# Select the best candidates using efficient numpy operations


def selection(population):
    # Get the first element of the fitness score tuple
    fitness_scores = np.array([fitness(puzzle)[0] for puzzle in population])
    best_indices = np.argsort(
        fitness_scores)[-POPULATION_SIZE // 2:]  # Select top 50%
    # Convert to integer indices
    return [population[int(i)] for i in best_indices]

# Destructive two-point crossover using NumPy slicing


def two_point_crossover(parent1, parent2):
    crossover_point1 = random.randint(1, 6)
    crossover_point2 = random.randint(crossover_point1 + 1, 7)
    child1 = np.vstack(
        (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.vstack(
        (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
    return creator.Individual(child1), creator.Individual(child2)

# Uniform crossover using NumPy's efficient indexing


def uniform_crossover(parent1, parent2):
    mask = np.random.rand(8, 8) > 0.5
    child1 = np.where(mask[:, :, None], parent1, parent2)
    child2 = np.where(mask[:, :, None], parent2, parent1)
    return creator.Individual(child1), creator.Individual(child2)

# Mutate a candidate solution


def mutate(puzzle, generation, fitness_score):
    mutation_rate = MUTATION_RATE if generation < 0.75 * \
        GENERATIONS else MUTATION_RATE * (1 - fitness_score)

    if np.random.rand() < mutation_rate:
        # Perform tile swaps
        num_swaps = 2 if generation < 0.75 * GENERATIONS else 1
        for _ in range(num_swaps):
            idx1 = np.random.randint(0, 8, size=2)
            idx2 = np.random.randint(0, 8, size=2)
            puzzle[idx1[0], idx1[1]], puzzle[idx2[0], idx2[1]] = puzzle[idx2[0],
                                                                        idx2[1]].copy(), puzzle[idx1[0], idx1[1]].copy()
    return puzzle

# Run the genetic algorithm


def run_genetic_algorithm(tiles):
    population = initialize_population(tiles)
    best_solution = None
    best_score = -1

    start_time = time.time()  # Record the start time

    for generation in range(GENERATIONS):
        # Selection
        population = selection(population)

        # Crossover phase
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)

            if generation < 0.75 * GENERATIONS:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1, child2 = uniform_crossover(parent1, parent2)

            # Mutate and append to new population
            child1 = mutate(child1, generation, fitness(child1)[0])
            child2 = mutate(child2, generation, fitness(child2)[0])
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        # Track best solution
        for puzzle in population:
            score = fitness(puzzle)[0]
            if score > best_score:
                best_score = score
                best_solution = puzzle

        # Log the best fitness score for the current generation with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"[{timestamp}] Generation {generation + 1}/{GENERATIONS}: Best Fitness (score) = {best_score}")

    end_time = time.time()  # Record the end time
    total_run_time = end_time - start_time

    # Log total run time
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
