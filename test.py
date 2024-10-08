import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from deap import tools, base, creator

# Constants
POPULATION_SIZE = 1000
GENERATIONS = 100
INITIAL_MUTATION_RATE = 0.9  # Higher initial mutation rate
FINAL_MUTATION_RATE = 0.05    # Lower final mutation rate

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Read the input file


def read_input(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None)
    tiles = []
    for row in df.values:
        for tile in row:
            tile_digits = [int(digit) for digit in f"{tile:04d}"]
            tiles.append(tile_digits)
    return np.array(tiles).reshape(64, 4)

# Function to randomly rotate a tile


def random_rotate(tile):
    return np.roll(tile, random.randint(0, 3))  # Rotate the tile randomly

# Initialize population with diverse tile arrangements


def initialize_population(tiles):
    population = set()  # Use a set to ensure uniqueness

    while len(population) < POPULATION_SIZE:
        arrangement = np.random.permutation(tiles)  # Shuffle tiles
        grid_arrangement = arrangement.reshape(8, 8, 4)

        # Randomly rotate each tile for uniqueness
        for i in range(8):
            for j in range(8):
                grid_arrangement[i][j] = random_rotate(grid_arrangement[i][j])

        # Convert the entire grid arrangement to a tuple of tuples
        individual = tuple(
            tuple(tuple(grid_arrangement[i][j]) for j in range(8)) for i in range(8))
        population.add(individual)  # Add the individual to the set

    # Convert the set back to individuals
    return [creator.Individual(np.array(individual)) for individual in population]

# Evaluate fitness based on edge matches


def fitness(puzzle):
    score = 0
    edge_weight = 2.0
    mismatch_penalty = -1.0

    # Check right and bottom edge matches
    right_edges = puzzle[:, :-1, 1] == puzzle[:, 1:, 3]
    bottom_edges = puzzle[:-1, :, 2] == puzzle[1:, :, 0]

    score += np.sum(right_edges) * edge_weight
    score += np.sum(bottom_edges) * edge_weight
    score += np.sum(~right_edges) * mismatch_penalty
    score += np.sum(~bottom_edges) * mismatch_penalty

    return score

# Selection using rank selection


def rank_selection(population):
    # Sort population based on fitness
    ranked_population = sorted(population, key=fitness, reverse=True)

    # Assign probabilities based on rank
    rank_sum = sum(range(len(ranked_population) + 1))
    probabilities = np.arange(len(ranked_population), 0, -1) / rank_sum

    # Select individuals based on their probabilities
    selected_indices = np.random.choice(
        len(ranked_population), size=len(population)//2, p=probabilities)
    selected = [ranked_population[i]
                for i in selected_indices]  # Use indices to select individuals
    return selected  # Ensure it is a list

# Crossover to create new individuals


def two_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, 6)
    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    return creator.Individual(child1), creator.Individual(child2)

# Mutation function


def mutate(puzzle, generation):
    # Dynamically adjust mutation rate
    mutation_rate = INITIAL_MUTATION_RATE * \
        (1 - generation / GENERATIONS) + \
        FINAL_MUTATION_RATE * (generation / GENERATIONS)

    if np.random.rand() < mutation_rate:
        # Perform tile orientation changes
        num_mutations = 4 if generation < 0.5 * \
            GENERATIONS else 1  # More mutations initially
        for _ in range(num_mutations):
            # Choose a random tile to mutate
            row = np.random.randint(0, 8)  # Random row
            col = np.random.randint(0, 8)  # Random column

            # Rotate the selected tile
            tile = puzzle[row, col]
            # Rotate the tile 90 degrees clockwise
            puzzle[row, col] = np.array(
                [tile[3], tile[0], tile[1], tile[2]])  # Rotate right
    return puzzle

# Run the genetic algorithm


def run_genetic_algorithm(tiles):
    population = initialize_population(tiles)
    best_solution = None
    best_score = -1

    start_time = time.time()

    for generation in range(GENERATIONS):
        population = rank_selection(population)

        # Crossover phase
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = two_point_crossover(parent1, parent2)

            # Mutate children with dynamic mutation strategy
            new_population.extend(
                [mutate(child1, generation), mutate(child2, generation)])

        # Apply elitism: keep the best solution from the previous generation
        best_individual = max(population, key=fitness)
        # Leave space for the best individual
        new_population = new_population[:POPULATION_SIZE - 1]
        new_population.append(best_individual)  # Add the best individual

        population = new_population[:POPULATION_SIZE]

        # Track best solution
        for puzzle in population:
            score = fitness(puzzle)
            if score > best_score:
                best_score = score
                best_solution = puzzle

        # Log best fitness score for current generation
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"[{timestamp}] Generation {generation + 1}/{GENERATIONS}: Best Fitness = {best_score}")

    end_time = time.time()
    print(f"Total Run Time: {end_time - start_time:.2f} seconds")

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
