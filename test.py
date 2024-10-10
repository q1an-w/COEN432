import numpy as np
import random
import pandas as pd
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Configuration
N_TILES = 64  # Number of tiles (8x8 grid)
MUTATION_RATE = 0.1
MUTATION_AMOUNT = 1  # Maximum number of mutations
TOURNAMENT_SIZE = 3
NUM_GENERATIONS = 100
ELITE_SIZE = 2  # Number of elite individuals to keep

# Load tile data from file (implement this function based on your input structure)


def load_tiles(file_name):
    # Placeholder for loading tile data
    # Return a list of tiles (or other data structures needed)
    # Example: replace with actual loading logic
    return [i for i in range(N_TILES)]

# Fitness evaluation function


def evaluate(individual, tiles):
    # Define the fitness function here
    # Assume each tile has a 'top', 'bottom', 'left', 'right' property
    # Calculate the number of mismatches based on the arrangement
    mismatches = 0
    for row in range(8):
        for col in range(8):
            tile = individual[row * 8 + col]
            if row > 0:  # Check above
                if tiles[individual[(row - 1) * 8 + col]]['bottom'] != tiles[tile]['top']:
                    mismatches += 1
            if col > 0:  # Check left
                if tiles[individual[row * 8 + (col - 1)]]['right'] != tiles[tile]['left']:
                    mismatches += 1
    # Fitness score is inverted, lower mismatch is better
    return (N_TILES - mismatches,)

# Mutation function


def mutate(individual):
    # Apply controlled mutation with a fixed mutation amount
    for _ in range(random.randint(1, MUTATION_AMOUNT)):
        idx1, idx2 = random.sample(range(N_TILES), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual,

# Crossover function


def crossover(parent1, parent2):
    # Use uniform crossover with a restoration process
    child1 = list(parent1)
    child2 = list(parent2)
    for i in range(N_TILES):
        if random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]

    # Restoration to ensure valid tiles
    # Assuming we can implement a tile restoration logic
    return child1, child2

# Main genetic algorithm function


def genetic_algorithm(file_name):
    # Load tiles
    tiles = load_tiles(file_name)

    # Setup DEAP framework
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(N_TILES), N_TILES)
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, tiles=tiles)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Create initial population
    population = toolbox.population(n=50)

    # Run the genetic algorithm
    fits = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fits):
        ind.fitness.values = fit

    for gen in range(NUM_GENERATIONS):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - ELITE_SIZE)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_RATE:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the fitness of the offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit

        # Replace the old population by the offspring
        population[:] = offspring + tools.selBest(population, ELITE_SIZE)

        # Optional: Log the best individual of the generation
        fits = [ind.fitness.values[0] for ind in population]
        print(f"Generation {gen}: Best Fitness = {min(fits)}")

    # Output the best solution found
    best_individual = tools.selBest(population, 1)[0]
    print("Best Individual:", best_individual)
    print("Best Fitness:", best_individual.fitness.values[0])
    return best_individual


# Execute the genetic algorithm
if __name__ == "__main__":
    input_file = "Ass1Input.txt"
    output_file = "Ass1Output.txt"

    best_solution = genetic_algorithm(input_file)

    with open(output_file, "w") as f:
        f.write(f"Best Individual: {best_solution}\n")
        f.write(f"Best Fitness: {best_solution.fitness.values[0]}\n")
