import random
import numpy as np

# Constants
POPULATION_SIZE = 1000
GENERATIONS = 1000
MUTATION_RATE = 0.1

# Read the input file and parse the tiles


def read_input(file_path):
    with open(file_path, 'r') as file:
        tiles = []
        for line in file:
            row = line.split()  # Split the line into 4-digit strings
            # Split each 4-digit number into its digits
            tile_edges = [[int(digit) for digit in str(tile)] for tile in row]
            tiles.extend(tile_edges)  # Add the split tiles to the list
    # Convert to numpy array and reshape to (64, 4)
    return np.array(tiles)

# Initialize the population with unique arrangements of tiles


def initialize_population(tiles):
    population = []
    for _ in range(POPULATION_SIZE):
        arrangement = tiles.copy()  # Copy the tiles to shuffle without affecting the original
        np.random.shuffle(arrangement)  # Shuffle the tiles
        grid_arrangement = arrangement.reshape(
            8, 8, 4)  # Reshape into 8x8 grid of tiles
        population.append(grid_arrangement)
    return population

# Fitness function to evaluate the candidate solutions


def fitness(puzzle):
    score = 0
    for i in range(8):
        for j in range(8):
            current_tile = puzzle[i, j]

            # Check right edge (current_tile[1] vs next tile's left edge)
            if j < 7:
                right_tile = puzzle[i, j + 1]
                if current_tile[1] == right_tile[3]:
                    score += 1

            # Check bottom edge (current_tile[2] vs tile below's top edge)
            if i < 7:
                bottom_tile = puzzle[i + 1, j]
                if current_tile[2] == bottom_tile[0]:
                    score += 1
    return score

# Select the best candidates based on fitness


def selection(population):
    scores = [fitness(puzzle) for puzzle in population]
    sorted_population = [x for _, x in sorted(
        zip(scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:POPULATION_SIZE // 2]  # Select top 50%

# Crossover between two parents to create offspring


def crossover(parent1, parent2):
    crossover_point = random.randint(1, 7)
    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Mutate a candidate solution


def mutate(puzzle):
    if random.random() < MUTATION_RATE:
        i1, j1 = random.randint(0, 7), random.randint(0, 7)
        i2, j2 = random.randint(0, 7), random.randint(0, 7)
        # Swap two tiles
        puzzle[i1, j1], puzzle[i2, j2] = puzzle[i2, j2], puzzle[i1, j1]
    return puzzle

# Run the genetic algorithm


def run_genetic_algorithm(tiles):
    population = initialize_population(tiles)
    best_solution = None
    best_score = -1

    for generation in range(GENERATIONS):
        # Evaluate fitness
        population = selection(population)

        # Crossover to create new population
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        # Limit to population size
        population = new_population[:POPULATION_SIZE]

        # Check for the best solution
        for puzzle in population:
            score = fitness(puzzle)
            if score > best_score:
                best_score = score
                best_solution = puzzle

        # Log the best fitness score for the current generation
        print(
            f"Generation {generation + 1}/{GENERATIONS}: Best Fitness (score) = {best_score}")

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
