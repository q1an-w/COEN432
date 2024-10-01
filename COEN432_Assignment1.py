import random
import numpy as np

# Constants
POPULATION_SIZE = 1000
GENERATIONS = 100
MUTATION_RATE = 0.8

# Read the input file and parse the tiles


def read_input(file_path):
    with open(file_path, 'r') as file:
        tiles = []
        for line in file:
            row = line.split()  # Split the line into 4-digit strings
            tile_edges = [[int(digit) for digit in str(tile)] for tile in row]
            tiles.extend(tile_edges)  # Add the split tiles to the list
    return np.array(tiles)  # Convert to numpy array and reshape to (64, 4)

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


def fitness(puzzle):
    score = 0
    edge_weight = 2.0  # Weight for matches
    mismatch_penalty = -1.0  # Penalty for mismatches
    contiguous_bonus = 1.0  # Bonus for both edges matching

    for i in range(8):
        for j in range(8):
            current_tile = puzzle[i, j]
            match_count = 0

            # Check right edge (current_tile[1] vs right_tile[3])
            if j < 7:
                right_tile = puzzle[i, j + 1]
                if current_tile[1] == right_tile[3]:
                    score += edge_weight  # Increment score for match
                    match_count += 1
                else:
                    score += mismatch_penalty  # Decrement score for mismatch

            # Check bottom edge (current_tile[2] vs bottom_tile[0])
            if i < 7:
                bottom_tile = puzzle[i + 1, j]
                if current_tile[2] == bottom_tile[0]:
                    score += edge_weight  # Increment score for match
                    match_count += 1
                else:
                    score += mismatch_penalty  # Decrement score for mismatch

            # Reward for both edges matching
            if match_count == 2:
                score += contiguous_bonus  # Increment score for contiguous matches

    return score


# Select the best candidates based on fitness


def selection(population):
    scores = [fitness(puzzle) for puzzle in population]
    sorted_population = [x for _, x in sorted(
        zip(scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:POPULATION_SIZE // 2]  # Select top 50%

# Destructive two-point crossover


def two_point_crossover(parent1, parent2):
    crossover_point1 = random.randint(1, 6)
    crossover_point2 = random.randint(crossover_point1 + 1, 7)
    child1 = np.vstack(
        (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.vstack(
        (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
    return child1, child2

# Uniform crossover


def uniform_crossover(parent1, parent2):
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)
    for i in range(8):
        for j in range(8):
            if random.random() > 0.5:
                child1[i, j] = parent1[i, j]
                child2[i, j] = parent2[i, j]
            else:
                child1[i, j] = parent2[i, j]
                child2[i, j] = parent1[i, j]
    return child1, child2

# Mutate a candidate solution


def mutate(puzzle, generation, fitness_score):
    # Adaptive mutation rate
    mutation_rate = MUTATION_RATE if generation < 0.75 * \
        GENERATIONS else MUTATION_RATE * (1 - fitness_score)

    if random.random() < mutation_rate:
        # Higher mutation rates for first 75% of generations
        # More swaps in early generations
        swaps = 2 if generation < 0.75 * GENERATIONS else 1
        for _ in range(swaps):
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
        # Selection
        population = selection(population)

        # Crossover phase based on the generation
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)

            if generation < 0.75 * GENERATIONS:
                # First 75%: Use destructive two-point crossover
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                # Last 25%: Use uniform crossover
                child1, child2 = uniform_crossover(parent1, parent2)

            # Mutate the children
            child1 = mutate(child1, generation, fitness(child1))
            child2 = mutate(child2, generation, fitness(child2))

            new_population.extend([child1, child2])

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
