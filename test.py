import random
from deap import base, creator, tools, algorithms
import argparse

# Define a class for Puzzle Pieces


class PuzzlePiece:
    def __init__(self, edges):
        self.edges = edges  # edges is a tuple of 4 edges

    def rotate(self):
        # Rotate the piece 90 degrees clockwise
        self.edges = (self.edges[-1],) + self.edges[:-1]

    def get_edges(self):
        return self.edges


# Define fitness function (minimizing mismatches between adjacent pieces)
def calculate_fitness(individual):
    mismatches = 0
    # Convert individual to 8x8 grid of PuzzlePieces
    puzzle = [individual[i * 8:(i + 1) * 8] for i in range(8)]

    for row in range(8):
        for col in range(8):
            piece = puzzle[row][col]

            # Check right neighbor
            if col < 7:
                right_neighbor = puzzle[row][col + 1]
                # Right edge != Left edge of neighbor
                if piece.get_edges()[1] != right_neighbor.get_edges()[3]:
                    mismatches += 1

            # Check bottom neighbor
            if row < 7:
                bottom_neighbor = puzzle[row + 1][col]
                # Bottom edge != Top edge of neighbor
                if piece.get_edges()[2] != bottom_neighbor.get_edges()[0]:
                    mismatches += 1

    return mismatches,


# Function to create random puzzle pieces (4 edges with 7 possible motifs)
def create_individual():
    pieces = [PuzzlePiece(tuple(random.choices(range(7), k=4)))
              for _ in range(64)]
    # Shuffle the pieces to create a new puzzle configuration
    random.shuffle(pieces)
    return creator.Individual(pieces)  # Return an Individual instance


def mutate_individual(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            if random.random() < 0.5:  # 50% chance to rotate
                individual[i].rotate()
            else:  # 50% chance to swap with another piece
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
    return individual,


def crossover(parent1, parent2):
    # One-point crossover
    point = random.randint(1, len(parent1) - 1)
    child1 = creator.Individual(parent1[:point] + parent2[point:])
    child2 = creator.Individual(parent2[:point] + parent1[point:])
    return child1, child2


# Register GA tools and components using DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate,
                 creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", crossover)  # One-point crossover
# Lower mutation probability
toolbox.register("mutate", mutate_individual, indpb=0.05)
toolbox.register("select", tools.selTournament,
                 tournsize=3)  # Tournament selection
toolbox.register("evaluate", calculate_fitness)


# Main Genetic Algorithm function
def run_ga(pop_size, num_generations):
    population = toolbox.population(n=pop_size)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    stats.register("avg", lambda ind: sum(fit[0] for fit in ind) / len(ind))

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=num_generations,
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True
    )

    return hall_of_fame[0]  # Return the best individual (solution)


# Function to save the best solution to an output file
def save_solution(best_solution, team_info):
    with open("Ass1Output.txt", "w") as f:
        # Write the first line with team information
        f.write(f"{team_info}\n")

        # Write the 8x8 solution in the required format
        for i in range(8):
            line = " ".join(
                "".join(map(str, best_solution[i * 8 + j].get_edges())) for j in range(8)
            )
            f.write(line + "\n")


# Command-line UI for population size and generation input
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-Matching Puzzle Solver")
    parser.add_argument("--pop_size", type=int, default=1000,
                        help="Population size (100-1000)")
    parser.add_argument("--num_generations", type=int,
                        default=50, help="Number of generations (1-100)")
    parser.add_argument("--team_info", type=str,
                        default="TeamName TeamID1 TeamID2", help="Team names and IDs")
    args = parser.parse_args()

    # Run the GA with user-defined parameters
    best_solution = run_ga(pop_size=args.pop_size,
                           num_generations=args.num_generations)

    # Save the best solution to the output file
    save_solution(best_solution, args.team_info)
