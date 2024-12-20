import random
import matplotlib.pyplot as plt

# Genetic Algorithm with Performance Tracking
def genetic_algorithm(objective_function, generate_individual, mutate, crossover, population_size, generations, mutation_rate):
    # Initialize population
    population = [generate_individual() for _ in range(population_size)]
    fitness = [objective_function(ind) for ind in population]
    best_fitness = []

    for generation in range(generations):
        # Select parents (tournament selection)
        selected_parents = [
            max(random.sample(population, 2), key=objective_function) for _ in range(population_size)
        ]

        # Apply crossover
        children = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_parents[i], selected_parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            children.extend([child1, child2])

        # Apply mutation
        for child in children:
            if random.random() < mutation_rate:
                mutate(child)

        # Evaluate new population
        population = children
        fitness = [objective_function(ind) for ind in population]
        best_fitness.append(max(fitness))  # Track best fitness over generations

    # Return the best solution
    best_idx = fitness.index(max(fitness))
    return population[best_idx], fitness[best_idx], best_fitness

# Test and Visualization
def test_genetic_algorithm():
    # Define the objective function
    def objective_function(x):
        return -(x**2) + 4 * x + 10  # Maximum at x = 2, f(x) = 14

    # Define individual generation
    def generate_individual():
        return random.uniform(-10, 10)

    # Define mutation
    def mutate(ind):
        ind += random.uniform(-1, 1)

    # Define crossover
    def crossover(p1, p2):
        return (0.5 * p1 + 0.5 * p2, 0.5 * p2 + 0.5 * p1)

    # Parameters
    population_size = 100
    generations = 25
    mutation_rate = 0.5

    # Execute the genetic algorithm
    best_solution, best_unique_fitness, best_fitness = genetic_algorithm(
        objective_function,
        generate_individual,
        mutate,
        crossover,
        population_size,
        generations,
        mutation_rate
    )

    # Results
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_unique_fitness}")

    # Plot the convergence
    plt.plot(range(len(best_fitness)), best_fitness, label="Genetic Algorithm Convergence")
    plt.title("Convergence of Genetic Algorithm")
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.show()

# Run the test
if __name__ == "__main__":
    test_genetic_algorithm()