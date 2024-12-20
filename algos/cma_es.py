import cma
import matplotlib.pyplot as plt
import numpy as np

# CMA-ES Algorithm
def cma_es(objective_function, bounds, generations, population_size, sigma0):
    # Calculate the initial mean (center of the bounds)
    x0 = [(bounds[0] + bounds[1]) / 2]

    # Initialize the CMA-ES optimizer
    options = {
        'popsize': population_size,  # Population size
        'bounds': bounds,           # Search space bounds
        'maxiter': generations,     # Maximum number of generations
        'seed': None                # Random seed (None makes it random each run)
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    
    best_fitness = []
    
    while not es.stop():
        # Sample a new population
        solutions = es.ask()
        # Apply a small random perturbation to solutions (optional for variability)
        solutions = [[x[0] + np.random.uniform(-1, 1)] for x in solutions]
        
        # Evaluate the objective function for each solution
        fitness = [objective_function(x[0]) for x in solutions]
        # Update the optimizer with evaluated fitness
        es.tell(solutions, [-f for f in fitness])  # CMA-ES minimizes, so we negate fitness
        # Track the best fitness
        best_fitness.append(max(fitness))
    
    # Get the best solution found
    best_solution = es.result.xbest
    best_value = -es.result.fbest  # Convert back for maximization

    return best_solution, best_value, best_fitness

# Test and Visualization
def test_cma_es():
    # Define the objective function
    def objective_function(x):
        return -(x**2) + 4 * x + 10  # Maximum at x = 2, f(x) = 14

    # Parameters
    bounds = [-20, 20]        # Broader search bounds
    generations = 100         # Maximum number of generations
    population_size = 2       # Smaller population size for more variability
    sigma0 = 7.0              # Larger initial standard deviation for exploration

    # Execute the CMA-ES algorithm
    best_solution, best_fitness, fitness_history = cma_es(
        objective_function,
        bounds,
        generations,
        population_size,
        sigma0
    )

    # Results
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fitness_history)), fitness_history, label="CMA-ES Convergence")
    plt.title("Convergence of CMA-ES")
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.show()

# Run the test
if __name__ == "__main__":
    test_cma_es()


