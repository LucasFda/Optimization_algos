import math
import random
import matplotlib.pyplot as plt

# Simulated Annealing Algorithm
def simulated_annealing(objective_function, initial_solution, neighbor_function, initial_temperature, cooling_rate, stop_temperature):
    current_solution = initial_solution
    current_cost = objective_function(current_solution)
    temperature = initial_temperature

    best_solution = current_solution
    best_cost = current_cost

    costs = []  # Track convergence

    while temperature > stop_temperature:
        # Generate a neighboring solution
        neighbor = neighbor_function(current_solution)
        neighbor_cost = objective_function(neighbor)

        # Calculate the cost difference
        cost_diff = neighbor_cost - current_cost

        # Accept the neighbor based on probability
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_solution = neighbor
            current_cost = neighbor_cost

            # Update the best solution
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

        # Cool down the temperature
        temperature *= cooling_rate
        costs.append(current_cost)  # Track cost over iterations

    return best_solution, best_cost, costs

# Test and Visualization
def test_simulated_annealing():
    # Define the objective function
    def objective_function(x):
        return (x - 2) ** 2 + 3  # Minimum at x = 2, f(x) = 3

    # Define the neighbor function
    def neighbor_function(x):
        return x + random.uniform(-1, 1)

    # Parameters
    initial_solution = random.uniform(-10, 10)
    initial_temperature = 100
    cooling_rate = 0.95
    stop_temperature = 0.1

    # Execute the simulated annealing algorithm
    best_solution, best_cost, costs = simulated_annealing(
        objective_function,
        initial_solution,
        neighbor_function,
        initial_temperature,
        cooling_rate,
        stop_temperature
    )

    # Results
    print(f"Best solution: {best_solution}")
    print(f"Best cost: {best_cost}")

    # Plot the convergence
    plt.plot(range(len(costs)), costs)
    plt.title("Convergence of Simulated Annealing")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

# Run the test
if __name__ == "__main__":
    test_simulated_annealing()
