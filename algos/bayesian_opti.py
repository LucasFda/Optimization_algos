import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real

# Bayesian Optimization Algorithm
def bayesian_optimization(objective_function, search_space, n_calls, n_initial_points, base_estimator='RF', acquisition_function="LCB"):
    # Run Bayesian Optimization
    result = gp_minimize(
        func=objective_function,     # Function to minimize
        dimensions=search_space,     # Search space
        n_calls=n_calls,             # Total number of function evaluations
        n_initial_points=n_initial_points,  # Random points before Bayesian model
        base_estimator=base_estimator,  # Surrogate model
        acq_func=acquisition_function  # Acquisition function
    )
    return result

# Test and Visualization
def test_bayesian_optimization():
    # Define the objective function (negative because gp_minimize minimizes by default)
    def objective_function(x):
        return -(x[0]**2) + 4 * x[0] + 10  # Maximum at x = 2, f(x) = 14

    # Define the search space
    search_space = [Real(-5, 5, name="x")]

    # Parameters for Bayesian Optimization
    n_calls = 200           # Total number of evaluations
    n_initial_points = 50   # Increase the initial exploration points

    # Execute the Bayesian Optimization
    result = bayesian_optimization(
        objective_function, 
        search_space, 
        n_calls, 
        n_initial_points, 
        base_estimator='RF',  # Random Forest
        acquisition_function="LCB"  # Lower Confidence Bound
    )

    # Results
    best_solution = result.x[0]
    best_value = -result.fun  # Convert back for maximization
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_value}")
    #print("Points évalués (x) :", result.x_iters)
    #print("Valeurs correspondantes (f(x)) :", [-val for val in result.func_vals])

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(result.func_vals)), -result.func_vals, label="Bayesian Optimization Convergence")
    plt.title("Convergence of Bayesian Optimization")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()

# Run the test
if __name__ == "__main__":
    test_bayesian_optimization()


