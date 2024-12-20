import numpy as np
import matplotlib.pyplot as plt

# ADAM Algorithm
def adam_optimizer(objective_function, gradient_func, initial_x, learning_rate, beta1, beta2, epsilon, max_iter):
    x = initial_x
    m = 0  # Momentum
    v = 0  # RMSProp
    trajectory = []  # Track cost for visualization

    for t in range(1, max_iter + 1):
        g = gradient_func(x)  # Compute gradient
        m = beta1 * m + (1 - beta1) * g  # Update biased first moment
        v = beta2 * v + (1 - beta2) * (g**2)  # Update biased second moment
        m_hat = m / (1 - beta1**t)  # Correct bias for m
        v_hat = v / (1 - beta2**t)  # Correct bias for v
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # Update parameter

        # Record cost for visualization
        trajectory.append(objective_function(x))

    return x, trajectory

# Test and Visualization
def test_adam_optimizer():
    # Define the objective function
    def objective_function(x):
        return x**2 + 2*x + 1  # Minimum at x = -1, f(x) = 0

    # Define the gradient
    def gradient_func(x):
        return 2*x + 2

    # Parameters
    initial_x = -5  # Initial point
    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    max_iter = 1000

    # Execute the ADAM algorithm
    best_x, costs = adam_optimizer(
        objective_function,
        gradient_func,
        initial_x,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        max_iter
    )

    # Results
    print(f"Solution optimale trouvée : x = {best_x}, f(x) = {objective_function(best_x)}")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(costs)), costs, label="ADAM Convergence")
    plt.title("Convergence de l'optimiseur ADAM")
    plt.xlabel("Itérations")
    plt.ylabel("Coût (f(x))")
    plt.legend()
    plt.show()

# Run the test
if __name__ == "__main__":
    test_adam_optimizer()
