import numpy as np
import pandas as pd
from scipy.linalg import eig

def calculate_stationary_distribution(csv_file):
    """
    Calculate the unique stationary distribution of a Markov Chain
    from a CSV file containing the transition matrix.
    
    Args:
        csv_file (str): Path to the CSV file containing the transition matrix
    
    Returns:
        numpy.ndarray: Stationary distribution vector
    """
    
    # Read the transition matrix from CSV file
    try:
        P = pd.read_csv(csv_file, header=None).values
        print(f"Transition matrix P:")
        print(P)
        print()
        
        # Verify that it's a stochastic matrix (rows sum to 1)
        if not np.allclose(P.sum(axis=1), 1):
            print("WARNING: The matrix is not stochastic (rows do not sum to 1)")
            return None
            
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Method 1: Solve the system π = πP using eigenvectors
    print("=== METHOD 1: Eigenvectors ===")
    
    # Transpose P because we want π = πP, so πP^T = π
    # We look for the left eigenvector with eigenvalue 1
    eigenvalues, eigenvectors = eig(P.T)
    
    # Find the index of the eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    stationary_vector = np.real(eigenvectors[:, idx])
    
    # Normalize to get a probability distribution
    stationary_distribution = stationary_vector / stationary_vector.sum()
    
    print(f"Stationary distribution π = {stationary_distribution}")
    print(f"Verification: π·P = {np.dot(stationary_distribution, P)}")
    print(f"Sum of components: {stationary_distribution.sum()}")
    print()
    
    # Method 2: Solve the linear system (I - P^T)π = 0
    print("=== METHOD 2: Linear system ===")
    
    n = P.shape[0]
    # Build the system (I - P^T)π = 0
    # Replace the last equation with the normalization condition
    A = np.eye(n) - P.T
    A[-1, :] = 1  # Normalization condition: sum of probabilities = 1
    b = np.zeros(n)
    b[-1] = 1    # Right-hand side: sum = 1
    
    # Solve the linear system
    pi_linear = np.linalg.solve(A, b)
    
    print(f"Stationary distribution π = {pi_linear}")
    print(f"Verification: π·P = {np.dot(pi_linear, P)}")
    print(f"Sum of components: {pi_linear.sum()}")
    print()
    
    # Method 3: Matrix powers (iterative method)
    print("=== METHOD 3: Matrix powers ===")
    
    # Calculate P^n for large n
    P_power = np.linalg.matrix_power(P, 100)
    # Each row of P^n converges to the stationary distribution
    pi_power = P_power[0, :]  # Take the first row
    
    print(f"Stationary distribution π = {pi_power}")
    print(f"Verification: π·P = {np.dot(pi_power, P)}")
    print(f"Sum of components: {pi_power.sum()}")
    print()
    
    return stationary_distribution

def create_example_csv():
    """
    Create a CSV file with an example transition matrix.
    """
    # Example transition matrix
    P_example = np.array([
        [0.6, 0.2, 0.2],  # State 1 transitions
        [0.4, 0.6, 0],    # State 2 transitions
        [0, 1, 0]         # State 3 transitions (absorbing to state 2)
    ])
    
    # Save to CSV file
    np.savetxt('transition_matrix.csv', P_example, delimiter=',', fmt='%.1f')
    print("File 'transition_matrix.csv' created with example matrix.")
    return 'transition_matrix.csv'

# Main execution example
if __name__ == "__main__":
    print("=== MARKOV CHAIN STATIONARY DISTRIBUTION CALCULATOR ===\n")
    
    # Create the example file
    filename = create_example_csv()
    
    print("\nCalculating stationary distribution:")
    print("-" * 50)
    
    # Calculate the stationary distribution
    distribution = calculate_stationary_distribution(filename)
    
    if distribution is not None:
        print("\n=== FINAL RESULT ===")
        print(f"The unique stationary distribution is:")
        for i, prob in enumerate(distribution):
            print(f"  State {i+1}: {prob:.6f}")
        
        print(f"\nAs percentages:")
        for i, prob in enumerate(distribution):
            print(f"  State {i+1}: {prob*100:.2f}%")