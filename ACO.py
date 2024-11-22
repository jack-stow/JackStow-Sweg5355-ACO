import numpy as np  # For numerical operations
import random  # For generating random numbers
import matplotlib.pyplot as plt  # For visualization
#import networkx as nx  # For creating and visualizing graphs
# import matplotlib.cm as cm  # For colormaps in visualizations
# from matplotlib.colors import Normalize  # For normalizing color values
# from matplotlib.animation import FuncAnimation
from sympy import false  # For creating animations

def generate_distance_matrix_from_file(file_path: str) -> np.ndarray:
    """
    Reads a distance matrix from a text file and returns it as a numpy array.
    Assumes the file is formatted correctly (each line contains comma-separated integers).
    """
    # Initialize an empty list to hold the rows of the distance matrix
    distance_matrix = []
    
    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by commas, convert to integers, and append to the matrix
            distance_matrix.append(list(map(int, line.strip().split(','))))
    
    # Convert the list of lists into a numpy array
    return np.array(distance_matrix)

#from ACO_Visualizer import animate_paths_history, animate_best_path, animate_pheromone_history, visualize_solution_path


# Generate a random distance matrix for the Traveling Salesman Problem (TSP)
def generate_distance_matrix(num_nodes) -> np.ndarray: 
    """
    Creates a symmetric distance matrix where each entry represents the distance
    between two nodes, and the diagonal entries are zero (distance to itself).
    """
    distance_matrix = np.random.randint(1, 20, size=(num_nodes, num_nodes))  # Random distances [1, 20)
    np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0 (distance to self)
    return distance_matrix

class AntColony:
    """
    Class implementing the Ant Colony Optimization (ACO) algorithm for solving the TSP.
    """
    def __init__(self, distance_matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=1, verbose=0):
        """
        Initialize the ACO parameters and data structures.

        Parameters:
        - distance_matrix: The distance matrix for the TSP.
        - n_ants: Number of ants in the simulation.
        - n_best: Number of best-performing ants contributing pheromones.
        - n_iterations: Number of iterations to run the simulation.
        - decay: Evaporation rate of pheromones.
        - alpha: Pheromone importance factor.
        - beta: Heuristic importance factor.
        """
        self.distance_matrix = distance_matrix
        self.n_ants = n_ants
        # The code snippet is setting the value of the variable `self.n_best` to the value of the
        # variable `n_best`.
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones(distance_matrix.shape, dtype=float) / len(distance_matrix)  # Normalized pheromones
        self.pheromone_history = []
        self.pheromone_history.append(np.copy(self.pheromone))
        self.best_path = None  # Best path found
        self.best_distance = float('inf')  # Best distance found (initialize to infinity)
        self.paths_history = []  # History of paths generated
        self.verbose = verbose

    def run(self):
        """
        Run the ACO algorithm for the specified number of iterations.
        """
        for iteration in range(self.n_iterations):
            all_paths = self.gen_all_paths()  # Generate paths for all ants
            
            # Ensure all paths are valid (visit all nodes exactly once)
            for path in all_paths:
                assert len(set(path)) == len(path), f"Path has repeated nodes: {path}"

            
            #self.paths_history.append(np.copy(all_paths))  # Save paths for visualization
            self.best_path, self.best_distance = self.spread_pheromone(all_paths, self.best_path, self.best_distance)
            self.pheromone *= self.decay  # Apply pheromone evaporation
            self.pheromone_history.append(np.copy(self.pheromone))
            if self.verbose > 1:
                print(f"Pheromone Matrix (after):\n{self.pheromone}")

        return self.best_path, self.best_distance

    def spread_pheromone(self, all_paths, best_path, best_distance):
        """
        Update pheromone levels based on the paths generated in the current iteration.
        """
        # Sort paths by distance
        sorted_paths = sorted(all_paths, key=lambda path: self.calculate_distance(path))
        
        self.paths_history.append(np.copy(sorted_paths[0]))
        
        # Only consider the n_best paths
        for path in sorted_paths[:self.n_best]:
            distance = self.calculate_distance(path)
            
            if distance < best_distance:
                best_distance = distance
                best_path = path
            
            # Add pheromone to the edges in the path
            for i in range(len(path) - 1):
                self.pheromone[path[i], path[i + 1]] += 1.0 / distance

        return best_path, best_distance

    def calculate_distance(self, path):
        """
        Calculate the total distance of a given path.
        """
        return sum(self.distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))

    def gen_all_paths(self):
        """
        Generate paths for all ants in the colony.
        """
        all_paths = []
        for _ in range(self.n_ants):
            path = self.gen_path()  # Generate a path for an individual ant
            if len(path) == len(self.distance_matrix):  # Ensure all nodes are visited
                all_paths.append(np.copy(path))
        return all_paths

    def gen_path(self):
        """
        Generate a path for a single ant using probabilistic rules.
        """
        n_nodes = len(self.distance_matrix)
        path = []
        visited = set()  # Track visited nodes
        current_node = random.randint(0, n_nodes - 1)  # Start at a random node
        path.append(np.copy(current_node))
        visited.add(current_node)

        while len(path) < n_nodes:
            probabilities = self.calculate_probabilities(current_node, visited)  # Calculate probabilities for next node
            next_node = int(np.random.choice(range(n_nodes), p=probabilities))  # Choose next node
            path.append(np.copy(next_node))
            visited.add(next_node)
            current_node = next_node

        return path

    def calculate_probabilities(self, current_node, visited):
        """
        Calculate the probabilities for choosing the next node based on pheromone levels and heuristic information.
        """
        pheromone = self.pheromone[current_node].copy()
        visited_indices = list(visited)
        pheromone[visited_indices] = 0  # Exclude visited nodes
        heuristic = 1 / (self.distance_matrix[current_node] + 1e-10)  # Heuristic: inverse of distance
        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)  # Combine pheromone and heuristic factors
        if self.verbose > 1:
            print(f"Current Node: {current_node}, Probabilities: {probabilities}")

        return probabilities / probabilities.sum()




def calculate_distance(distance_matrix, path):
    """
    Calculates the total distance of a given path.

    Parameters:
        distance_matrix (numpy.ndarray): A matrix where the entry (i, j) represents the distance between node i and node j.
        path (list): A list of node indices representing the path.

    Returns:
        float: The total distance of the given path.

    Functionality:
        - Sums distances for consecutive nodes in the path.
        - Includes the distance from the last node back to the first to form a complete tour.
    """
    distance = sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
    distance += distance_matrix[path[-1], path[0]]  # Return to the starting node
    return distance
