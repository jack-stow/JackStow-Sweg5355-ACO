import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import itertools
import time
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

NETWORK_SEED = 42

# Modify this function to generate a random distance matrix
def generate_distance_matrix(num_nodes):
    # Generate a random distance matrix with values between 1 and 20
    distance_matrix = np.random.randint(1, 20, size=(num_nodes, num_nodes))
    # Set the diagonal to zero (distance from a node to itself)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

class AntColony:
    def __init__(self, distance_matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distance_matrix = distance_matrix
        self.pheromone_matrix = np.ones_like(distance_matrix, dtype=float)  # Initialize as float matrix
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones(distance_matrix.shape, dtype=float) / len(distance_matrix)
        self.best_path = None
        self.best_distance = float('inf')
        self.paths_history = []

    def get_pheromone_level(self, path):
        pheromone_strengths = [
            self.pheromone_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)
        ]
        return min(pheromone_strengths) #type: ignore

    def run(self):
        for iteration in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            #if iteration % 5 == 0:  # Save only every 5th iteration for visualization
            self.paths_history.append(all_paths)
            self.best_path, self.best_distance = self.spread_pheromone(all_paths, self.best_path, self.best_distance)
            self.pheromone *= self.decay  # Evaporation

        return self.best_path, self.best_distance

    def spread_pheromone(self, all_paths, best_path, best_distance):
        for path in all_paths:
            distance = self.calculate_distance(path)
            if distance < best_distance:
                best_distance = distance
                best_path = path
            
            for i in range(len(path) - 1):
                self.pheromone[path[i], path[i + 1]] += 1.0 / distance

        return best_path, best_distance

    def calculate_distance(self, path):
        return sum(self.distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))

    def gen_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.gen_path()
            if len(path) == len(self.distance_matrix):  # Ensure all nodes are visited
                all_paths.append(path)
        return all_paths

    def gen_path(self):
        n_nodes = len(self.distance_matrix)
        path = []
        visited = set()
        current_node = random.randint(0, n_nodes - 1)
        path.append(current_node)
        visited.add(current_node)

        while len(path) < n_nodes:
            probabilities = self.calculate_probabilities(current_node, visited)
            next_node = int(np.random.choice(range(n_nodes), p=probabilities))
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node

        return path

    def calculate_probabilities(self, current_node, visited):
        pheromone = self.pheromone[current_node].copy()
        visited_indices = list(visited)
        pheromone[visited_indices] = 0  # Remove visited nodes
        heuristic = 1 / (self.distance_matrix[current_node] + 1e-10)
        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        return probabilities / probabilities.sum()

# Visualization Function
def visualize_aco(aco, path_index=0):
    G = nx.from_numpy_array(aco.distance_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=NETWORK_SEED)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)
    max_pheromone_value = np.max(aco.pheromone_matrix)

    # Get the specific path set to visualize based on path_index
    if path_index < len(aco.paths_history):
        path_set = aco.paths_history[path_index]
        for path in path_set:
            for i in range(len(path) - 1):
                start_node = path[i]
                end_node = path[i + 1]
                pheromone_strength = aco.pheromone_matrix[start_node, end_node]
                norm_strength = np.clip(pheromone_strength / max_pheromone_value, 0, 1)
                color = cm.plasma(norm_strength)

                ax.plot([pos[start_node][0], pos[end_node][0]], 
                        [pos[start_node][1], pos[end_node][1]], 
                        color=color, linewidth=1, alpha=0.7, zorder=0)
    
    ax.set_title(f'Ant Colony Optimization - Path at Iteration {path_index}')
    plt.axis('off')
    plt.show()



# Brute-force TSP function
def brute_force_tsp(distance_matrix):
    num_nodes = len(distance_matrix)
    all_nodes = range(num_nodes)
    
    # Initialize variables to store the best path and distance
    tsp_best_path = None
    best_distance = float('inf')
    
    # Generate all possible paths (permutations of nodes)
    for perm in itertools.permutations(all_nodes):
        # Calculate the total distance for this path
        distance = sum(distance_matrix[perm[i], perm[i + 1]] for i in range(num_nodes - 1))
        distance += distance_matrix[perm[-1], perm[0]]  # Return to the start node
        
        # Update the best path and distance if this one is shorter
        if distance < best_distance:
            best_distance = distance
            tsp_best_path = perm
    
    return list(tsp_best_path), best_distance


def calculate_distance(distance_matrix, path):
    distance = sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
    distance += distance_matrix[path[-1], path[0]]  # Return to the starting node
    return distance




# Example usage
num_nodes = 35  # Keep this small for brute force
# distance_matrix = np.loadtxt('distance_matrix.txt', delimiter=',')#generate_distance_matrix(num_nodes)
distance_matrix = generate_distance_matrix(num_nodes)
print(distance_matrix)
np.savetxt(f'output/distance_matrix-{num_nodes}-nodes.txt', distance_matrix, fmt='%d', delimiter=',')
print(f"Distance matrix saved as distance_matrix-{num_nodes}-nodes.txt")

# Measure time for ACO
start_time = time.time()
aco = AntColony(distance_matrix, n_ants=20, n_best=10, n_iterations=200, decay=0.95, alpha=1, beta=2)
aco_best_path, best_distance = aco.run()
aco_duration = time.time() - start_time
aco_distance = calculate_distance(distance_matrix, aco_best_path)
print("ACO Best path:", aco_best_path)
print("ACO Best distance:", aco_distance)
print(f"ACO Duration: {aco_duration:.4f} seconds")
print("ACO History Length:", len(aco.paths_history))


# Measure time for brute-force TSP
# start_time = time.time()
# tsp_best_path, tsp_best_distance = brute_force_tsp(distance_matrix)
# tsp_duration = time.time() - start_time
# tsp_distance = calculate_distance(distance_matrix, tsp_best_path)
# print("TSP Best path:", tsp_best_path)
# print("TSP Best distance:", tsp_best_distance)
# print(f"TSP Duration: {tsp_duration:.4f} seconds")


# Visualization Function with Color Gradient
def animate_aco_paths(aco, interval):
    G = nx.from_numpy_array(aco.distance_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=NETWORK_SEED)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)
    plt.axis('off')

    # Define colormap for gradient (using rainbow spectrum)
    cmap = plt.get_cmap("rainbow")
    num_edges = len(aco.distance_matrix) - 1
    colors = cmap(np.linspace(0, 1, num_edges))  # Generate evenly spaced colors

    def update(frame):
        ax.clear()
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)
        
        path = aco.paths_history[frame][0]  # Get the first path from this iteration's paths
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            color = colors[i % num_edges]  # Get the color for this edge
            ax.plot(
                [pos[start_node][0], pos[end_node][0]], 
                [pos[start_node][1], pos[end_node][1]], 
                color=color, linewidth=2, zorder=0
            )

        # Adding title and iteration number
        ax.set_title(f'Ant Colony Optimization - Iteration {frame + 1}/{len(aco.paths_history)}')

    ani = FuncAnimation(fig, update, frames=len(aco.paths_history), repeat=False, interval=interval)
    plt.show()
    plt.show()


# Run the animated visualizer
#animate_aco_paths(aco, interval=100)


# Visualization Function for a Static Path with Color Gradient
def visualize_solution_path(distance_matrix, solution_path):
    G = nx.from_numpy_array(distance_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=NETWORK_SEED)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)
    plt.axis('off')

    # Define colormap for gradient (using rainbow spectrum)
    cmap = plt.get_cmap("rainbow")
    num_edges = len(solution_path) - 1
    colors = cmap(np.linspace(0, 1, num_edges))  # Generate evenly spaced colors

    # Plot the solution path with gradient colors
    for i in range(len(solution_path) - 1):
        start_node = solution_path[i]
        end_node = solution_path[i + 1]
        color = colors[i % num_edges]  # Get the color for this edge
        ax.plot(
            [pos[start_node][0], pos[end_node][0]], 
            [pos[start_node][1], pos[end_node][1]], 
            color=color, linewidth=2, zorder=0
        )

    # Connect the last node back to the first to complete the tour
    start_node = solution_path[-1]
    end_node = solution_path[0]
    ax.plot(
        [pos[start_node][0], pos[end_node][0]], 
        [pos[start_node][1], pos[end_node][1]], 
        color=colors[-1], linewidth=2, zorder=0
    )

    # Set title to indicate it's a TSP solution path
    ax.set_title('TSP Solution Path Visualization')
    plt.show()

# Example usage
# Assuming `distance_matrix` is defined and `solution_path` is a list of node indices in the optimal order
#visualize_solution_path(distance_matrix, aco_best_path)

#visualize_solution_path(distance_matrix, tsp_best_path)
# Run the visualizer
#visualize_aco(aco, path_index=5)


def animate_best_path(aco, interval):
    G = nx.from_numpy_array(aco.distance_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=NETWORK_SEED)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)
    plt.axis('off')

    # Define colormap for gradient (using rainbow spectrum)
    cmap = plt.get_cmap("rainbow")
    num_edges = len(aco.best_path)
    colors = cmap(np.linspace(0, 1, num_edges))  # Generate evenly spaced colors

    def update(frame):
        ax.clear()
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)

        # Add edges up to the current frame
        for i in range(frame + 1):
            start_node = aco.best_path[i]
            end_node = aco.best_path[(i + 1) % num_edges]  # Wrap around to form a cycle
            color = colors[i % num_edges]  # Get the color for this edge
            ax.plot(
                [pos[start_node][0], pos[end_node][0]],
                [pos[start_node][1], pos[end_node][1]],
                color=color, linewidth=2, zorder=0
            )

        # Add title with current frame number
        ax.set_title(f'ACO Path Animation - Step {frame + 1}/{num_edges}')

    ani = FuncAnimation(fig, update, frames=num_edges, repeat=False, interval=interval)
    plt.show()

# Run the animated visualizer
animate_best_path(aco, interval=500)
