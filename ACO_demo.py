from ACO import AntColony, generate_distance_matrix, calculate_distance,  generate_distance_matrix_from_file
import time  # For measuring execution time
from ACO_Visualizer import animate_paths_history, animate_best_path, animate_pheromone_history, visualize_solution_path
import numpy as np  # For numerical operations
from TSP_bruteforce import brute_force_tsp

def ACO_demo(distance_matrix=None, num_nodes=20, n_ants=10, n_best=2, n_iterations=100, decay=0.5, alpha=2, beta=3, verbose=0, visualize=True, step=1, interval=100, save_matrix = False, threshold=0):
    # Example usage
    #num_nodes = 20  # Keep this small for brute force
    # distance_matrix = np.loadtxt('distance_matrix.txt', delimiter=',')#generate_distance_matrix(num_nodes)
    
    if verbose > 0:
        print(f"num_nodes={num_nodes}, n_ants={n_ants}, n_best={n_best}, n_iterations={n_iterations}, decay={decay}, alpha={alpha}, beta={beta}")
    if distance_matrix is None:
        distance_matrix = generate_distance_matrix(num_nodes)
    #print(distance_matrix)
    if save_matrix:
        np.savetxt(f'output/distance_matrix-{num_nodes}-nodes.txt', distance_matrix, fmt='%d', delimiter=',')
        print(f"Distance matrix saved as distance_matrix-{num_nodes}-nodes.txt")

    if threshold == 0:
        print("="*100)
        # Measure time for ACO
        start_time = time.time()
        aco = AntColony(distance_matrix, n_ants=n_ants, n_best=n_best, n_iterations=n_iterations, decay=decay, alpha=alpha, beta=beta, verbose=verbose)
        aco_best_path, best_distance = aco.run()
        aco_duration = time.time() - start_time
        aco_distance = calculate_distance(distance_matrix, aco_best_path)
        print("ACO Best path:", aco_best_path)
        print("ACO Best distance:", aco_distance)
        print(f"ACO Duration: {aco_duration:.4f} seconds")
    else:
        aco_distance = float('inf')
        
        while aco_distance > threshold:
            print("="*100)
            # Measure time for ACO
            start_time = time.time()
            aco = AntColony(distance_matrix, n_ants=n_ants, n_best=n_best, n_iterations=n_iterations, decay=decay, alpha=alpha, beta=beta, verbose=verbose)
            aco_best_path, best_distance = aco.run()
            aco_duration = time.time() - start_time
            aco_distance = calculate_distance(distance_matrix, aco_best_path)
            print("ACO Best path:", aco_best_path)
            print("ACO Best distance:", aco_distance)
            print(f"ACO Duration: {aco_duration:.4f} seconds")
    
    # Assuming `distance_matrix` is defined and `solution_path` is a list of node indices in the optimal order
    #visualize_solution_path(distance_matrix, aco_best_path)
    
    if visualize:
        # Run the animated visualizer
        animate_best_path(aco, interval=interval)
        animate_pheromone_history(aco, step=step, interval=interval)
        animate_paths_history(aco, step=step, interval=interval)
    return aco_distance, aco_best_path
    
def TSP_brute_force_demo(distance_matrix=None, num_nodes=20, visualize=False):
    print("="*100)
    if distance_matrix is None:
        distance_matrix = generate_distance_matrix(num_nodes)
    # Measure time for brute-force TSP
    bruteforce_start_time = time.time()
    tsp_best_path, tsp_best_distance = brute_force_tsp(distance_matrix)
    tsp_duration = time.time() - bruteforce_start_time
    #tsp_distance = calculate_distance(distance_matrix, tsp_best_path)
    if visualize:
        visualize_solution_path(distance_matrix, tsp_best_path)
    print("TSP Best path:", tsp_best_path)
    print("TSP Best distance:", tsp_best_distance)
    print(f"TSP Duration: {tsp_duration:.4f} seconds")
    print("="*100)


# Parameter tuning for ACO depends on graph size and complexity, but here are some general guidelines:

# Number of Ants (n_ants):

# Small graphs (10-30 nodes): 10-20 ants
# Medium graphs (30-100 nodes): 30-50 ants
# Large graphs (100+ nodes): 50-100 ants

# n_best:

# Typically 10-20% of total ants
# For 50 ants, consider 5-10 best ants

# n_iterations:

# Small graphs: 50-100 iterations
# Medium graphs: 100-200 iterations
# Large graphs: 200-500 iterations

# decay:

# Recommended range: 0.1 to 0.5
# Lower values (0.1-0.2) for more persistent pheromone trails
# Higher values (0.4-0.5) for more exploration

# alpha and beta:

# Typical ranges: 1-5 for both
# Higher alpha (2-3): more emphasis on previous successful paths
# Higher beta (3-5): stronger preference for shorter routes

# generate_distance_matrix will generate a random graph with n nodes. 
#distance_matrix = generate_distance_matrix(50)
#generate_distance_matrix_from_file will open a pre-generated graph
distance_matrix = generate_distance_matrix_from_file("output/distance_matrix-10-nodes.txt")
num_nodes = distance_matrix.shape[0]
# num_nodes = 14
# distance_matrix = generate_distance_matrix(num_nodes=num_nodes)
print("="*100)
# 10-node matrix shortest path is 31. 
# 11-node matrix shortest path is 36, takes 80.4 seconds
# 12-node matrix shortest path is 33, takes 1096.5 seconds
# 13-node matrix would take nearly 4 hours to brute force. shortest path i've gotten: 26 in 0.0414 seconds, [ 0  4 10  6  9  5  1 11  8 12  7  2  3], num_nodes=13, n_ants=5, n_best=1, n_iterations=25, decay=0.85, alpha=2, beta=4
#TSP_brute_force_demo(distance_matrix)

# ACO_demo(
#     distance_matrix=distance_matrix, 
#     num_nodes=num_nodes, 
#     n_ants=5, 
#     n_best=1, 
#     n_iterations=25, 
#     decay=0.85, 
#     alpha=2, 
#     beta=4, 
#     verbose=1, 
#     visualize=False,
#     step=1,
#     interval=50#, save_matrix=True
#     )
# aco_distance = 26
# while (aco_distance >= 26):
#     aco_distance, aco_best_path = ACO_demo(
#     distance_matrix=distance_matrix, 
#     num_nodes=num_nodes, 
#     n_ants=8, 
#     n_best=1, 
#     n_iterations=50, 
#     decay=0.85, 
#     alpha=2, 
#     beta=4, 
#     verbose=1, 
#     visualize=False,
#     step=1,
#     interval=50
#     )
# print(f"Distance: {aco_distance}")

# print(f"Best Path: {aco_best_path}")
ACO_demo(
    distance_matrix=distance_matrix, 
    num_nodes=num_nodes, 
    n_ants=15, 
    n_best=5, 
    n_iterations=75, 
    decay=0.87, 
    alpha=4, 
    beta=4, 
    verbose=1, 
    visualize=True,
    step=1,
    interval=150,
    threshold=30
    )


# ACO_demo(
#     distance_matrix=distance_matrix, 
#     num_nodes=num_nodes, 
#     n_ants=20, 
#     n_best=5, 
#     n_iterations=100, 
#     decay=0.85, 
#     alpha=2, 
#     beta=4, 
#     verbose=1, 
#     visualize=False,
#     step=1,
#     interval=50
#     )
