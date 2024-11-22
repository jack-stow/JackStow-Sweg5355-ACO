
import itertools  # For generating permutations (used in brute-force TSP)

# Brute-force TSP function
def brute_force_tsp(distance_matrix):
    """
    Solves the Traveling Salesperson Problem using a brute-force approach.

    Parameters:
        distance_matrix (numpy.ndarray): A matrix where the entry (i, j) represents the distance between node i and node j.

    Returns:
        tuple: A tuple containing the best path as a list of node indices and the total distance of this path.

    Functionality:
        - Iterates through all permutations of nodes.
        - Calculates the total distance for each path.
        - Keeps track of the shortest path and its corresponding distance.
    """
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
    
    return list(tsp_best_path), best_distance #type: ignore
