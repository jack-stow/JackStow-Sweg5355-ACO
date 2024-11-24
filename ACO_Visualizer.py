import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
import networkx as nx  # For creating and visualizing graphs
import matplotlib.cm as cm  # For colormaps in visualizations
from matplotlib.colors import Normalize  # For normalizing color values
from matplotlib.animation import FuncAnimation  # For creating animations


# Seed for consistent graph layout visualization
NETWORK_SEED = 42

# Visualization Function
def visualize_aco(aco, path_index=0):
    """
    Visualizes a specific iteration of paths discovered by the Ant Colony Optimization algorithm.

    Parameters:
        aco (AntColony): The Ant Colony instance containing the distance matrix and pheromone matrix.
        path_index (int): The iteration index to visualize the paths from.

    Functionality:
        - Creates a graph visualization with nodes and edges.
        - Uses a color gradient to represent pheromone strength on edges.
        - Highlights the paths explored by ants at a specific iteration.
    """
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
                color = cm.plasma(norm_strength) #type: ignore

                ax.plot([pos[start_node][0], pos[end_node][0]], 
                        [pos[start_node][1], pos[end_node][1]], 
                        color=color, linewidth=1, alpha=0.7, zorder=0)
    
    ax.set_title(f'Ant Colony Optimization - Path at Iteration {path_index}')
    plt.axis('off')
    plt.show()



# Visualization Function with Color Gradient
def animate_aco_paths(aco, interval):
    """
    Animates the paths discovered by the Ant Colony Optimization algorithm over iterations.

    Parameters:
        aco (AntColony): The Ant Colony instance containing the paths history and distance matrix.
        interval (int): Time interval between frames in milliseconds.

    Functionality:
        - Creates an animated visualization of the paths discovered over iterations.
        - Highlights the evolution of paths and pheromone strength using color gradients.
        - Updates the graph frame by frame with edges being drawn based on the paths history.
    """
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

    ani = FuncAnimation(fig, update, frames=len(aco.paths_history), repeat=False, interval=interval) #type: ignore
    plt.show()
    plt.show()


# Visualization Function for a Static Path with Color Gradient
def visualize_solution_path(distance_matrix, solution_path):
    """
    Visualizes a static Traveling Salesperson Problem solution path.

    Parameters:
        distance_matrix (numpy.ndarray): A matrix where the entry (i, j) represents the distance between node i and node j.
        solution_path (list): A list of node indices representing the solution path.

    Functionality:
        - Draws a graph with nodes and edges.
        - Uses a color gradient to differentiate edges in the solution path.
        - Connects the last node back to the first to form a complete cycle.
    """
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




def animate_best_path(aco, interval):
    """
    Animates the best path discovered by the Ant Colony Optimization algorithm.

    Parameters:
        aco (AntColony): The Ant Colony instance containing the best path and distance matrix.
        interval (int): Time interval between frames in milliseconds.

    Functionality:
        - Creates an animation of the best path being constructed step by step.
        - Highlights each edge in the path sequentially with a color gradient.
        - Updates the graph frame by frame with edges being drawn up to the current step.
    """
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

    ani = FuncAnimation(fig, update, frames=num_edges, repeat=False, interval=interval) #type: ignore
    plt.show()

def animate_pheromone_history(aco, interval=1, step=1):
    """
    Visualizes the pheromone history of the Ant Colony Optimization algorithm as an animation.

    Parameters:
        aco (AntColony): The Ant Colony instance containing the pheromone history and distance matrix.
        interval (int): The number of iterations to skip between frames in the animation.
    """
    # Create a NetworkX graph from the distance matrix (same as before)
    G = nx.from_numpy_array(aco.distance_matrix)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=NETWORK_SEED)  # Positions for the nodes
    
    # Normalize the pheromone strength for coloring
    max_pheromone_value = np.max(aco.pheromone)
    norm = Normalize(vmin=0, vmax=max_pheromone_value)
    
    # Create a colormap (using a rainbow colormap)
    cmap = cm.rainbow #type: ignore
    
    def update(frame):
        ax.clear()  # Clear the previous plot
        
        # Get the pheromone matrix for the current iteration (frame)
        pheromone_matrix = aco.pheromone_history[frame]
        
        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)

        # Draw the edges with color based on pheromone strength
        for u, v in G.edges():
            pheromone_strength = pheromone_matrix[u, v]
            pheromone_normalized = norm(pheromone_strength)
            color = cmap(pheromone_normalized)  # Map pheromone strength to color
            if pheromone_normalized > 0.01:
                ax.plot([pos[u][0], pos[v][0]], 
                        [pos[u][1], pos[v][1]], 
                        color=color, linewidth=5*pheromone_normalized, alpha=0.8)
        
        # Set title for the current iteration
        ax.set_title(f'Pheromone Evolution - Iteration {frame}')
        ax.axis('off')  # Hide axes

    # Generate the frames, skipping iterations based on the step argument
    frames = list(range(0, len(aco.pheromone_history), step))
    if frames[-1] != len(aco.pheromone_history) - 1:
        frames.append(len(aco.pheromone_history) - 1)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False) #type: ignore
    
    # Show the animation
    plt.show()

def animate_paths_history(aco, step=1, interval=100):
    """
    Animates the history of paths discovered by the Ant Colony Optimization algorithm.

    Parameters:
        aco (AntColony): The Ant Colony instance containing paths and distance matrix.
        interval (int): Time interval between frames in milliseconds.
    """
    G = nx.from_numpy_array(aco.distance_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout
    
    # Use rainbow colormap from red (first edge) to violet (last edge)
    cmap = plt.get_cmap("rainbow")

    def update(frame):
        ax.clear()
        plt.axis('off')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)

        # Select current path
        path = aco.paths_history[frame]

        # Draw path edges with color gradient
        for i in range(len(path)):
            start_node = path[i]
            end_node = path[(i + 1) % len(path)]  # Wrap around to form a cycle

            # Ensure we're working with scalar node indices
            start_node = start_node[0] if isinstance(start_node, np.ndarray) else start_node
            end_node = end_node[0] if isinstance(end_node, np.ndarray) else end_node

            # Color edges across the entire path spectrum
            edge_color = cmap(i / (len(path) - 1))

            # Draw edge
            ax.plot(
                [pos[start_node][0], pos[end_node][0]],
                [pos[start_node][1], pos[end_node][1]],
                color=edge_color, linewidth=3, zorder=0
            )

        # Add informative title
        path_distance = aco.calculate_distance(path)
        ax.set_title(f'Path {frame+1}/{len(aco.paths_history)} - Distance: {path_distance:.2f}')

    frames = list(range(0, len(aco.paths_history), step))
    if frames[-1] != len(aco.paths_history) - 1:
        frames.append(len(aco.paths_history) - 1)
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False) #type: ignore
    plt.show()