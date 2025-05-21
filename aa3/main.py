import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from collections import deque
import os
from matplotlib.lines import Line2D
import matplotlib
# import sys
# sys.setrecursionlimit(3000) 

matplotlib.use("Agg")  # Use Agg backend for non-interactive mode

# Create a directory for saving visualizations
os.makedirs("graphs", exist_ok=True)


# Implementations of DFS and BFS
def dfs(graph, start, visited=None, path=None, steps=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []
    if steps is None:
        steps = []

    visited.add(start)
    path.append(start)

    # Save current state as a step
    steps.append(
        {
            "visited": visited.copy(),
            "path": path.copy(),
            "current": start,
            "frontier": [n for n in graph[start] if n not in visited],
        }
    )

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, path, steps)

    return path, steps


def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    path = [start]
    steps = [
        {
            "visited": visited.copy(),
            "path": path.copy(),
            "current": start,
            "queue": list(queue),
            "frontier": graph[start],
        }
    ]

    while queue:
        current = queue.popleft()

        frontier = []
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                path.append(neighbor)
                frontier.append(neighbor)

        if frontier:
            # Save current state as a step
            steps.append(
                {
                    "visited": visited.copy(),
                    "path": path.copy(),
                    "current": current,
                    "queue": list(queue),
                    "frontier": frontier,
                }
            )

    return path, steps


# Function to measure algorithm performance
def measure_performance(graph, start_node, algorithm):
    start_time = time.time()
    if algorithm == "DFS":
        path, _ = dfs(graph, start_node)
    else:  # BFS
        path, _ = bfs(graph, start_node)
    end_time = time.time()

    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    memory_usage = len(
        path
    )  # Simple proxy for memory usage - size of the resulting path

    return {
        "execution_time": execution_time,
        "memory_usage": memory_usage,
        "path_length": len(path),
        "path": path,
    }


# Function to create animated visualization of algorithm steps
def create_animated_visualization(G, steps, algorithm, graph_type, directed=False):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get node positions (constant for all steps)
    if graph_type in ["grid"]:
        pos = {node: node for node in G.nodes()}
    else:
        pos = nx.spring_layout(G, seed=42)

    def update(frame_idx):
        ax.clear()
        step = steps[frame_idx]

        visited = step["visited"]
        current = step["current"]
        frontier = step.get("frontier", [])

        # Draw the graph
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=directed, alpha=0.3)

        # Draw nodes with different colors
        all_nodes = list(G.nodes())
        node_colors = []
        node_sizes = []

        for node in all_nodes:
            if node == current:
                node_colors.append("red")  # Current node
                node_sizes.append(700)
            elif node in frontier:
                node_colors.append("orange")  # Frontier nodes
                node_sizes.append(500)
            elif node in visited:
                node_colors.append("green")  # Visited nodes
                node_sizes.append(500)
            else:
                node_colors.append("lightgray")  # Unvisited nodes
                node_sizes.append(500)

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=all_nodes,
            node_color=node_colors,
            node_size=node_sizes,
        )
        nx.draw_networkx_labels(G, pos, ax=ax)

        # Add step information
        queue_str = f"Queue: {step.get('queue', [])}" if algorithm == "BFS" else ""

        # Create legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=15,
                label="Current Node",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="orange",
                markersize=15,
                label="Frontier Nodes",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=15,
                label="Visited Nodes",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lightgray",
                markersize=15,
                label="Unvisited Nodes",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        ax.set_title(
            f"{algorithm} on {graph_type.title()} Graph - Step {frame_idx+1}/{len(steps)}\n{queue_str}"
        )
        ax.axis("off")

    ani = animation.FuncAnimation(
        fig, update, frames=len(steps), interval=3000, repeat=True
    )
    plt.tight_layout()

    # Save animation
    ani.save(
        f"graphs/{algorithm}_{graph_type}_animated.gif",
        writer="pillow",
        fps=1 / 3,
    )
    plt.close()


# Function to compare algorithms for various node sizes
def compare_algorithms_scaled(graph_type, sizes=[10, 20, 50, 100], directed=False):
    results = {
        "DFS": {"execution_time": [], "memory_usage": [], "path_length": []},
        "BFS": {"execution_time": [], "memory_usage": [], "path_length": []},
    }

    for size in sizes:
        # Generate graph based on type and size
        if graph_type == "path":
            G = nx.path_graph(size)
        elif graph_type == "cycle":
            G = nx.cycle_graph(size)
        elif graph_type == "complete":
            G = nx.complete_graph(size)
        elif graph_type == "star":
            G = nx.star_graph(size - 1)  # -1 because star graph adds a central node
        elif graph_type == "bipartite":
            half_size = size // 2
            G = nx.complete_bipartite_graph(half_size, size - half_size)
        elif graph_type == "binary_tree":
            depth = int(np.log2(size + 1))  # Approximate depth for given size
            G = nx.balanced_tree(2, depth)
        elif graph_type == "forest":
            depth = int(np.log2(size // 2 + 1))
            G1 = nx.balanced_tree(2, depth)
            G2 = nx.balanced_tree(2, depth)
            G = nx.disjoint_union(G1, G2)
            # Add an edge between the two trees
            n1 = len(G1)
            G.add_edge(0, n1)
        elif graph_type == "dag":
            G = nx.DiGraph()
            for i in range(size - 1):
                G.add_edge(i, i + 1)
                if i < size - 2:  # Add some branching
                    G.add_edge(i, i + 2)
        elif graph_type == "directed_cycle":
            G = nx.DiGraph()
            for i in range(size - 1):
                G.add_edge(i, i + 1)
            G.add_edge(size - 1, 0)  # Add cycle
            # Add some additional edges
            for i in range(0, size - 2, 2):
                G.add_edge(i, i + 2)
        elif graph_type == "grid":
            grid_size = int(np.sqrt(size))
            G = nx.grid_2d_graph(grid_size, grid_size)
        elif graph_type == "sparse":
            G = nx.gnp_random_graph(size, 0.2, seed=42)
            # Ensure graph is connected
            while not nx.is_connected(G):
                G = nx.gnp_random_graph(size, 0.2, seed=np.random.randint(1000))
        elif graph_type == "dense":
            G = nx.gnp_random_graph(size, 0.7, seed=42)

        # Ensure we have a valid start node
        start_node = list(G.nodes())[0]

        # Convert NetworkX graph to adjacency list representation
        graph_adj_list = {node: list(G.neighbors(node)) for node in G.nodes()}

        # Run DFS and measure performance
        dfs_result = measure_performance(graph_adj_list, start_node, "DFS")
        results["DFS"]["execution_time"].append(dfs_result["execution_time"])
        results["DFS"]["memory_usage"].append(dfs_result["memory_usage"])
        results["DFS"]["path_length"].append(dfs_result["path_length"])

        # Run BFS and measure performance
        bfs_result = measure_performance(graph_adj_list, start_node, "BFS")
        results["BFS"]["execution_time"].append(bfs_result["execution_time"])
        results["BFS"]["memory_usage"].append(bfs_result["memory_usage"])
        results["BFS"]["path_length"].append(bfs_result["path_length"])

        # Create animated visualizations for the smallest size only
        if size == sizes[0]:
            _, dfs_steps = dfs(graph_adj_list, start_node)
            _, bfs_steps = bfs(graph_adj_list, start_node)
            create_animated_visualization(G, dfs_steps, "DFS", graph_type, directed)
            create_animated_visualization(G, bfs_steps, "BFS", graph_type, directed)

    # Plot comparison for all sizes
    # fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # metrics = ["execution_time", "memory_usage", "path_length"]
    # titles = ["Execution Time (ms)", "Memory Usage (nodes)", "Path Length"]

    # for i, (metric, title) in enumerate(zip(metrics, titles)):
    #     axs[i].plot(sizes, results["DFS"][metric], "o-", label="DFS")
    #     axs[i].plot(sizes, results["BFS"][metric], "s-", label="BFS")
    #     axs[i].set_title(f"{title} - {graph_type.title()} Graph")
    #     axs[i].set_xlabel("Number of Nodes")
    #     axs[i].set_ylabel(title)
    #     axs[i].legend()
    #     axs[i].grid(True)

    # plt.tight_layout()
    # plt.savefig(f"graphs/comparison_{graph_type}_scaled.png", dpi=300)
    # plt.close()

    # Also create a bar chart comparison for all sizes
    # fig, axs = plt.subplots(1, len(sizes), figsize=(20, 8))

    # for i, size in enumerate(sizes):
    #     x = np.arange(len(metrics))
    #     width = 0.35

    #     dfs_values = [results["DFS"][metric][i] for metric in metrics]
    #     bfs_values = [results["BFS"][metric][i] for metric in metrics]

    #     rects1 = axs[i].bar(x - width / 2, dfs_values, width, label="DFS")
    #     rects2 = axs[i].bar(x + width / 2, bfs_values, width, label="BFS")

    #     axs[i].set_title(f"{size} Nodes")
    #     axs[i].set_xticks(x)
    #     axs[i].set_xticklabels(["Time (ms)", "Memory", "Path Length"], rotation=45)
    #     axs[i].legend()

    #     # Add values on top of bars
    #     def autolabel(rects):
    #         for rect in rects:
    #             height = rect.get_height()
    #             axs[i].annotate(
    #                 f"{height:.1f}",
    #                 xy=(rect.get_x() + rect.get_width() / 2, height),
    #                 xytext=(0, 3),
    #                 textcoords="offset points",
    #                 ha="center",
    #                 va="bottom",
    #                 fontsize=8,
    #             )

    #     autolabel(rects1)
    #     autolabel(rects2)

    # plt.suptitle(
    #     f"DFS vs BFS on {graph_type.title()} Graph - Performance Comparison",
    #     fontsize=16,
    # )
    # plt.tight_layout()
    # plt.savefig(f"graphs/bar_comparison_{graph_type}_scaled.png", dpi=300)
    # plt.close()

    return results

def plot_all_execution_times(all_results, sizes):
    graph_types = list(all_results.keys())
    n_types = len(graph_types)
    n_cols = 4
    n_rows = (n_types + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs = axs.flatten()

    for idx, graph_type in enumerate(graph_types):
        ax = axs[idx]
        dfs_times = all_results[graph_type]["DFS"]["execution_time"]
        bfs_times = all_results[graph_type]["BFS"]["execution_time"]

        ax.plot(sizes, dfs_times, "o-", label="DFS")
        ax.plot(sizes, bfs_times, "s-", label="BFS")
        ax.set_title(f"{graph_type.replace('_', ' ').title()}")
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Execution Time (ms)")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for idx in range(len(graph_types), len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout()
    plt.suptitle("DFS vs BFS Execution Time Across Graph Types", fontsize=18, y=1.02)
    plt.savefig("graphs/all_execution_times_grid.png", dpi=300, bbox_inches="tight")
    plt.close()

# Generate different types of graphs and compare algorithms
def generate_and_analyze_graphs(sizes):
    graph_types = [
        "path",
        "cycle",
        "complete",
        "star",
        "bipartite",
        "binary_tree",
        "forest",
        "dag",
        "directed_cycle",
        "grid",
        "sparse",
        "dense",
    ]

    directed_types = ["dag", "directed_cycle"]

    results = {}
    for graph_type in graph_types:
        print(f"Analyzing {graph_type} graphs...")
        results[graph_type] = compare_algorithms_scaled(
            graph_type,
            sizes=sizes,
            directed=graph_type in directed_types,
        )

    return results


# Create a summary visualization of all graph types
def create_summary_visualization(results):
    graph_types = list(results.keys())
    sizes = [100, 200, 300, 400]

    # Plot execution time comparison for all graph types at different sizes
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    axs = axs.flatten()

    for i, size in enumerate(sizes):
        dfs_times = [results[g]["DFS"]["execution_time"][i] for g in graph_types]
        bfs_times = [results[g]["BFS"]["execution_time"][i] for g in graph_types]

        x = np.arange(len(graph_types))
        width = 0.35

        rects1 = axs[i].bar(x - width / 2, dfs_times, width, label="DFS")
        rects2 = axs[i].bar(x + width / 2, bfs_times, width, label="BFS")

        axs[i].set_ylabel("Execution Time (ms)")
        axs[i].set_title(f"Comparison at {size} Nodes")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(
            [g.replace("_", " ").title() for g in graph_types], rotation=45
        )
        axs[i].legend()

        # Add values on top of bars for larger differences
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 1:  # Only label if value is significant
                    axs[i].annotate(
                        f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        autolabel(rects1)
        autolabel(rects2)

    plt.suptitle("DFS vs BFS Execution Time Across Different Graph Types", fontsize=16)
    plt.tight_layout()
    plt.savefig("graphs/overall_time_comparison.png", dpi=300)
    plt.close()

    # Plot memory usage comparison
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    axs = axs.flatten()

    for i, size in enumerate(sizes):
        dfs_memory = [results[g]["DFS"]["memory_usage"][i] for g in graph_types]
        bfs_memory = [results[g]["BFS"]["memory_usage"][i] for g in graph_types]

        x = np.arange(len(graph_types))
        width = 0.35

        rects1 = axs[i].bar(x - width / 2, dfs_memory, width, label="DFS")
        rects2 = axs[i].bar(x + width / 2, bfs_memory, width, label="BFS")

        axs[i].set_ylabel("Memory Usage (nodes)")
        axs[i].set_title(f"Comparison at {size} Nodes")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(
            [g.replace("_", " ").title() for g in graph_types], rotation=45
        )
        axs[i].legend()

    plt.suptitle("DFS vs BFS Memory Usage Across Different Graph Types", fontsize=16)
    plt.tight_layout()
    plt.savefig("graphs/overall_memory_comparison.png", dpi=300)
    plt.close()


# Run the analysis
sizes = [i * 10 for i in range(1, 50)]  # Use the same sizes as in generate_and_analyze_graphs
results = generate_and_analyze_graphs(sizes=sizes)
plot_all_execution_times(results, sizes)
create_summary_visualization(results)

print(
    "Analysis complete! All visualizations saved in the 'graphs' directory."
)
