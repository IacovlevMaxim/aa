import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import networkx as nx

class GraphAnalyzer:
    def __init__(self):
        self.animation_frames = []

    def generate_graph(self, num_nodes, density=None, graph_type="sparse"):
        """
        Generate a random weighted graph
        - num_nodes: number of nodes in the graph
        - density: probability of edge between any two nodes (0 to 1)
        """
        # Initialize adjacency matrix with infinity
        if graph_type == "path":
            G = nx.path_graph(num_nodes)
        elif graph_type == "cycle":
            G = nx.cycle_graph(num_nodes)
        elif graph_type == "complete":
            G = nx.complete_graph(num_nodes)
        elif graph_type == "star":
            G = nx.star_graph(num_nodes - 1)
        elif graph_type == "bipartite":
            half = num_nodes // 2
            G = nx.complete_bipartite_graph(half, num_nodes - half)
        elif graph_type == "binary_tree":
            depth = int(np.log2(num_nodes + 1))
            G = nx.balanced_tree(2, depth)
            # Trim to num_nodes if needed
            if len(G.nodes) > num_nodes:
                G = G.subgraph(list(G.nodes)[:num_nodes]).copy()
        elif graph_type == "forest":
            depth = int(np.log2(num_nodes // 2 + 1))
            G1 = nx.balanced_tree(2, depth)
            G2 = nx.balanced_tree(2, depth)
            G = nx.disjoint_union(G1, G2)
            n1 = len(G1)
            if n1 < len(G.nodes):
                G.add_edge(0, n1)
            # Trim to num_nodes if needed
            if len(G.nodes) > num_nodes:
                G = G.subgraph(list(G.nodes)[:num_nodes]).copy()
        elif graph_type == "dag":
            G = nx.DiGraph()
            for i in range(num_nodes - 1):
                G.add_edge(i, i + 1)
                if i < num_nodes - 2:
                    G.add_edge(i, i + 2)
        elif graph_type == "directed_cycle":
            G = nx.DiGraph()
            for i in range(num_nodes - 1):
                G.add_edge(i, i + 1)
            G.add_edge(num_nodes - 1, 0)
            for i in range(0, num_nodes - 2, 2):
                G.add_edge(i, i + 2)
        elif graph_type == "grid":
            grid_size = int(np.sqrt(num_nodes))
            G = nx.grid_2d_graph(grid_size, grid_size)
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
        elif graph_type == "sparse":
            G = nx.gnp_random_graph(num_nodes, 0.1 if density is None else density, seed=42)
            while not nx.is_connected(G):
                G = nx.gnp_random_graph(num_nodes, 0.1 if density is None else density, seed=random.randint(0, 10000))
        elif graph_type == "dense":
            G = nx.gnp_random_graph(num_nodes, 0.7 if density is None else density, seed=42)
        else:
            # Default: random sparse
            G = nx.gnp_random_graph(num_nodes, 0.1 if density is None else density, seed=42)
            while not nx.is_connected(G):
                G = nx.gnp_random_graph(num_nodes, 0.1 if density is None else density, seed=random.randint(0, 10000))

        # Assign random weights to edges
        for u, v in G.edges():
            G[u][v]["weight"] = random.randint(1, 20)

        # Convert to adjacency matrix
        graph = np.full((num_nodes, num_nodes), float("inf"))
        np.fill_diagonal(graph, 0)
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1)
            graph[u][v] = w
            if not G.is_directed():
                graph[v][u] = w

        return graph
    
    def dijkstra(self, graph, start_node, animate=False):
        """
        Dijkstra's algorithm for finding shortest paths from start_node to all other nodes
        With step-by-step visualization if animate is True
        """
        num_nodes = len(graph)
        distances = [float("inf")] * num_nodes
        distances[start_node] = 0
        visited = [False] * num_nodes
        previous = [None] * num_nodes

        if animate:
            self.animation_frames = []
            G = nx.Graph()

            # Add nodes and edges to networkx graph
            for i in range(num_nodes):
                G.add_node(i)

            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if graph[i][j] != float("inf"):
                        G.add_edge(i, j, weight=graph[i][j])

            pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout

        # Dijkstra algorithm
        for _ in range(num_nodes):
            # Find the unvisited node with minimum distance
            min_distance = float("inf")
            min_node = -1
            for i in range(num_nodes):
                if not visited[i] and distances[i] < min_distance:
                    min_distance = distances[i]
                    min_node = i

            if min_node == -1:
                break

            visited[min_node] = True

            # Update distances to neighbors
            for neighbor in range(num_nodes):
                if (
                    not visited[neighbor]
                    and graph[min_node][neighbor] != float("inf")
                    and distances[min_node] + graph[min_node][neighbor]
                    < distances[neighbor]
                ):
                    distances[neighbor] = (
                        distances[min_node] + graph[min_node][neighbor]
                    )
                    previous[neighbor] = min_node

            if animate:
                # Create a frame for animation
                frame = {
                    "G": G.copy(),
                    "pos": pos,
                    "visited": visited.copy(),
                    "current": min_node,
                    "distances": distances.copy(),
                    "previous": previous.copy(),
                }
                self.animation_frames.append(frame)

        return distances, previous


    def dijkstra_all_pairs(self, graph, animate=False):
        """
        Run Dijkstra's algorithm from every node (all-pairs shortest paths).
        Returns a matrix of shortest distances between all pairs of nodes.
        If animate=True, creates frames for the first start node only.
        """
        num_nodes = len(graph)
        all_distances = np.full((num_nodes, num_nodes), float("inf"))
        all_previous = [None] * num_nodes

        if animate:
            self.animation_frames = []
            G = nx.Graph()
            for i in range(num_nodes):
                G.add_node(i)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if graph[i][j] != float("inf"):
                        G.add_edge(i, j, weight=graph[i][j])
            pos = nx.spring_layout(G, seed=42)

        for start_node in range(num_nodes):
            distances = [float("inf")] * num_nodes
            distances[start_node] = 0
            visited = [False] * num_nodes
            previous = [None] * num_nodes

            for _ in range(num_nodes):
                min_distance = float("inf")
                min_node = -1
                for i in range(num_nodes):
                    if not visited[i] and distances[i] < min_distance:
                        min_distance = distances[i]
                        min_node = i
                if min_node == -1:
                    break
                visited[min_node] = True
                for neighbor in range(num_nodes):
                    if (
                        not visited[neighbor]
                        and graph[min_node][neighbor] != float("inf")
                        and distances[min_node] + graph[min_node][neighbor] < distances[neighbor]
                    ):
                        distances[neighbor] = distances[min_node] + graph[min_node][neighbor]
                        previous[neighbor] = min_node

                # Only animate for the first start_node (to avoid huge memory usage)
                if animate and start_node == 0:
                    frame = {
                        "G": G.copy(),
                        "pos": pos,
                        "visited": visited.copy(),
                        "current": min_node,
                        "distances": distances.copy(),
                        "previous": previous.copy(),
                    }
                    self.animation_frames.append(frame)

            all_distances[start_node] = distances
            all_previous[start_node] = previous

        return all_distances, all_previous

    def floyd_warshall(self, graph, animate=False):
        """
        Floyd-Warshall algorithm for finding all-pairs shortest paths
        With step-by-step visualization if animate is True
        """
        num_nodes = len(graph)
        # Make a copy of the graph to avoid modifying the original
        dist = np.copy(graph)
        next_node = np.full((num_nodes, num_nodes), None)

        # Initialize next_node matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and dist[i][j] != float("inf"):
                    next_node[i][j] = j

        if animate:
            self.animation_frames = []
            G = nx.Graph()

            # Add nodes and edges to networkx graph
            for i in range(num_nodes):
                G.add_node(i)

            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if graph[i][j] != float("inf"):
                        G.add_edge(i, j, weight=graph[i][j])

            pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout

        # Floyd-Warshall algorithm
        for k in range(num_nodes):
            # Broadcasting: dist[:, k, np.newaxis] + dist[k, :]
            new_dist = dist[:, k, np.newaxis] + dist[k, :]
            mask = new_dist < dist
            dist = np.where(mask, new_dist, dist)
            # Update next_node where path is improved
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if mask[i, j]:
                        next_node[i, j] = next_node[i, k]

            if animate and k < num_nodes - 1:  # Don't add the last frame twice
                # Create a frame for animation
                frame = {
                    "G": G.copy(),
                    "pos": pos,
                    "k": k,
                    "dist": np.copy(dist),
                    "next_node": next_node.copy(),
                }
                self.animation_frames.append(frame)

        # Add final frame
        if animate:
            frame = {
                "G": G.copy(),
                "pos": pos,
                "k": num_nodes - 1,
                "dist": np.copy(dist),
                "next_node": next_node.copy(),
            }
            self.animation_frames.append(frame)

        return dist, next_node

    def reconstruct_path(self, previous, start, end):
        """
        Reconstruct path from start to end using previous array from Dijkstra
        """
        if previous[end] is None and start != end:
            return []

        path = [end]
        while path[0] != start:
            path.insert(0, previous[path[0]])

        return path

    def reconstruct_floyd_path(self, next_node, start, end):
        """
        Reconstruct path from start to end using next_node matrix from Floyd-Warshall
        """
        if next_node[start][end] is None:
            return []

        path = [start]
        while path[-1] != end:
            path.append(next_node[path[-1]][end])

        return path

    def animate_dijkstra(self, start_node, end_node=None):
        """
        Create animation of Dijkstra's algorithm
        """
        if not self.animation_frames:
            print("No animation frames found. Run dijkstra with animate=True first.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        def update(frame_idx):
            ax.clear()
            frame = self.animation_frames[frame_idx]
            G = frame["G"]
            pos = frame["pos"]
            visited = frame["visited"]
            current = frame["current"]
            distances = frame["distances"]
            previous = frame["previous"]

            # Draw the graph
            # Draw edges
            for u, v, data in G.edges(data=True):
                ax.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    "k-",
                    alpha=0.3,
                    linewidth=data["weight"] / 10,
                )

            # Draw nodes
            for node in G.nodes():
                if node == current:
                    color = "red"  # Current node
                elif visited[node]:
                    color = "green"  # Visited nodes
                else:
                    color = "skyblue"  # Unvisited nodes

                ax.plot(pos[node][0], pos[node][1], "o", markersize=15, color=color)
                ax.annotate(
                    f"{node}\n({distances[node] if distances[node] != float('inf') else '∞'})",
                    xy=pos[node],
                    ha="center",
                    va="center",
                    fontsize=9,
                )

            # Draw the current shortest paths
            for node in range(len(visited)):
                if previous[node] is not None:
                    ax.plot(
                        [pos[previous[node]][0], pos[node][0]],
                        [pos[previous[node]][1], pos[node][1]],
                        "b-",
                        alpha=0.5,
                        linewidth=2,
                    )

            # Draw the final path if specified
            if end_node is not None and frame_idx == len(self.animation_frames) - 1:
                path = self.reconstruct_path(previous, start_node, end_node)
                for i in range(len(path) - 1):
                    ax.plot(
                        [pos[path[i]][0], pos[path[i + 1]][0]],
                        [pos[path[i]][1], pos[path[i + 1]][1]],
                        "r-",
                        linewidth=3,
                    )

            # Add legend
            red_patch = mpatches.Patch(color="red", label="Current Node")
            green_patch = mpatches.Patch(color="green", label="Visited Nodes")
            blue_patch = mpatches.Patch(color="skyblue", label="Unvisited Nodes")
            blue_line = mpatches.Patch(
                color="blue", label="Current Shortest Paths", alpha=0.5
            )

            legend_items = [red_patch, green_patch, blue_patch, blue_line]

            if end_node is not None and frame_idx == len(self.animation_frames) - 1:
                red_line = mpatches.Patch(color="red", label="Final Path")
                legend_items.append(red_line)

            ax.legend(handles=legend_items, loc="upper right")

            ax.set_title(
                f"Dijkstra's Algorithm - Step {frame_idx+1}/{len(self.animation_frames)}"
            )
            ax.axis("off")

        ani = FuncAnimation(
            fig, update, frames=len(self.animation_frames), interval=1000, repeat=False
        )
        plt.tight_layout()
        plt.show()

        return ani

    def animate_floyd_warshall(self, start_node=None, end_node=None):
        """
        Create animation of Floyd-Warshall algorithm
        """
        if not self.animation_frames:
            print(
                "No animation frames found. Run floyd_warshall with animate=True first."
            )
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        def update(frame_idx):
            ax.clear()
            frame = self.animation_frames[frame_idx]
            G = frame["G"]
            pos = frame["pos"]
            k = frame["k"]
            dist = frame["dist"]
            next_node = frame["next_node"]

            # Draw the graph
            # Draw edges
            for u, v, data in G.edges(data=True):
                ax.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    "k-",
                    alpha=0.3,
                    linewidth=data["weight"] / 10,
                )

            # Draw nodes
            for node in G.nodes():
                if node == k:
                    color = "red"  # Current intermediate node k
                else:
                    color = "skyblue"

                ax.plot(pos[node][0], pos[node][1], "o", markersize=15, color=color)
                ax.annotate(
                    f"{node}", xy=pos[node], ha="center", va="center", fontsize=9
                )

            # Draw the final path if specified
            if (
                start_node is not None
                and end_node is not None
                and frame_idx == len(self.animation_frames) - 1
            ):
                path = self.reconstruct_floyd_path(next_node, start_node, end_node)
                for i in range(len(path) - 1):
                    ax.plot(
                        [pos[path[i]][0], pos[path[i + 1]][0]],
                        [pos[path[i]][1], pos[path[i + 1]][1]],
                        "r-",
                        linewidth=3,
                    )

            # Add legend
            red_patch = mpatches.Patch(
                color="red", label=f"Current Intermediate Node k={k}"
            )
            blue_patch = mpatches.Patch(color="skyblue", label="Other Nodes")

            legend_items = [red_patch, blue_patch]

            if (
                start_node is not None
                and end_node is not None
                and frame_idx == len(self.animation_frames) - 1
            ):
                red_line = mpatches.Patch(color="red", label="Final Path")
                legend_items.append(red_line)

            ax.legend(handles=legend_items, loc="upper right")

            ax.set_title(
                f"Floyd-Warshall Algorithm - Step {frame_idx+1}/{len(self.animation_frames)}"
            )
            ax.axis("off")

        ani = FuncAnimation(
            fig, update, frames=len(self.animation_frames), interval=1000, repeat=False
        )
        plt.tight_layout()
        plt.show()

        return ani

    def empirical_analysis(self, start_sizes, end_sizes, step, repetitions=5):
        """
        Perform empirical analysis of both algorithms on sparse and dense graphs
        """
        """
        Perform empirical analysis of both algorithms on various types of graphs.
        """
        sizes = range(start_sizes, end_sizes + 1, step)
        graph_types = [
            "path", "cycle", "complete", "star", "bipartite", "binary_tree",
            "forest", "dag", "directed_cycle", "grid", "sparse", "dense"
        ]

        results = {}

        for graph_type in graph_types:
            dijkstra_times = []
            floyd_times = []

            for size in sizes:
                dij_time = 0
                fw_time = 0

                for _ in range(repetitions):
                    graph = self.generate_graph(size, graph_type=graph_type)

                    # Dijkstra (from node 0)
                    start_time = time.time()
                    self.dijkstra(graph, start_node=start_node)
                    # self.dijkstra_all_pairs(graph)
                    dij_time += time.time() - start_time

                    # Floyd-Warshall
                    start_time = time.time()
                    self.floyd_warshall(graph)
                    fw_time += time.time() - start_time

                dijkstra_times.append(dij_time / repetitions)
                floyd_times.append(fw_time / repetitions)

                print(f"Completed {graph_type} graph with {size} nodes")

            results[graph_type] = {
                "sizes": list(sizes),
                "dijkstra": dijkstra_times,
                "floyd_warshall": floyd_times,
            }

        # Plotting
        plt.figure(figsize=(16, 12))
        for i, graph_type in enumerate(graph_types):
            plt.subplot(4, 3, i + 1)
            plt.plot(results[graph_type]["sizes"], results[graph_type]["dijkstra"], "b-o", label="Dijkstra")
            plt.plot(results[graph_type]["sizes"], results[graph_type]["floyd_warshall"], "r-s", label="Floyd-Warshall")
            plt.title(graph_type.replace("_", " ").title())
            plt.xlabel("Number of Nodes")
            plt.ylabel("Avg Time (s)")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        return results


# Example usage
if __name__ == "__main__":
    analyzer = GraphAnalyzer()

    # Generate a small graph for visualization
    print("Generating a small graph for visualization...")
    num_nodes = 22
    graph = analyzer.generate_graph(num_nodes, 0.3)

    print("\nDijkstra's Algorithm visualization:")
    start_node = 0
    end_node = 21
    # Run Dijkstra with animation
    distances, previous = analyzer.dijkstra_all_pairs(graph, animate=True)

    # Print shortest path
    # path = analyzer.reconstruct_path(previous, start_node, end_node)
    # print(f"Shortest path from {start_node} to {end_node}: {path}")
    # print(f"Distance: {distances[end_node]}")

    # Animate
    analyzer.animate_dijkstra(start_node, end_node)

    print("\nFloyd-Warshall Algorithm visualization:")
    # Run Floyd-Warshall with animation
    dist, next_node = analyzer.floyd_warshall(graph, animate=True)

    # Print all-pairs shortest paths
    print(
        f"Shortest path from {start_node} to {end_node} (Floyd-Warshall): {analyzer.reconstruct_floyd_path(next_node, start_node, end_node)}"
    )
    print(f"Distance: {dist[start_node][end_node]}")

    # Animate
    analyzer.animate_floyd_warshall(start_node, end_node)

    # Empirical analysis
    print("\nPerforming empirical analysis...")
    results = analyzer.empirical_analysis(start_sizes=10, end_sizes=100, step=10)

    # Print analysis results
    print("\nAnalysis Results:")
    print("Graph Sizes:", list(results["sizes"]))
    print("Dijkstra - Sparse:", [round(t, 5) for t in results["dijkstra_sparse"]])
    print("Dijkstra - Dense:", [round(t, 5) for t in results["dijkstra_dense"]])
    print(
        "Floyd-Warshall - Sparse:",
        [round(t, 5) for t in results["floyd_warshall_sparse"]],
    )
    print(
        "Floyd-Warshall - Dense:",
        [round(t, 5) for t in results["floyd_warshall_dense"]],
    )
