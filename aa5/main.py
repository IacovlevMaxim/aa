"""
Minimum Spanning Tree Algorithms: Prim's and Kruskal's with Visualization
"""

import time
import heapq
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation


class DisjointSet:
    """Implementation of Disjoint Set data structure for Kruskal's algorithm"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Find the representative of the set containing x with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union of two sets using rank heuristic"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True


def generate_random_graph(n, density=0.3, graph_type="sparse"):
    """
    Generate a random weighted graph with n nodes and given edge density or type.
    Supports various graph types.
    """
    # Borrowed logic from GraphAnalyzer.generate_graph
    import networkx as nx
    import numpy as np
    import random

    if graph_type == "path":
        G = nx.path_graph(n)
    elif graph_type == "cycle":
        G = nx.cycle_graph(n)
    elif graph_type == "complete":
        G = nx.complete_graph(n)
    elif graph_type == "star":
        G = nx.star_graph(n - 1)
    elif graph_type == "bipartite":
        half = n // 2
        G = nx.complete_bipartite_graph(half, n - half)
    elif graph_type == "binary_tree":
        depth = int(np.log2(n + 1))
        G = nx.balanced_tree(2, depth)
        if len(G.nodes) > n:
            G = G.subgraph(list(G.nodes)[:n]).copy()
    elif graph_type == "forest":
        depth = int(np.log2(n // 2 + 1))
        G1 = nx.balanced_tree(2, depth)
        G2 = nx.balanced_tree(2, depth)
        G = nx.disjoint_union(G1, G2)
        n1 = len(G1)
        if n1 < len(G.nodes):
            G.add_edge(0, n1)
        if len(G.nodes) > n:
            G = G.subgraph(list(G.nodes)[:n]).copy()
    elif graph_type == "dag":
        G = nx.DiGraph()
        for i in range(n - 1):
            G.add_edge(i, i + 1)
            if i < n - 2:
                G.add_edge(i, i + 2)
    elif graph_type == "directed_cycle":
        G = nx.DiGraph()
        for i in range(n - 1):
            G.add_edge(i, i + 1)
        G.add_edge(n - 1, 0)
        for i in range(0, n - 2, 2):
            G.add_edge(i, i + 2)
    elif graph_type == "grid":
        grid_size = int(np.sqrt(n))
        G = nx.grid_2d_graph(grid_size, grid_size)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    elif graph_type == "sparse":
        G = nx.gnp_random_graph(n, 0.1 if density is None else density, seed=42)
        while not nx.is_connected(G):
            G = nx.gnp_random_graph(n, 0.1 if density is None else density, seed=random.randint(0, 10000))
    elif graph_type == "dense":
        G = nx.gnp_random_graph(n, 0.7 if density is None else density, seed=42)
    else:
        G = nx.gnp_random_graph(n, 0.1 if density is None else density, seed=42)
        while not nx.is_connected(G):
            G = nx.gnp_random_graph(n, 0.1 if density is None else density, seed=random.randint(0, 10000))

    # Assign random positions for visualization
    positions = {i: (random.random(), random.random()) for i in range(len(G.nodes))}
    for i in G.nodes:
        G.nodes[i]["pos"] = positions[i]

    # Assign random weights to edges
    for u, v in G.edges():
        G[u][v]["weight"] = random.randint(1, 20)

    return G, positions


def kruskal_mst(G):
    """
    Implementation of Kruskal's algorithm to find minimum spanning tree
    Returns the MST and intermediate steps for visualization
    """
    n = G.number_of_nodes()
    edges = [(G[u][v]["weight"], u, v) for u, v in G.edges()]
    edges.sort()  # Sort edges by weight

    disjoint_set = DisjointSet(n)
    mst = []
    mst_weight = 0

    # For visualization, track steps
    steps = []

    for weight, u, v in edges:
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            mst.append((u, v, weight))
            mst_weight += weight

            # Record this step for visualization
            current_mst = mst.copy()
            steps.append((current_mst, (u, v)))

    return mst, mst_weight, steps


def prim_mst(G):
    """
    Implementation of Prim's algorithm to find minimum spanning tree
    Returns the MST and intermediate steps for visualization
    """
    if not G.nodes():
        return [], 0, []

    start_node = list(G.nodes())[0]
    mst = []
    mst_weight = 0
    visited = {start_node}
    edges = [
        (G[start_node][v]["weight"], start_node, v) for v in G.neighbors(start_node)
    ]
    heapq.heapify(edges)

    # For visualization, track steps
    steps = []

    while edges and len(visited) < G.number_of_nodes():
        weight, u, v = heapq.heappop(edges)

        if v not in visited:
            visited.add(v)
            mst.append((u, v, weight))
            mst_weight += weight

            # Record this step for visualization
            current_mst = mst.copy()
            steps.append((current_mst, (u, v)))

            # Add new edges to the priority queue
            for neighbor in G.neighbors(v):
                if neighbor not in visited:
                    heapq.heappush(edges, (G[v][neighbor]["weight"], v, neighbor))

    return mst, mst_weight, steps


def visualize_algorithm_steps(G, pos, steps, algorithm_name):
    """
    Visualize the steps of MST algorithm execution
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title(f"{algorithm_name} Algorithm Visualization")

    # Draw the original graph
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}

    def update(frame_idx):
        ax.clear()
        plt.title(f"{algorithm_name} Algorithm - Step {frame_idx+1}/{len(steps)}")

        # Draw all edges of the original graph (faded)
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue", ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)

        # Draw MST edges built so far
        mst_edges_so_far = [(u, v) for u, v, _ in steps[frame_idx][0]]
        if mst_edges_so_far:
            nx.draw_networkx_edges(
                G, pos, edgelist=mst_edges_so_far, width=2.0, edge_color="blue", ax=ax
            )

        # Highlight the edge just added in this step
        latest_edge = steps[frame_idx][1]
        if latest_edge:
            nx.draw_networkx_edges(
                G, pos, edgelist=[latest_edge], width=3.0, edge_color="red", ax=ax
            )

        # Add edge labels
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=8, ax=ax
        )

    ani = FuncAnimation(fig, update, frames=len(steps), interval=1000, repeat=False)
    plt.tight_layout()
    plt.show()


def visualize_final_mst(G, pos, mst, algorithm_name):
    """
    Visualize the final minimum spanning tree
    """
    plt.figure(figsize=(10, 8))
    plt.title(f"Final Minimum Spanning Tree - {algorithm_name}")

    # Draw all edges of the original graph (faded)
    nx.draw_networkx_edges(G, pos, alpha=0.2)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue")
    nx.draw_networkx_labels(G, pos)

    # Draw MST edges
    mst_edges = [(u, v) for u, v, _ in mst]
    nx.draw_networkx_edges(G, pos, edgelist=mst_edges, width=2.0, edge_color="red")

    # Add edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.tight_layout()
    plt.show()


def compare_algorithms(node_sizes, trials=3, density=0.3):
    """
    Compare the performance of Kruskal's and Prim's algorithms
    with increasing graph sizes and various graph types.
    """
    graph_types = [
        "path", "cycle", "complete", "star", "bipartite", "binary_tree",
        "forest", "grid", "sparse", "dense"
    ]
    results = {}

    for graph_type in graph_types:
        kruskal_times = []
        prim_times = []
        print(f"\nTesting graph type: {graph_type}")
        for n in node_sizes:
            k_time = 0
            p_time = 0
            for _ in range(trials):
                G, pos = generate_random_graph(n, density, graph_type=graph_type)
                # Time Kruskal's algorithm
                start = time.time()
                kruskal_mst(G)
                k_time += time.time() - start
                # Time Prim's algorithm
                start = time.time()
                prim_mst(G)
                p_time += time.time() - start
            kruskal_times.append(k_time / trials)
            prim_times.append(p_time / trials)
            print(f"Nodes: {n}, Kruskal: {k_time/trials:.6f}s, Prim: {p_time/trials:.6f}s")
        results[graph_type] = {
            "node_sizes": node_sizes,
            "kruskal_times": kruskal_times,
            "prim_times": prim_times,
        }
    return results


def plot_performance(results):
    """
    Plot performance comparison between Kruskal's and Prim's algorithms for all graph types.
    """
    import matplotlib.pyplot as plt

    num_types = len(results)
    cols = 3
    rows = (num_types + cols - 1) // cols
    plt.figure(figsize=(6 * cols, 4 * rows))

    for idx, (graph_type, data) in enumerate(results.items()):
        plt.subplot(rows, cols, idx + 1)
        plt.plot(data["node_sizes"], data["kruskal_times"], "o-", label="Kruskal's")
        plt.plot(data["node_sizes"], data["prim_times"], "s-", label="Prim's")
        plt.xlabel("Number of Nodes")
        plt.ylabel("Execution Time (s)")
        plt.title(graph_type.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main function to demonstrate the algorithms
def main():
    # Generate a random graph for demonstration
    n = 22  # number of nodes
    G, pos = generate_random_graph(n)

    print(f"Generated graph with {n} nodes and {G.number_of_edges()} edges")

    # Run Kruskal's algorithm
    kruskal_result, kruskal_weight, kruskal_steps = kruskal_mst(G)
    print(f"Kruskal's MST weight: {kruskal_weight:.2f}")

    # Run Prim's algorithm
    prim_result, prim_weight, prim_steps = prim_mst(G)
    print(f"Prim's MST weight: {prim_weight:.2f}")

    # Visualize algorithm steps
    visualize_algorithm_steps(G, pos, kruskal_steps, "Kruskal's")
    visualize_algorithm_steps(G, pos, prim_steps, "Prim's")

    # # Visualize final MSTs
    visualize_final_mst(G, pos, kruskal_result, "Kruskal's")
    visualize_final_mst(G, pos, prim_result, "Prim's")

    # Performance comparison with different graph sizes
    node_sizes = [i * 10 for i in range(1, 30)]
    results = compare_algorithms(node_sizes)
    plot_performance(results)


if __name__ == "__main__":
    main()
