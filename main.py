import math
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue, LifoQueue, PriorityQueue
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.colors import Normalize
from matplotlib import cm
import networkx as nx


# NOMBRES GRAFOS:

# 8 ELECTRODOS:
# graph8_3
# graph8_4
# graph8_5
# graph8_6

# 32 ELECTRODOS:
# graph32A
# graph32B

class DisjointSet:
    def __init__(self, vertices):
        self.parent = {vertex: vertex for vertex in vertices}

    def find(self, vertex):
        if self.parent[vertex] == vertex:
            return vertex
        self.parent[vertex] = self.find(self.parent[vertex])  # Path compression
        return self.parent[vertex]

    def union(self, vertex1, vertex2):
        root1 = self.find(vertex1)
        root2 = self.find(vertex2)
        if root1 != root2:
            self.parent[root1] = root2

    def are_connected(self, vertex1, vertex2):
        return self.find(vertex1) == self.find(vertex2)


class InitialGraph:
    def __init__(self):
        pass

    def distance3D(punto1, punto2):
        x1, y1, z1 = punto1
        x2, y2, z2 = punto2

        distancia = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distancia

    def calculate_angle(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def plot_voronoi(points2D, adjacency_matrix):
        vor = Voronoi(points2D)
        num_edges = np.sum(adjacency_matrix, axis=1)

        # Colormap based on the number of edges
        norm = Normalize(vmin=min(num_edges), vmax=max(num_edges))
        cmap = cm.ScalarMappable(norm=norm, cmap=cm.plasma)

        fig, ax = plt.subplots()

        voronoi_plot_2d(vor, show_vertices=False, line_colors='k', line_alpha=0.5, ax=ax)

        for region, color in zip(vor.regions, num_edges):
            if -1 not in region and len(region) > 2:
                polygon = [vor.vertices[i] for i in region]
                poly = plt.Polygon(polygon, facecolor=cmap.to_rgba(color), edgecolor='k', alpha=0.7)
                ax.add_patch(poly)

        # Circumcircles for the corresponding region
        for i, color in enumerate(num_edges):
            plt.scatter(vor.vertices[i, 0], vor.vertices[i, 1], c=[cmap.to_rgba(color)], edgecolors='k', s=50,
                        zorder=10)

        plt.colorbar(cmap, label='Number of Edges')
        plt.xlim(min(points2D[:, 0]) - 1, max(points2D[:, 0]) + 1)
        plt.ylim(min(points2D[:, 1]) - 1, max(points2D[:, 1]) + 1)
        plt.title('Voronoi Diagram with Edge Count Colors')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

    def plot8(adjacency_matrix):
        channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

        points3D = np.array([[0, 0.71934, 0.694658], [-0.71934, 0, 0.694658], [0, 0, 1], [0.71934, 0, 0.694658],
                             [0, -0.71934, 0.694658], [-0.587427, -0.808524, - 0.0348995], [0, -0.999391, -0.0348995],
                             [0.587427, -0.808524, -0.0348995]])

        r = np.sqrt(points3D[:, 0] ** 2 + points3D[:, 1]
                    ** 2 + points3D[:, 2] ** 2)
        t = r / (r + points3D[:, 2])
        x = r * points3D[:, 0]
        y = r * points3D[:, 1]
        points2D = np.column_stack((x, y))

        # InitialGraph.plot_voronoi(points2D, adjacency_matrix)

        circle = plt.Circle((0, 0), 1, color='r', alpha=0.25, fill=False)
        plt.scatter(points2D[:, 0], points2D[:, 1])
        plt.gca().add_patch(circle)

        for i in range(len(points2D)):
            plt.text(points2D[i, 0] - 0.02, points2D[i, 1] + 0.025, channels[i])

        for i in range(len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix[i])):
                if adjacency_matrix[i, j] == 1:
                    plt.plot([points2D[i, 0], points2D[j, 0]], [
                        points2D[i, 1], points2D[j, 1]], 'k-', alpha=0.5)

                    vector1 = points2D[j] - points2D[i]
                    vector2 = np.array([1, 0])
                    angle = InitialGraph.calculate_angle(vector1, vector2)
                    print(
                        f"Angle between {channels[i]} and {channels[j]}: {angle:.2f} degrees")

        plt.axis('equal')
        plt.show()

    def plot32(adjacency_matrix):
        channels = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3',
                    'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1',
                    'Oz', 'O2']

        points3D = [[-0.308829, 0.950477, -0.0348995], [0.308829, 0.950477, -0.0348995],
                    [-0.406247, 0.871199, 0.275637], [0.406247, 0.871199,
                                                      0.275637], [-0.808524, 0.587427, -0.0348995],
                    [-0.545007, 0.673028, 0.5], [0, 0.71934,
                                                 0.694658], [0.545007, 0.673028, 0.5],
                    [0.808524, 0.587427, -0.0348995], [-0.887888,
                                                       0.340828, 0.309017], [-0.37471, 0.37471, 0.848048],
                    [0.37471, 0.37471, 0.848048], [0.887888, 0.340828,
                                                   0.309017], [-0.999391, 0, -0.0348995],
                    [-0.71934, 0, 0.694658], [0, 0, 1], [0.71934,
                                                         0, 0.694658], [0.999391, 0, -0.0348995],
                    [-0.887888, -0.340828, 0.309017], [-0.37471, -
            0.37471, 0.848048], [0.37471, -0.37471, 0.848048],
                    [0.887888, -0.340828, 0.309017], [-0.808524, -
            0.587427, -0.0348995], [-0.545007, -0.673028, 0.5],
                    [0, -0.71934, 0.694658], [0.545007, -0.673028,
                                              0.5], [0.808524, -0.587427, -0.0348995],
                    [-0.406247, -0.871199, 0.275637], [0.406247, -0.871199, 0.275637],
                    [-0.308829, -0.950477, -0.0348995], [0, -0.999391, -0.0348995], [0.308829, -0.950477, -0.0348995]]
        points3D = np.array(points3D)

        r = np.sqrt(points3D[:, 0] ** 2 + points3D[:, 1]
                    ** 2 + points3D[:, 2] ** 2)
        t = r / (r + points3D[:, 2])
        x = r * points3D[:, 0]
        y = r * points3D[:, 1]
        points2D = np.column_stack((x, y))
        # InitialGraph.plot_voronoi(points2D, adjacency_matrix)

        circle = plt.Circle((0, 0), 1, color='r', alpha=0.25, fill=False)
        plt.scatter(points2D[:, 0], points2D[:, 1])
        plt.gca().add_patch(circle)

        for i in range(len(points2D)):
            plt.text(points2D[i, 0] - 0.02,
                     points2D[i, 1] + 0.025, channels[i])

        for i in range(len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix[i])):
                if adjacency_matrix[i, j] == 1:
                    plt.plot([points2D[i, 0], points2D[j, 0]], [
                        points2D[i, 1], points2D[j, 1]], 'k-', alpha=0.5)

                    vector1 = points2D[j] - points2D[i]
                    vector2 = np.array([1, 0])
                    angle = InitialGraph.calculate_angle(vector1, vector2)
                    print(
                        f"Angle between {channels[i]} and {channels[j]}: {angle:.2f} degrees")

        plt.axis('equal')
        plt.show()

    def toWeigthedMatriz(posiciones, connectivity_matrix):
        electrodes = list(posiciones.keys())
        weighted_graph = np.zeros((len(electrodes), len(electrodes)))

        for i in range(len(electrodes)):
            for j in range(i + 1, len(electrodes)):
                electrode1 = electrodes[i]
                electrode2 = electrodes[j]

                if connectivity_matrix[i][j] == 1:
                    distance = InitialGraph.distance3D(
                        posiciones[electrode1], posiciones[electrode2])
                    weighted_graph[i][j] = distance
                    weighted_graph[j][i] = distance

        return weighted_graph


class WeightedGraph:
    directed = True
    _vertices = []  # The list of vertices.
    _adjacency_list = []

    def __init__(self, directed=False):
        self.directed = directed
        self._vertices = []
        self._adjacency_list = {}  # Cambiar el nombre de la variable

    def vertices(self):
        # Cambiar el nombre de la variable
        return list(self._adjacency_list.keys())

    def edges(self):
        e = []
        for v in self._adjacency_list:
            for edge in self._adjacency_list[v]:
                if (edge[0], v, edge[1]) not in e:
                    e.append((v, edge[0], edge[1]))
        return e

    def number_of_vertices(self):
        return len(self._adjacency_list)

    def add_vertex(self, v):
        if v not in self._adjacency_list:
            self._adjacency_list[v] = []

    def add_edge(self, v1, v2, e=0):
        if v1 not in self._adjacency_list or v2 not in self._adjacency_list:
            print("El vertice no existe")
            return
        if not self.directed and v1 == v2:
            print("Error en los vertices")
            return
        for vertex, weight in self._adjacency_list[v1]:
            if vertex == v2:
                print(f"El vertice ya existe")
                return
        self._adjacency_list[v1].append((v2, e))
        if not self.directed:
            self._adjacency_list[v2].append((v1, e))

    def adjacent_vertices(self, v):
        if v not in self._adjacency_list:
            print("El vertice no existe")
            return []
        return self._adjacency_list[v]

    # Prims algorithm for part 3-----------------------------------------------------------------
    def kruskal(self):
        selected_edges = []
        min_spanning_tree_cost = 0

        # Sort the edges by weight
        edges = self.edges()
        edges.sort(key=lambda x: x[2])

        # Initialize disjoint-set data structure
        disjoint_set = DisjointSet(self.vertices())

        for edge in edges:
            v1, v2, cost = edge
            if not disjoint_set.are_connected(v1, v2):
                selected_edges.append(edge)
                min_spanning_tree_cost += cost
                disjoint_set.union(v1, v2)

        return selected_edges, min_spanning_tree_cost

    def prim(self):
        selected_vertices = []
        min_spanning_tree_cost = 0

        # Start with an arbitrary vertex (you can choose any vertex)
        start_vertex = list(self._adjacency_list.keys())[0]
        selected_vertices.append(start_vertex)

        while len(selected_vertices) < self.number_of_vertices():
            min_cost = float('inf')
            next_vertex = None

            for v in selected_vertices:
                for neighbor, cost in self._adjacency_list[v]:
                    if neighbor not in selected_vertices and cost < min_cost:
                        min_cost = cost
                        next_vertex = neighbor

            if next_vertex is None:
                # No solution found
                return None, None

            selected_vertices.append(next_vertex)
            min_spanning_tree_cost += min_cost

        return selected_vertices, min_spanning_tree_cost

    def prim_edges(self):
        selected_edges = []
        selected_vertices = set()
        min_spanning_tree_cost = 0

        # Inicia con un vértice arbitrario (puedes elegir cualquier vértice)
        start_vertex = list(self._adjacency_list.keys())[0]
        selected_vertices.add(start_vertex)

        while len(selected_vertices) < self.number_of_vertices():
            min_cost = float('inf')
            next_edge = None

            for v in selected_vertices:
                for neighbor, cost in self._adjacency_list[v]:
                    if neighbor not in selected_vertices and cost < min_cost:
                        min_cost = cost
                        next_edge = (v, neighbor)

            if next_edge is None:
                # No se encontró solución
                return None, None

            selected_vertices.add(next_edge[1])
            selected_edges.append(next_edge)
            min_spanning_tree_cost += min_cost

        return selected_edges, min_spanning_tree_cost

    def print_graph(self):
        for vertex in self._adjacency_list:
            for edges in self._adjacency_list[vertex]:
                print(vertex, " -> ", edges[0], " edge weight: ", edges[1])


def graham_scan(points):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def graham_compare(p1, p2):
        o = orientation(p0, p1, p2)
        if o == 0:
            return -1 if distance3D(p0, p2) >= distance3D(p0, p1) else 1
        return -1 if o == 2 else 1

    def distance3D(punto1, punto2):
        x1, y1, z1 = punto1
        x2, y2, z2 = punto2
        return math.sqrt((x2 - x1) * 2 + (y2 - y1) * 2 + (z2 - z1) ** 2)

    n = len(points)
    if n < 3:
        return []

    p0 = min(points, key=lambda point: (point[2], point[1]))
    sorted_points = sorted(points, key=lambda point: (math.atan2(point[1] - p0[1], point[0] - p0[0])))
    stack = [p0, sorted_points[0], sorted_points[1]]
    i = 2

    while i < n:
        top = stack[-1]
        next_point = sorted_points[i]

        # Elimina puntos no necesarios del casco convexo.
        while len(stack) > 1 and orientation(stack[-2], top, next_point) != 2:
            stack.pop()

        stack.append(next_point)
        i += 1

    return stack


class TreeNode:
    def __init__(self, parent, v, c):
        self.parent = parent
        self.v = v
        self.c = c
        self.visited = False

    def path(self):
        node = self
        path = []
        while node:
            path.insert(0, node.v)
            node = node.parent
        return path

    def __lt__(self, other):
        return self.c < other.c

    def bfs(graph, v0, vg):
        if v0 not in graph.vertices():
            print("El vertice", v0, "no existe")
            return None
        if vg not in graph.vertices():
            print("El vertice", vg, "no existe")
            return None

        frontier = Queue()
        start_node = TreeNode(None, v0, 0)
        frontier.put(start_node)

        visited = set()

        while not frontier.empty():
            node = frontier.get()
            if node.v == vg:
                return {"Path": node.path(), "Cost": node.c}
            if node.v not in visited:
                visited.add(node.v)
                adjacent_vertices = graph.adjacent_vertices(node.v)
                for vertex, cost in adjacent_vertices:
                    new_node = TreeNode(node, vertex, cost + node.c)
                    frontier.put(new_node)

        print("No se encontró un camino hasta el nodo", vg)
        return None

    def dfs(graph, v0, vg):
        if v0 not in graph.vertices():
            print("El vertice", v0, "no existe")
        if vg not in graph.vertices():
            print("El vertice", vg, "no existe")

        frontier = LifoQueue()
        frontier.put(TreeNode(None, v0, 0))
        explored_set = set()

        while True:
            if frontier.empty():
                return None
            node = frontier.get()
            if node.v == vg:
                return {"Path": node.path(), "Cost": node.c}
            if node.v not in explored_set:
                adjacent_vertices = graph.adjacent_vertices(node.v)
                for vertex in adjacent_vertices:
                    frontier.put(TreeNode(node, vertex[0], vertex[1] + node.c))
                explored_set.add(node.v)

    def ucs(graph, v0, vg):
        if v0 not in graph.vertices():
            print("El vertice", v0, "no existe")
        if vg not in graph.vertices():
            print("El vertice", vg, "no existe")
        frontier = PriorityQueue()
        frontier.put((0, TreeNode(None, v0, 0)))
        explored_set = {}

        while True:
            if frontier.empty():
                return None
            node = frontier.get()[1]
            if node.v == vg:
                return {"Path": node.path(), "Cost": node.c}
            if node.v not in explored_set:
                adjacent_vertices = graph.adjacent_vertices(node.v)
                for vertex in adjacent_vertices:
                    cost = vertex[1] + node.c
                    frontier.put(
                        (cost, TreeNode(node, vertex[0], vertex[1] + node.c)))
            explored_set[node.v] = 0

    def floyd_marshall(self, start_vertex, end_vertex):
        vertices = self.vertices()

        if start_vertex not in vertices:
            print(f"The start vertex {start_vertex} is not in the list of vertices.")
            return None

        n = len(vertices)
        distance = [[float('inf')] * n for _ in range(n)]
        next_node = [[-1] * n for _ in range(n)]

        for i in range(n):
            for edge in self._adjacency_list[vertices[i]]:
                neighbor_index = vertices.index(edge[0])
                distance[i][neighbor_index] = edge[1]
                next_node[i][neighbor_index] = neighbor_index

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distance[i][j] > distance[i][k] + distance[k][j]:
                        distance[i][j] = distance[i][k] + distance[k][j]
                        next_node[i][j] = next_node[i][k]

        try:
            start_index = vertices.index(start_vertex)
            end_index = vertices.index(end_vertex)
        except ValueError as e:
            print(f"The vertex {e.args[0]} is not in the list of vertices.")
            return None

        shortest_distance = distance[start_index][end_index]

        return shortest_distance


def plotMinSpanningTree(graph, kruskal_selected_edges):
    idAndNameOfVertices = []
    vertices = graph.vertices()

    for i in range(len(vertices)):
        newTuple = (i, vertices[i])
        idAndNameOfVertices.append(newTuple)

    # Create graph
    G = nx.Graph()
    for i in range(len(idAndNameOfVertices)):
        G.add_node(idAndNameOfVertices[i][0], label=idAndNameOfVertices[i][1])

    # Create edges
    for i in range(len(kruskal_selected_edges)):
        nameOfStatingVertex = kruskal_selected_edges[i][0]
        nameOfEndingVertex = kruskal_selected_edges[i][1]
        weight = round(kruskal_selected_edges[i][2], 2)
        startingVertexId = 0
        endingVertexId = 0

        for k in range(len(idAndNameOfVertices)):
            if nameOfStatingVertex == idAndNameOfVertices[k][1]:
                startingVertexId = idAndNameOfVertices[k][0]

            if nameOfEndingVertex == idAndNameOfVertices[k][1]:
                endingVertexId = idAndNameOfVertices[k][0]

        G.add_edge(startingVertexId, endingVertexId, weight=weight)

        # Set node positions
    pos = nx.spring_layout(G)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'))

    # Draw edges with weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_edges(G, pos)

    # Show graph
    plt.axis('off')
    plt.show()


def plotMatrices8(matrizLectura, matrizMemoria, matrizOperaciones):
    InitialGraph.plot8(matrizLectura)
    InitialGraph.plot8(matrizMemoria)
    InitialGraph.plot8(matrizOperaciones)


def plotMatrices32(matrizLectura, matrizMemoria, matrizOperaciones):
    InitialGraph.plot32(matrizLectura)
    InitialGraph.plot32(matrizMemoria)
    InitialGraph.plot32(matrizOperaciones)


def plotConvexHull(prim_edges, posiciones8):
    unique_points = set()
    for edge in prim_edges:
        unique_points.add(edge[0])
        unique_points.add(edge[1])
    unique_points = list(unique_points)

    points_3d = [posiciones8[point] for point in unique_points]

    convex_hull = graham_scan(points_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for point in points_3d:
        ax.scatter(point[0], point[1], point[2], c='blue', marker='o')

    # edges of the minimum spanning tree.
    for edge in prim_edges:
        start_point = posiciones8[edge[0]]
        end_point = posiciones8[edge[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]],
                c='green', linestyle='--')

    convex_hull.append(convex_hull[0])  # Para cerrar el casco convexo.
    convex_hull_points = np.array(convex_hull)
    ax.plot(convex_hull_points[:, 0], convex_hull_points[:, 1], convex_hull_points[:, 2], c='red', linestyle='-',
            linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def main():
    '''
        8 electrodos:
    '''
    matrizLecturaSujeto3 = np.loadtxt('S3/Lectura.txt')
    matrizMemoriaSujeto3 = np.loadtxt('S3/Memoria.txt')
    matrizOperacionesSujeto3 = np.loadtxt('S3/Operaciones.txt')

    matrizLecturaSujeto4 = np.loadtxt('S4/Lectura.txt')
    matrizMemoriaSujeto4 = np.loadtxt('S4/Memoria.txt')
    matrizOperacionesSujeto4 = np.loadtxt('S4/Operaciones.txt')

    matrizLecturaSujeto5 = np.loadtxt('S5/Lectura.txt')
    matrizMemoriaSujeto5 = np.loadtxt('S5/Memoria.txt')
    matrizOperacionesSujeto5 = np.loadtxt('S5/Operaciones.txt')

    matrizLecturaSujeto6 = np.loadtxt('S6/Lectura.txt')
    matrizMemoriaSujeto6 = np.loadtxt('S6/Memoria.txt')
    matrizOperacionesSujeto6 = np.loadtxt('S6/Operaciones.txt')

    plotMatrices8(matrizLecturaSujeto3, matrizMemoriaSujeto3, matrizOperacionesSujeto3)
    plotMatrices8(matrizLecturaSujeto4, matrizMemoriaSujeto4, matrizOperacionesSujeto4)
    plotMatrices8(matrizLecturaSujeto5, matrizMemoriaSujeto5, matrizOperacionesSujeto5)
    plotMatrices8(matrizLecturaSujeto6, matrizMemoriaSujeto6, matrizOperacionesSujeto6)

    posiciones8 = {
        'Fz': (0, 0.71934, 0.694658),
        'C3': (-0.71934, 0, 0.694658),
        'Cz': (0, 0, 1),
        'C4': (0.71934, 0, 0.694658),
        'Pz': (0, -0.71934, 0.694658),
        'PO7': (-0.587427, -0.808524, -0.0348995),
        'Oz': (0, -0.999391, -0.0348995),
        'PO8': (0.587427, -0.808524, -0.0348995),
    }

    posName8 = list(posiciones8.keys())
    ItoC8 = {str(indice): clave for indice, clave in enumerate(posiciones8)}

    # GRAFO S3
    matrizPonderada8_3_1 = InitialGraph.toWeigthedMatriz(posiciones8, matrizLecturaSujeto3)
    matrizPonderada8_3_2 = InitialGraph.toWeigthedMatriz(posiciones8, matrizMemoriaSujeto3)
    matrizPonderada8_3_3 = InitialGraph.toWeigthedMatriz(posiciones8, matrizOperacionesSujeto3)

    graph8_3_1 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_3_1)):
        for j in range(len(matrizPonderada8_3_1[i])):
            if (matrizPonderada8_3_1[i][j] != 0):
                graph8_3_1.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_3_1)):
        for j in range(len(matrizPonderada8_3_1[i])):
            if (matrizPonderada8_3_1[i][j] != 0):
                graph8_3_1.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_3_1[i][j])

    graph8_3_2 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_3_2)):
        for j in range(len(matrizPonderada8_3_2[i])):
            if (matrizPonderada8_3_2[i][j] != 0):
                graph8_3_2.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_3_2)):
        for j in range(len(matrizPonderada8_3_2[i])):
            if (matrizPonderada8_3_2[i][j] != 0):
                graph8_3_2.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_3_2[i][j])

    graph8_3_3 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_3_3)):
        for j in range(len(matrizPonderada8_3_3[i])):
            if (matrizPonderada8_3_3[i][j] != 0):
                graph8_3_3.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_3_3)):
        for j in range(len(matrizPonderada8_3_3[i])):
            if (matrizPonderada8_3_3[i][j] != 0):
                graph8_3_3.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_3_3[i][j])

    # GRAFO S4
    matrizPonderada8_4_1 = InitialGraph.toWeigthedMatriz(posiciones8, matrizLecturaSujeto4)
    matrizPonderada8_4_2 = InitialGraph.toWeigthedMatriz(posiciones8, matrizMemoriaSujeto4)
    matrizPonderada8_4_3 = InitialGraph.toWeigthedMatriz(posiciones8, matrizOperacionesSujeto4)

    graph8_4_1 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_4_1)):
        for j in range(len(matrizPonderada8_4_1[i])):
            if (matrizPonderada8_4_1[i][j] != 0):
                graph8_4_1.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_4_1)):
        for j in range(len(matrizPonderada8_4_1[i])):
            if (matrizPonderada8_4_1[i][j] != 0):
                graph8_4_1.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_4_1[i][j])

    graph8_4_2 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_4_2)):
        for j in range(len(matrizPonderada8_4_2[i])):
            if (matrizPonderada8_4_2[i][j] != 0):
                graph8_4_2.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_4_2)):
        for j in range(len(matrizPonderada8_4_2[i])):
            if (matrizPonderada8_4_2[i][j] != 0):
                graph8_4_2.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_4_2[i][j])

    graph8_4_3 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_4_3)):
        for j in range(len(matrizPonderada8_4_3[i])):
            if (matrizPonderada8_4_3[i][j] != 0):
                graph8_4_3.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_4_3)):
        for j in range(len(matrizPonderada8_4_3[i])):
            if (matrizPonderada8_4_3[i][j] != 0):
                graph8_4_3.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_4_3[i][j])

    # GRAFO S5
    matrizPonderada8_5_1 = InitialGraph.toWeigthedMatriz(posiciones8, matrizLecturaSujeto5)
    matrizPonderada8_5_2 = InitialGraph.toWeigthedMatriz(posiciones8, matrizMemoriaSujeto5)
    matrizPonderada8_5_3 = InitialGraph.toWeigthedMatriz(posiciones8, matrizOperacionesSujeto5)

    graph8_5_1 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_5_1)):
        for j in range(len(matrizPonderada8_5_1[i])):
            if (matrizPonderada8_5_1[i][j] != 0):
                graph8_5_1.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_5_1)):
        for j in range(len(matrizPonderada8_5_1[i])):
            if (matrizPonderada8_5_1[i][j] != 0):
                graph8_5_1.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_5_1[i][j])

    graph8_5_2 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_5_2)):
        for j in range(len(matrizPonderada8_5_2[i])):
            if (matrizPonderada8_5_2[i][j] != 0):
                graph8_5_2.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_5_2)):
        for j in range(len(matrizPonderada8_5_2[i])):
            if (matrizPonderada8_5_2[i][j] != 0):
                graph8_5_2.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_5_2[i][j])

    graph8_5_3 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_5_3)):
        for j in range(len(matrizPonderada8_5_3[i])):
            if (matrizPonderada8_5_3[i][j] != 0):
                graph8_5_3.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_5_3)):
        for j in range(len(matrizPonderada8_5_3[i])):
            if (matrizPonderada8_5_3[i][j] != 0):
                graph8_5_3.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_5_3[i][j])

    # GRAFO S6
    matrizPonderada8_6_1 = InitialGraph.toWeigthedMatriz(posiciones8, matrizLecturaSujeto6)
    matrizPonderada8_6_2 = InitialGraph.toWeigthedMatriz(posiciones8, matrizMemoriaSujeto6)
    matrizPonderada8_6_3 = InitialGraph.toWeigthedMatriz(posiciones8, matrizOperacionesSujeto6)

    graph8_6_1 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_6_1)):
        for j in range(len(matrizPonderada8_6_1[i])):
            if (matrizPonderada8_6_1[i][j] != 0):
                graph8_6_1.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_6_1)):
        for j in range(len(matrizPonderada8_6_1[i])):
            if (matrizPonderada8_6_1[i][j] != 0):
                graph8_6_1.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_6_1[i][j])

    graph8_6_2 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_6_2)):
        for j in range(len(matrizPonderada8_6_2[i])):
            if (matrizPonderada8_6_2[i][j] != 0):
                graph8_6_2.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_6_2)):
        for j in range(len(matrizPonderada8_6_2[i])):
            if (matrizPonderada8_6_2[i][j] != 0):
                graph8_6_2.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_6_2[i][j])

    graph8_6_3 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8_6_3)):
        for j in range(len(matrizPonderada8_6_3[i])):
            if (matrizPonderada8_6_3[i][j] != 0):
                graph8_6_3.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8_6_3)):
        for j in range(len(matrizPonderada8_6_3[i])):
            if (matrizPonderada8_6_3[i][j] != 0):
                graph8_6_3.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8_6_3[i][j])

    print('\n\n-----MATRIZ 8 ELECTRODOS-----\n\n')
    # graphPaths(graph8_4_1, 'Fz', 'PO8')
    # graphPaths(graph8_4_1, 'C3', 'Oz')
    # graphPaths(graph8_4_1, 'P07', 'C4')

    '''
        BFS:
    '''
    print('\n---BFS---\n')
    print('\nSujeto 3:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph8_3_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_3_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_3_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph8_3_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_3_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_3_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph8_3_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_3_2, 'C3', 'Oz'))
    print(TreeNode.bfs(graph8_3_3, 'C3', 'Oz'))

    print('\nSujeto 4:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph8_4_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_4_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_4_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph8_4_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_4_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_4_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph8_4_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_4_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_4_3, 'C3', 'Oz'))

    print('\nSujeto 5:\n')
    print('\nFz - PO8\n')
    # Fz - PO8
    print('Matriz Lectura: ', TreeNode.bfs(graph8_5_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_5_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_5_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph8_5_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_5_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_5_3, 'C3', 'Oz'))
    # P07 - C4
    print('Matriz Lectura: ', TreeNode.bfs(graph8_5_1, 'C3', 'Oz'))
    print(TreeNode.bfs(graph8_5_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_5_3, 'C3', 'Oz'))

    print('\nSujeto 6:\n')
    # Fz - PO8
    print('Matriz Lectura: ', TreeNode.bfs(graph8_6_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_6_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_6_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph8_6_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_6_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_6_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph8_6_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.bfs(graph8_6_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph8_6_3, 'C3', 'Oz'))

    '''
        DFS:
    '''
    print('\n---DFS---\n')
    print('\nSujeto 3:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_3_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_3_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_3_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_3_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_3_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_3_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_3_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_3_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_3_3, 'P07', 'C4'))

    print('\nSujeto 4:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_4_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_4_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_4_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_4_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_4_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_4_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_4_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_4_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_4_3, 'P07', 'C4'))

    print('\nSujeto 5:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_5_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_5_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_5_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_5_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_5_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_5_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_5_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_5_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_5_3, 'P07', 'C4'))

    print('\nSujeto 6:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_6_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_6_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_6_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_6_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_6_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_6_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph8_6_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.dfs(graph8_6_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph8_6_3, 'P07', 'C4'))

    '''
        UCS:
    '''
    print('\n---UCS---\n')
    print('\nSujeto 3:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_3_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_3_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_3_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_3_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_3_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_3_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_3_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_3_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_3_3, 'P07', 'C4'))

    print('\nSujeto 4:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_4_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_4_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_4_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_4_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_4_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_4_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_4_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_4_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_4_3, 'P07', 'C4'))

    print('\nSujeto 5:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_5_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_5_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_5_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_5_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_5_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_5_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_5_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_5_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_5_3, 'P07', 'C4'))

    print('\nSujeto 6:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_6_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_6_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_6_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_6_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_6_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_6_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph8_6_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.ucs(graph8_6_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph8_6_3, 'P07', 'C4'))
    '''
        Floyd - Marshall:
    '''
    print('\n---Floyd - Marshall---\n')
    print('\nSujeto 3:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_3_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_3_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_3_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_3_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_3_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_3_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_3_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_3_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_3_3, 'P07', 'C4'))

    print('\nSujeto 4:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_4_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_4_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_4_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_4_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_4_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_4_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_4_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_4_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_4_3, 'P07', 'C4'))

    print('\nSujeto 5:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_5_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_5_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_5_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_5_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_5_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_5_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_5_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_5_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_5_3, 'P07', 'C4'))

    print('\nSujeto 6:\n')
    # Fz - PO8
    print('\nFz - PO8\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_6_1, 'Fz', 'PO8'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_6_2, 'Fz', 'PO8'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_6_3, 'Fz', 'PO8'))
    # C3 - Oz
    print('\nC3 - Oz\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_6_1, 'C3', 'Oz'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_6_2, 'C3', 'Oz'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_6_3, 'C3', 'Oz'))
    # P07 - C4
    print('\nP07 - C4\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph8_6_1, 'P07', 'C4'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph8_6_2, 'P07', 'C4'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph8_6_3, 'P07', 'C4'))

    '''
        32 electrodos:
    '''

    matrizLectura32A = np.loadtxt('S0A/Lectura.txt')
    matrizMemoria32A = np.loadtxt('S0A/Memoria.txt')
    matrizOperaciones32A = np.loadtxt('S0A/Operaciones.txt')

    matrizLectura32B = np.loadtxt('S0B/Lectura.txt')
    matrizMemoria32B = np.loadtxt('S0B/Memoria.txt')
    matrizOperaciones32B = np.loadtxt('S0B/Operaciones.txt')

    plotMatrices32(matrizLectura32A, matrizMemoria32A, matrizOperaciones32A)
    plotMatrices32(matrizLectura32B, matrizMemoria32B, matrizOperaciones32B)

    posiciones32 = {
        'Fp1': (-0.308829, 0.950477, -0.0348995),
        'Fp2': (0.308829, 0.950477, -0.0348995),
        'AF3': (-0.406247, 0.871199, 0.275637),
        'AF4': (0.406247, 0.871199, 0.275637),
        'F7': (-0.808524, 0.587427, -0.0348995),
        'F3': (-0.545007, 0.673028, 0.5),
        'Fz': (0, 0.71934, 0.694658),
        'F4': (0.545007, 0.673028, 0.5),
        'F8': (0.808524, 0.587427, -0.0348995),
        'FC5': (-0.887888, 0.340828, 0.309017),
        'FC1': (-0.37471, 0.37471, 0.848048),
        'FC2': (0.37471, 0.37471, 0.848048),
        'FC6': (0.887888, 0.340828, 0.309017),
        'T7': (-0.999391, 0, -0.0348995),
        'C3': (-0.71934, 0, 0.694658),
        'Cz': (0, 0, 1),
        'C4': (0.71934, 0, 0.694658),
        'T8': (0.999391, 0, -0.0348995),
        'CP5': (-0.887888, -0.340828, 0.309017),
        'CP1': (-0.37471, -0.37471, 0.848048),
        'CP2': (0.37471, -0.37471, 0.848048),
        'CP6': (0.887888, -0.340828, 0.309017),
        'P7': (-0.808524, -0.587427, -0.0348995),
        'P3': (-0.545007, -0.673028, 0.5),
        'Pz': (0, -0.71934, 0.694658),
        'P4': (0.545007, -0.673028, 0.5),
        'P8': (0.808524, -0.587427, -0.0348995),
        'PO3': (-0.406247, -0.871199, 0.275637),
        'PO4': (0.406247, -0.871199, 0.275637),
        'O1': (-0.308829, -0.950477, -0.0348995),
        'Oz': (0, -0.999391, -0.0348995),
        'O2': (0.308829, -0.950477, -0.0348995),
    }

    posName32 = list(posiciones32.keys())
    ItoC32 = {str(indice): clave for indice, clave in enumerate(posiciones32)}

    # Matriz S0A
    matrizPonderada32A_1 = InitialGraph.toWeigthedMatriz(posiciones32, matrizLectura32A)
    matrizPonderada32A_2 = InitialGraph.toWeigthedMatriz(posiciones32, matrizMemoria32A)
    matrizPonderada32A_3 = InitialGraph.toWeigthedMatriz(posiciones32, matrizOperaciones32A)

    graph32A_1 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada32A_1)):
        for j in range(len(matrizPonderada32A_1[i])):
            if (matrizPonderada32A_1[i][j] != 0):
                graph32A_1.add_vertex(posName32[i])

    for i in range(len(matrizPonderada32A_1)):
        for j in range(len(matrizPonderada32A_1[i])):
            if (matrizPonderada32A_1[i][j] != 0):
                graph32A_1.add_edge(ItoC32[str(i)], ItoC32[str(j)], matrizPonderada32A_1[i][j])

    graph32A_2 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada32A_2)):
        for j in range(len(matrizPonderada32A_2[i])):
            if (matrizPonderada32A_2[i][j] != 0):
                graph32A_2.add_vertex(posName32[i])

    for i in range(len(matrizPonderada32A_2)):
        for j in range(len(matrizPonderada32A_2[i])):
            if (matrizPonderada32A_2[i][j] != 0):
                graph32A_2.add_edge(ItoC32[str(i)], ItoC32[str(j)], matrizPonderada32A_2[i][j])

    graph32A_3 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada32A_3)):
        for j in range(len(matrizPonderada32A_3[i])):
            if (matrizPonderada32A_3[i][j] != 0):
                graph32A_3.add_vertex(posName32[i])

    for i in range(len(matrizPonderada32A_3)):
        for j in range(len(matrizPonderada32A_3[i])):
            if (matrizPonderada32A_3[i][j] != 0):
                graph32A_3.add_edge(ItoC32[str(i)], ItoC32[str(j)], matrizPonderada32A_3[i][j])

    # Matriz S0B
    matrizPonderada32B_1 = InitialGraph.toWeigthedMatriz(posiciones32, matrizLectura32B)
    matrizPonderada32B_2 = InitialGraph.toWeigthedMatriz(posiciones32, matrizMemoria32B)
    matrizPonderada32B_3 = InitialGraph.toWeigthedMatriz(posiciones32, matrizOperaciones32B)

    graph32B_1 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada32B_1)):
        for j in range(len(matrizPonderada32B_1[i])):
            if (matrizPonderada32B_1[i][j] != 0):
                graph32B_1.add_vertex(posName32[i])

    for i in range(len(matrizPonderada32B_1)):
        for j in range(len(matrizPonderada32B_1[i])):
            if (matrizPonderada32B_1[i][j] != 0):
                graph32B_1.add_edge(ItoC32[str(i)], ItoC32[str(j)], matrizPonderada32B_1[i][j])

    graph32B_2 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada32B_2)):
        for j in range(len(matrizPonderada32B_2[i])):
            if (matrizPonderada32B_2[i][j] != 0):
                graph32B_2.add_vertex(posName32[i])

    for i in range(len(matrizPonderada32B_2)):
        for j in range(len(matrizPonderada32B_2[i])):
            if (matrizPonderada32B_2[i][j] != 0):
                graph32B_2.add_edge(ItoC32[str(i)], ItoC32[str(j)], matrizPonderada32B_2[i][j])

    graph32B_3 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada32B_3)):
        for j in range(len(matrizPonderada32B_3[i])):
            if (matrizPonderada32B_3[i][j] != 0):
                graph32B_3.add_vertex(posName32[i])

    for i in range(len(matrizPonderada32B_3)):
        for j in range(len(matrizPonderada32B_3[i])):
            if (matrizPonderada32B_3[i][j] != 0):
                graph32B_3.add_edge(ItoC32[str(i)], ItoC32[str(j)], matrizPonderada32B_3[i][j])

    print('\n\n-----MATRIZ 32 ELECTRODOS-----\n\n')

    '''
        BFS:
    '''
    print('\n---BFS---\n')
    # F7 - PO4
    print('\nF7 - PO4\n')
    print('Matriz Lectura: ', TreeNode.bfs(graph32A_1, 'F7', 'PO4'))
    print('Matriz Memoria: ', TreeNode.bfs(graph32A_2, 'F7', 'PO4'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph32A_3, 'F7', 'PO4'))
    # CP5 - O2
    print('Matriz Lectura: ', TreeNode.bfs(graph32A_1, 'CP5', 'O2'))
    print('Matriz Memoria: ', TreeNode.bfs(graph32A_2, 'CP5', 'O2'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph32A_3, 'CP5', 'O2'))
    # P4 - T7
    print('Matriz Lectura: ', TreeNode.bfs(graph32A_1, 'P4', 'T7'))
    print('Matriz Memoria: ', TreeNode.bfs(graph32A_2, 'P4', 'T7'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph32A_3, 'P4', 'T7'))
    # AF3 - CP6
    print('Matriz Lectura: ', TreeNode.bfs(graph32A_1, 'AF3', 'CP6'))
    print('Matriz Memoria: ', TreeNode.bfs(graph32A_2, 'AF3', 'CP6'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph32A_3, 'AF3', 'CP6'))
    # F8 - CP2
    print('Matriz Lectura: ', TreeNode.bfs(graph32A_1, 'F8', 'CP2'))
    print('Matriz Memoria: ', TreeNode.bfs(graph32A_2, 'F8', 'CP2'))
    print('Matriz Operaciones: ', TreeNode.bfs(graph32A_3, 'F8', 'CP2'))

    '''
        DFS:
    '''
    print('\n---DFS---\n')
    # F7 - PO4
    print('\nF7 - PO4\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph32A_1, 'F7', 'PO4'))
    print('Matriz Memoria: ', TreeNode.dfs(graph32A_2, 'F7', 'PO4'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph32A_3, 'F7', 'PO4'))
    # CP5 - O2
    print('\nCP5 - O2\n')
    print('Matriz Lectura: ', TreeNode.dfs(graph32A_1, 'CP5', 'O2'))
    print('Matriz Memoria: ', TreeNode.dfs(graph32A_2, 'CP5', 'O2'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph32A_3, 'CP5', 'O2'))
    # P4 - T7
    print('Matriz Lectura: ', TreeNode.dfs(graph32A_1, 'P4', 'T7'))
    print('Matriz Memoria: ', TreeNode.dfs(graph32A_2, 'P4', 'T7'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph32A_3, 'P4', 'T7'))
    # AF3 - CP6
    print('Matriz Lectura: ', TreeNode.dfs(graph32A_1, 'AF3', 'CP6'))
    print('Matriz Memoria: ', TreeNode.dfs(graph32A_2, 'AF3', 'CP6'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph32A_3, 'AF3', 'CP6'))
    # F8 - CP2
    print('Matriz Lectura: ', TreeNode.dfs(graph32A_1, 'F8', 'CP2'))
    print('Matriz Memoria: ', TreeNode.dfs(graph32A_2, 'F8', 'CP2'))
    print('Matriz Operaciones: ', TreeNode.dfs(graph32A_3, 'F8', 'CP2'))

    '''
        UCS:
    '''
    print('\n---UCS---\n')
    # F7 - PO4
    print('\nF7 - PO4\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph32A_1, 'F7', 'PO4'))
    print('Matriz Memoria: ', TreeNode.ucs(graph32A_2, 'F7', 'PO4'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph32A_3, 'F7', 'PO4'))
    # CP5 - O2
    print('\nCP5 - O2\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph32A_1, 'CP5', 'O2'))
    print('Matriz Memoria: ', TreeNode.ucs(graph32A_2, 'CP5', 'O2'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph32A_3, 'CP5', 'O2'))
    # P4 - T7
    print('\nP4 - T7\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph32A_1, 'P4', 'T7'))
    print('Matriz Memoria: ', TreeNode.ucs(graph32A_2, 'P4', 'T7'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph32A_3, 'P4', 'T7'))
    # AF3 - CP6
    print('\nAF3 - CP6\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph32A_1, 'AF3', 'CP6'))
    print('Matriz Memoria: ', TreeNode.ucs(graph32A_2, 'AF3', 'CP6'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph32A_3, 'AF3', 'CP6'))
    # F8 - CP2
    print('\nF8 - CP2\n')
    print('Matriz Lectura: ', TreeNode.ucs(graph32A_1, 'F8', 'CP2'))
    print('Matriz Memoria: ', TreeNode.ucs(graph32A_2, 'F8', 'CP2'))
    print('Matriz Operaciones: ', TreeNode.ucs(graph32A_3, 'F8', 'CP2'))

    '''
        Floyd - Marshall:
    '''
    print('\n---Floyd - Marshall---\n')
    # F7 - PO4
    print('\nF7 - PO4\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph32A_1, 'F7', 'PO4'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph32A_2, 'F7', 'PO4'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph32A_3, 'F7', 'PO4'))
    # CP5 - O2
    print('\nCP5 - O2\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph32A_1, 'CP5', 'O2'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph32A_2, 'CP5', 'O2'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph32A_3, 'CP5', 'O2'))
    # P4 - T7
    print('\nP4 - T7\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph32A_1, 'P4', 'T7'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph32A_2, 'P4', 'T7'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph32A_3, 'P4', 'T7'))
    # AF3 - CP6
    print('\nAF3 - CP6\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph32A_1, 'AF3', 'CP6'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph32A_2, 'AF3', 'CP6'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph32A_3, 'AF3', 'CP6'))
    # F8 - CP2
    print('\nF8 - CP2\n')
    print('Matriz Lectura: ', TreeNode.floyd_marshall(graph32A_1, 'F8', 'CP2'))
    print('Matriz Memoria: ', TreeNode.floyd_marshall(graph32A_2, 'F8', 'CP2'))
    print('Matriz Operaciones: ', TreeNode.floyd_marshall(graph32A_3, 'F8', 'CP2'))

    print('\n')
    # Parte 3----------------------------------------------------------------------------------------

    # Sujeto 3--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_3_1.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_3_1.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_3_1, kruskal_selected_edges)

    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_3_2.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_3_2.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_3_2, kruskal_selected_edges)
    # -----------------------------------------------------------------------------------------------

    # Sujeto 4--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_4_1.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_4_1.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_4_1, kruskal_selected_edges)

    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_4_2.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_4_2.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_4_2, kruskal_selected_edges)

    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_4_3.prim()
    plotMinSpanningTree(graph8_4_3, kruskal_selected_edges)
    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_4_3, kruskal_selected_edges)
    # -----------------------------------------------------------------------------------------------

    # Sujeto 5--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_5_1.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_5_1.kruskal()
    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_5_1, kruskal_selected_edges)

    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_5_2.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_5_2.kruskal()
    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_5_2, kruskal_selected_edges)

    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_5_3.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_5_3.kruskal()
    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_5_3, kruskal_selected_edges)
    # -----------------------------------------------------------------------------------------------

    # Sujeto 6--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_6_1.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_6_1.kruskal()
    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_6_1, kruskal_selected_edges)

    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_6_2.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_6_2.kruskal()
    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_6_2, kruskal_selected_edges)

    prim_selected_vertices, prim_min_spanning_tree_cost = graph8_6_3.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8_6_3.kruskal()
    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8_6_3, kruskal_selected_edges)
    # -----------------------------------------------------------------------------------------------

    # 32 A graph 1--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph32A_1.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph32A_1.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph32A_1, kruskal_selected_edges)
    # --------------------------------------------------------------------------------------

    # 32 A graph 2--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph32A_2.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph32A_2.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph32A_2, kruskal_selected_edges)
    # --------------------------------------------------------------------------------------

    # 32 A graph 3--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph32A_3.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph32A_3.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph32A_3, kruskal_selected_edges)
    # --------------------------------------------------------------------------------------

    # 32 B graph 1--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph32B_1.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph32B_1.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph32B_1, kruskal_selected_edges)
    # --------------------------------------------------------------------------------------

    # 32 B graph 2--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph32B_2.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph32B_2.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph32B_2, kruskal_selected_edges)
    # --------------------------------------------------------------------------------------

    # 32 B graph 3--------------------------------------------------------------------------------------
    prim_selected_vertices, prim_min_spanning_tree_cost = graph32B_3.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph32B_3.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph32B_3, kruskal_selected_edges)
    # --------------------------------------------------------------------------------------

    # Part 4 ------------------------------------------
    # Convex hull

    # Sujeto 3--------------------------------------------------------------------------------------
    prim_edges, _ = graph8_3_1.prim_edges()
    plotConvexHull(prim_edges, posiciones8)

    prim_edges, _ = graph8_3_2.prim_edges()
    plotConvexHull(prim_edges, posiciones8)

    prim_edges, _ = graph8_3_3.prim_edges()
    plotConvexHull(prim_edges, posiciones8)
    # --------------------------------------------------------------------------------------

    # Sujeto 4--------------------------------------------------------------------------------------
    prim_edges, _ = graph8_4_1.prim_edges()
    plotConvexHull(prim_edges, posiciones8)

    prim_edges, _ = graph8_4_2.prim_edges()
    plotConvexHull(prim_edges, posiciones8)

    prim_edges, _ = graph8_4_3.prim_edges()
    plotConvexHull(prim_edges, posiciones8)
    # --------------------------------------------------------------------------------------

    # Sujeto 5--------------------------------------------------------------------------------------
    prim_edges, _ = graph8_5_1.prim_edges()
    plotConvexHull(prim_edges, posiciones8)

    prim_edges, _ = graph8_5_2.prim_edges()
    plotConvexHull(prim_edges, posiciones8)

    prim_edges, _ = graph8_5_3.prim_edges()
    plotConvexHull(prim_edges, posiciones8)
    # --------------------------------------------------------------------------------------

    # Sujeto 6--------------------------------------------------------------------------------------
    prim_edges, _ = graph8_6_1.prim_edges()
    plotConvexHull(prim_edges, posiciones8)

    prim_edges, _ = graph8_6_2.prim_edges()
    plotConvexHull(prim_edges, posiciones8)

    prim_edges, _ = graph8_6_3.prim_edges()
    plotConvexHull(prim_edges, posiciones8)
    # --------------------------------------------------------------------------------------

    prim_edges, _ = graph32A_1.prim_edges()
    plotConvexHull(prim_edges, posiciones32)


if __name__ == "__main__":
    main()
