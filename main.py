import math
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue, LifoQueue, PriorityQueue
import networkx as nx


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

    def plot8(adjacency_matrix):
        channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

        points3D = np.array([[0, 0.71934, 0.694658], [-0.71934, 0, 0.694658], [0, 0, 1], [0.71934, 0, 0.694658],
                             [0, -0.71934, 0.694658], [-0.587427, -0.808524, -0.0348995], [0, -0.999391, -0.0348995],
                             [0.587427, -0.808524, -0.0348995]])

        r = np.sqrt(points3D[:, 0] ** 2 + points3D[:, 1] ** 2 + points3D[:, 2] ** 2)
        t = r / (r + points3D[:, 2])
        x = r * points3D[:, 0]
        y = r * points3D[:, 1]
        points2D = np.column_stack((x, y))

        circle = plt.Circle((0, 0), 1, color='r', alpha=0.25, fill=False)
        plt.scatter(points2D[:, 0], points2D[:, 1])
        plt.gca().add_patch(circle)

        for i in range(len(points2D)):
            plt.text(points2D[i, 0] - 0.02, points2D[i, 1] + 0.025, channels[i])

        for i in range(len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix[i])):
                if adjacency_matrix[i, j] == 1:
                    plt.plot([points2D[i, 0], points2D[j, 0]], [points2D[i, 1], points2D[j, 1]], 'k-', alpha=0.5)

        plt.axis('equal')
        plt.show()

    def plot32(adjacency_matrix):
        channels = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3',
                    'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1',
                    'Oz', 'O2']

        points3D = [[-0.308829, 0.950477, -0.0348995], [0.308829, 0.950477, -0.0348995],
                    [-0.406247, 0.871199, 0.275637], [0.406247, 0.871199, 0.275637], [-0.808524, 0.587427, -0.0348995],
                    [-0.545007, 0.673028, 0.5], [0, 0.71934, 0.694658], [0.545007, 0.673028, 0.5],
                    [0.808524, 0.587427, -0.0348995], [-0.887888, 0.340828, 0.309017], [-0.37471, 0.37471, 0.848048],
                    [0.37471, 0.37471, 0.848048], [0.887888, 0.340828, 0.309017], [-0.999391, 0, -0.0348995],
                    [-0.71934, 0, 0.694658], [0, 0, 1], [0.71934, 0, 0.694658], [0.999391, 0, -0.0348995],
                    [-0.887888, -0.340828, 0.309017], [-0.37471, -0.37471, 0.848048], [0.37471, -0.37471, 0.848048],
                    [0.887888, -0.340828, 0.309017], [-0.808524, -0.587427, -0.0348995], [-0.545007, -0.673028, 0.5],
                    [0, -0.71934, 0.694658], [0.545007, -0.673028, 0.5], [0.808524, -0.587427, -0.0348995],
                    [-0.406247, -0.871199, 0.275637], [0.406247, -0.871199, 0.275637],
                    [-0.308829, -0.950477, -0.0348995], [0, -0.999391, -0.0348995], [0.308829, -0.950477, -0.0348995]]
        points3D = np.array(points3D)

        r = np.sqrt(points3D[:, 0] ** 2 + points3D[:, 1] ** 2 + points3D[:, 2] ** 2)
        t = r / (r + points3D[:, 2])
        x = r * points3D[:, 0]
        y = r * points3D[:, 1]
        points2D = np.column_stack((x, y))

        circle = plt.Circle((0, 0), 1, color='r', alpha=0.25, fill=False)
        plt.scatter(points2D[:, 0], points2D[:, 1])
        plt.gca().add_patch(circle)

        for i in range(len(points2D)):
            plt.text(points2D[i, 0] - 0.02, points2D[i, 1] + 0.025, channels[i])

        for i in range(len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix[i])):
                if adjacency_matrix[i, j] == 1:
                    plt.plot([points2D[i, 0], points2D[j, 0]], [points2D[i, 1], points2D[j, 1]], 'k-', alpha=0.5)

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
                    distance = InitialGraph.distance3D(posiciones[electrode1], posiciones[electrode2])
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
        return list(self._adjacency_list.keys())  # Cambiar el nombre de la variable

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

    def print_graph(self):
        for vertex in self._adjacency_list:
            for edges in self._adjacency_list[vertex]:
                print(vertex, " -> ", edges[0], " edge weight: ", edges[1])


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

        while not frontier.empty():
            node = frontier.get()
            if node.v == vg:
                return {"Path": node.path(), "Cost": node.c}
            if not node.visited:
                node.visited = True
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
                    frontier.put((cost, TreeNode(node, vertex[0], vertex[1] + node.c)))
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
            distance[i][i] = 0
            for edge in self._adjacency_list[vertices[i]]:
                distance[i][vertices.index(edge[0])] = edge[1]
                next_node[i][vertices.index(edge[0])] = vertices.index(edge[0])

        try:
            end_index = vertices.index(end_vertex)
        except ValueError:
            print(f"The end vertex {end_vertex} is not in the list of vertices.")
            return None

        shortest_distance = distance[vertices.index(start_vertex)][end_index]

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
    """
      # Add nodes
      G.add_node(1, label='A')
      G.add_node(2, label='B')
      G.add_node(3, label='C')
      G.add_node(4, label='D')

      # Add edges
      G.add_edge(1, 2)
      G.add_edge(2, 3)
      G.add_edge(3, 1)
      G.add_edge(1, 4)
      """


def plotMatrices8(matrizLectura, matrizMemoria, matrizOperaciones):
    InitialGraph.plot8(matrizLectura)
    InitialGraph.plot8(matrizMemoria)
    InitialGraph.plot8(matrizOperaciones)

def plotMatrices32(matrizLectura, matrizMemoria, matrizOperaciones):
    InitialGraph.plot32(matrizLectura)
    InitialGraph.plot32(matrizMemoria)
    InitialGraph.plot32(matrizOperaciones)


def graphPaths(graph, initialVertex, finalVertex):
    '''
            BFS:
        '''
    print('\n---BFS---\n')
    # Fz - PO8
    res = TreeNode.bfs(graph, initialVertex, finalVertex)
    print(res)

    '''
        DFS:
    '''
    print('\n---DFS---\n')
    res = TreeNode.dfs(graph, initialVertex, finalVertex)
    print(res)
    '''
        UCS:
    '''
    print('\n---UCS---\n')
    res = TreeNode.ucs(graph, initialVertex, finalVertex)
    print(res)

    '''
        Floyd - Marshall:
    '''
    print('\n---Floyd - Marshall---\n')
    res = TreeNode.floyd_marshall(graph, initialVertex, finalVertex)
    print(res)


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

    matrizPonderada8 = InitialGraph.toWeigthedMatriz(posiciones8, matrizLecturaSujeto3)

    graph8 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada8)):
        for j in range(len(matrizPonderada8[i])):
            if (matrizPonderada8[i][j] != 0):
                graph8.add_vertex(posName8[i])

    for i in range(len(matrizPonderada8)):
        for j in range(len(matrizPonderada8[i])):
            if (matrizPonderada8[i][j] != 0):
                graph8.add_edge(ItoC8[str(i)], ItoC8[str(j)], matrizPonderada8[i][j])

    print('\n\n-----MATRIZ 8 ELECTRODOS-----\n\n')
    # graphPaths(graph8, 'Fz', 'PO8')
    # graphPaths(graph8, 'C3', 'Oz')
    # graphPaths(graph8, 'P07', 'C4')

    '''
        BFS:
    '''
    print('\n---BFS---\n')
    # Fz - PO8
    res = TreeNode.bfs(graph8, 'Fz', 'PO8')
    print(res)
    # C3 - Oz
    res = TreeNode.bfs(graph8, 'C3', 'Oz')
    print(res)
    # P07 - C4
    res = TreeNode.bfs(graph8, 'P07', 'C4')
    print(res)

    '''
        DFS:
    '''
    print('\n---DFS---\n')
    res = TreeNode.dfs(graph8, 'Fz', 'PO8')
    print(res)
    # C3 - Oz
    res = TreeNode.dfs(graph8, 'C3', 'Oz')
    print(res)
    # P07 - C4
    res = TreeNode.dfs(graph8, 'P07', 'C4')
    print(res)

    '''
        UCS:
    '''
    print('\n---UCS---\n')
    # Fz - PO8
    res = TreeNode.ucs(graph8, 'Fz', 'PO8')
    print(res)
    # C3 - Oz
    res = TreeNode.ucs(graph8, 'C3', 'Oz')
    print(res)
    # P07 - C4
    res = TreeNode.ucs(graph8, 'P07', 'C4')
    print(res)

    '''
        Floyd - Marshall:
    '''
    print('\n---Floyd - Marshall---\n')
    # Fz - PO8
    res = TreeNode.floyd_marshall(graph8, 'Fz', 'PO8')
    print(res)
    # C3 - Oz
    res = TreeNode.floyd_marshall(graph8, 'C3', 'Oz')
    print(res)
    # P07 - C4
    res = TreeNode.floyd_marshall(graph8, 'P07', 'C4')
    print(res)

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

    matrizPonderada32 = InitialGraph.toWeigthedMatriz(posiciones32, matrizLectura32A)
    graph32 = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada32)):
        for j in range(len(matrizPonderada32[i])):
            if (matrizPonderada32[i][j] != 0):
                graph32.add_vertex(posName32[i])

    for i in range(len(matrizPonderada32)):
        for j in range(len(matrizPonderada32[i])):
            if (matrizPonderada32[i][j] != 0):
                graph32.add_edge(ItoC32[str(i)], ItoC32[str(j)], matrizPonderada32[i][j])

    print('\n\n-----MATRIZ 32 ELECTRODOS-----\n\n')

    '''
        BFS:
    '''
    print('\n---BFS---\n')
    # F7 - PO4
    res = TreeNode.bfs(graph32, 'F7', 'PO4')
    print(res)
    # CP5 - O2
    res = TreeNode.bfs(graph32, 'CP5', 'O2')
    print(res)
    # P4 - T7
    res = TreeNode.bfs(graph32, 'P4', 'T7')
    print(res)
    # AF3 - CP6
    res = TreeNode.bfs(graph32, 'AF3', 'CP6')
    print(res)
    # F8 - CP2
    res = TreeNode.bfs(graph32, 'F8', 'CP2')
    print(res)

    '''
        DFS:
    '''
    print('\n---DFS---\n')
    # F7 - PO4
    res = TreeNode.dfs(graph32, 'F7', 'PO4')
    print(res)
    # CP5 - O2
    res = TreeNode.dfs(graph32, 'CP5', 'O2')
    print(res)
    # P4 - T7
    res = TreeNode.dfs(graph32, 'P4', 'T7')
    print(res)
    # AF3 - CP6
    res = TreeNode.dfs(graph32, 'AF3', 'CP6')
    print(res)
    # F8 - CP2
    res = TreeNode.dfs(graph32, 'F8', 'CP2')
    print(res)

    '''
        UCS:
    '''
    print('\n---UCS---\n')
    # F7 - PO4
    res = TreeNode.ucs(graph32, 'F7', 'PO4')
    print(res)
    # CP5 - O2
    res = TreeNode.ucs(graph32, 'CP5', 'O2')
    print(res)
    # P4 - T7
    res = TreeNode.ucs(graph32, 'P4', 'T7')
    print(res)
    # AF3 - CP6
    res = TreeNode.ucs(graph32, 'AF3', 'CP6')
    print(res)
    # F8 - CP2
    res = TreeNode.ucs(graph32, 'F8', 'CP2')
    print(res)

    '''
        Floyd - Marshall:
    '''
    print('\n---Floyd - Marshall---\n')
    # F7 - PO4
    res = TreeNode.floyd_marshall(graph32, 'F7', 'PO4')
    print(res)
    # CP5 - O2
    res = TreeNode.floyd_marshall(graph32, 'CP5', 'O2')
    print(res)
    # P4 - T7
    res = TreeNode.floyd_marshall(graph32, 'P4', 'T7')
    print(res)
    # AF3 - CP6
    res = TreeNode.floyd_marshall(graph32, 'AF3', 'CP6')
    print(res)
    # F8 - CP2
    res = TreeNode.floyd_marshall(graph32, 'F8', 'CP2')
    print(res)

    print('\n')
    # Parte 3----------------------------------------------------------------------------------------
    # graph8.print_graph()
    prim_selected_vertices, prim_min_spanning_tree_cost = graph8.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph8.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph8, kruskal_selected_edges)

    # graph32.print_graph()
    prim_selected_vertices, prim_min_spanning_tree_cost = graph32.prim()
    kruskal_selected_edges, kruskal_min_spanning_tree_cost = graph32.kruskal()

    # print("Prim's Minimum Spanning Tree Vertices:", prim_selected_vertices)
    # print("Prim's Minimum Spanning Tree Cost:", prim_min_spanning_tree_cost)
    print("\n")
    print("Kruskal's Minimum Spanning Tree Edges:", kruskal_selected_edges)
    print("Kruskal's Minimum Spanning Tree Cost:", kruskal_min_spanning_tree_cost)
    print("\n")
    plotMinSpanningTree(graph32, kruskal_selected_edges)


if __name__ == "__main__":
    main()
