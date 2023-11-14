import math
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue, LifoQueue, PriorityQueue

class InitialGraph:
    def __init__(self):
        pass

    def distance2D(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def to3D(x, y, z):
        r = math.sqrt(x**2 + y**2 + z**2)
        t = r / (r + z)
        xPrima = t * x
        yPrima = t * y
        return xPrima, yPrima

    def plot(adjacency_matrix):
        channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

        points3D = np.array([[0, 0.71934, 0.694658], [-0.71934, 0, 0.694658], [0, 0, 1], [0.71934, 0, 0.694658],
                             [0, -0.71934, 0.694658], [-0.587427, -0.808524, -0.0348995], [0, -0.999391, -0.0348995],
                             [0.587427, -0.808524, -0.0348995]])

        r = np.sqrt(points3D[:, 0]**2 + points3D[:, 1]**2 + points3D[:, 2]**2)
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
    def __init__(self, directed=False):
        self.directed = directed
        self.adjacency_list = {}

    def vertices(self):
        return list(self.adjacency_list.keys())

    def add_vertex(self, v):
        if v not in self.adjacency_list:
            self.adjacency_list[v] = []

    def add_edge(self, v1, v2, e=0):
        if v1 not in self.adjacency_list or v2 not in self.adjacency_list:
            print("El vertice no existe")
            return
        if not self.directed and v1 == v2:
            print("Error en los vertices")
            return
        for vertex, weight in self.adjacency_list[v1]:
            if vertex == v2:
                print(f"El vertice ya existe")
                return
        self.adjacency_list[v1].append((v2, e))
        if not self.directed:
            self.adjacency_list[v2].append((v1, e))
    
    def adjacent_vertices(self, v):
        if v not in self.adjacency_list:
            print("El vertice no existe")
            return []
        return self.adjacency_list[v]
    
    def floyd_marshall(self, start_vertex, end_vertex):
        n = len(self.vertices())
        distance = [[float('inf')] * n for _ in range(n)]
        next_node = [[-1] * n for _ in range(n)]

        for i in range(n):
            distance[i][i] = 0
            for edge in self.adjacency_list[self.vertices()[i]]:
                distance[i][self.vertices().index(edge[0])] = edge[1]
                next_node[i][self.vertices().index(edge[0])] = self.vertices().index(edge[0])

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distance[i][k] != float('inf') and distance[k][j] != float('inf'):
                        if distance[i][j] > distance[i][k] + distance[k][j]:
                            distance[i][j] = distance[i][k] + distance[k][j]
                            next_node[i][j] = next_node[i][k]

        start_index = self.vertices().index(start_vertex)
        end_index = self.vertices().index(end_vertex)
        shortest_distance = distance[start_index][end_index]

        return shortest_distance
    
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

    def uniform_cost(graph:WeightedGraph, v0, vg):
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

def main():
    matrizLectura = np.loadtxt('Lectura.txt')
    matrizMemoria = np.loadtxt('Memoria.txt')
    matrizOperaciones = np.loadtxt('Operaciones.txt')

    InitialGraph.plot(matrizLectura)
    InitialGraph.plot(matrizMemoria)
    InitialGraph.plot(matrizOperaciones)

    posiciones = {
    'Fz': np.array([0, 0.71934, 0.694658]),
    'C3': np.array([-0.71934, 0, 0.694658]),
    'Cz': np.array([0, 0, 1]),
    'C4': np.array([0.71934, 0, 0.694658]),
    'Pz': np.array([0, -0.71934, 0.694658]),
    'PO7': np.array([-0.587427, -0.808524, -0.0348995]),
    'Oz': np.array([0, -0.999391, -0.0348995]),
    'PO8': np.array([0.587427, -0.808524, -0.0348995]),
    }

    matriz_conectividad = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ])
    grafo_ponderado = InitialGraph.toWeigthedMatriz(posiciones, matriz_conectividad)

    # Imprimir la matriz ponderada
    print("Matriz Ponderada:")
    print(grafo_ponderado)

if __name__ == "__main__":
    main()

import numpy as np

# Definir las posiciones 3D de los electrod

# Función para calcular la distancia euclidiana entre dos puntos 3D
def distancia_euclidiana(punto1, punto2):
    return np.linalg.norm(punto1 - punto2)