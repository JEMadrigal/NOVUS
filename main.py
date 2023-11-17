import math
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue, LifoQueue, PriorityQueue

class InitialGraph:
    def __init__(self):
        pass

    def distance2D(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))
    
    def distance3D(punto1, punto2):
        """
        Calcula la distancia euclidiana entre dos puntos en el espacio tridimensional.

        :param punto1: Tupla con las coordenadas (x, y, z) del primer punto.
        :param punto2: Tupla con las coordenadas (x, y, z) del segundo punto.
        :return: La distancia entre los dos puntos.
        """
        x1, y1, z1 = punto1
        x2, y2, z2 = punto2

        distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return distancia

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
    directed = True 
    _vertices = [] # The list of vertices.
    _adjacency_list = [] 

    def __init__(self, directed=False):
        self.directed = directed
        self._vertices = []
        self._adjacency_list = {}  # Cambiar el nombre de la variable

    def vertices(self):
        return list(self._adjacency_list.keys())  # Cambiar el nombre de la variable

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

def main():
    matrizLectura = np.loadtxt('Lectura.txt')
    matrizMemoria = np.loadtxt('Memoria.txt')
    matrizOperaciones = np.loadtxt('Operaciones.txt')

    InitialGraph.plot(matrizLectura)
    #InitialGraph.plot(matrizMemoria)
    #InitialGraph.plot(matrizOperaciones)

    posiciones = {
        'Fz': (0, 0.71934, 0.694658),
        'C3': (-0.71934, 0, 0.694658),
        'Cz': (0, 0, 1),
        'C4': (0.71934, 0, 0.694658),
        'Pz': (0, -0.71934, 0.694658),
        'PO7': (-0.587427, -0.808524, -0.0348995),
        'Oz': (0, -0.999391, -0.0348995),
        'PO8': (0.587427, -0.808524, -0.0348995),
    }

    posName = list(posiciones.keys())  
    indice_a_clave = {str(indice): clave for indice, clave in enumerate(posiciones)}
    
    matrizPonderada = InitialGraph.toWeigthedMatriz(posiciones, matrizLectura)
    graph = WeightedGraph(directed=True)

    for i in range(len(matrizPonderada)):
        graph.add_vertex(posName[i])
        print(i)

    for i in range(len(matrizPonderada)):
        for j in range(len(matrizPonderada[i])):
            if(matrizPonderada[i][j] != 0):
                graph.add_edge(indice_a_clave[str(i)], indice_a_clave[str(j)], matrizPonderada[i][j])

    '''
        BFS:
    '''

    #Fz - PO8
    res = TreeNode.bfs(graph, 'Fz', 'PO8')
    print(res)

    graph.print_graph()

    #C3 - Oz
    res = TreeNode.bfs(graph, 'C3', 'C4')
    print(res)

    #P07 - C4
    res = TreeNode.bfs(graph, 'P07', 'C4')
    print(res)

if __name__ == "__main__":
    main()