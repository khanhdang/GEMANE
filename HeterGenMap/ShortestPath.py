import math
class ShortestPath():

	def __init__(self, input_vertices):
		self.vertices = input_vertices
		self.graph = [[0 for column in range(input_vertices)]
					for row in range(input_vertices)]

	def minDist(self, dist, S):

		min = math.inf
		min_idx = -1
		for v in range(self.vertices):
			if dist[v] < min and S[v] == False:
				min = dist[v]
				min_idx = v

		return min_idx

	def find_path(self, src):

		dist = [math.inf] * self.vertices
		dist[src] = 0
		S = [False] * self.vertices

		for cout in range(self.S):

			u = self.minDist(dist, S)

			S[u] = True

			for v in range(self.vertices):
				if (self.graph[u][v] > 0 and
				S[v] == False and
				dist[v] > dist[u] + self.graph[u][v]):
					dist[v] = dist[u] + self.graph[u][v]

		return dist