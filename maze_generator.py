import random
import numpy as np

class MazeGenerator:
    def __init__(self, dim):
        self.maze = [[0 for _ in range(dim)] for _ in range(dim)]
        self.dimension = dim
        self.stack = []

    def generate_maze(self):
        self.stack.append(Node(0, 0))
        while self.stack:
            next_node = self.stack.pop()
            if self.valid_next_node(next_node):
                self.maze[next_node.y][next_node.x] = 1
                neighbors = self.find_neighbors(next_node)
                self.randomly_add_nodes_to_stack(neighbors)

    def get_raw_maze(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.maze])

    def get_symbolic_maze(self):
        return '\n'.join([' '.join(['*' if cell == 1 else ' ' for cell in row]) for row in self.maze])

    def valid_next_node(self, node):
        num_neighboring_ones = 0
        for y in range(node.y - 1, node.y + 2):
            for x in range(node.x - 1, node.x + 2):
                if self.point_on_grid(x, y) and self.point_not_node(node, x, y) and self.maze[y][x] == 1:
                    num_neighboring_ones += 1
        return num_neighboring_ones < 3 and self.maze[node.y][node.x] != 1

    def randomly_add_nodes_to_stack(self, nodes):
        random.shuffle(nodes)  # Shuffle the list of nodes in place
        for node in nodes:
            self.stack.append(node)

    def find_neighbors(self, node):
        neighbors = []
        for y in range(node.y - 1, node.y + 2):
            for x in range(node.x - 1, node.x + 2):
                if self.point_on_grid(x, y) and self.point_not_corner(node, x, y) and self.point_not_node(node, x, y):
                    neighbors.append(Node(x, y))
        return neighbors

    def point_on_grid(self, x, y):
        return 0 <= x < self.dimension and 0 <= y < self.dimension

    def point_not_corner(self, node, x, y):
        return x == node.x or y == node.y

    def point_not_node(self, node, x, y):
        return not (x == node.x and y == node.y)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def main():
    maze_generator = MazeGenerator(10)
    maze_generator.generate_maze()

    print("RAW MAZE\n" + maze_generator.get_raw_maze())

if __name__ == "__main__":
    main()

