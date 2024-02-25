import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from constants import *


def expand_cell(cell):
    if cell in [0, 1]:
        return [[cell] * 3 for _ in range(3)]
    else:
        return [[0] * 3 for _ in range(3)]


def process_maze(maze_lines):
    start_position = goal_position = None
    expanded_maze = []

    for i, row in enumerate(maze_lines):
        expanded_rows = [[], [], []]
        for j, cell in enumerate(row.split()):
            expanded_cell = expand_cell(int(cell) if cell.isdigit() else cell)
            for k in range(3):
                expanded_rows[k].extend(expanded_cell[k])

            if cell == 'S':
                start_position = (i * 3 + 1, j * 3 + 1)
            elif cell == 'G':
                goal_position = (i * 3 + 1, j * 3 + 1)

        expanded_maze.extend(expanded_rows)

    return np.array(expanded_maze), start_position, goal_position


def parse_maze_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    mazes = []
    current_maze = []
    for line in lines:
        if line.endswith('.'):
            if current_maze:
                mazes.append(current_maze)
                current_maze = []
        else:
            current_maze.append(line)
    if current_maze:
        mazes.append(current_maze)

    maze_objects = []
    for maze in mazes:
        maze_array, start_pos, goal_pos = process_maze(maze)
        maze_objects.append(Maze(maze_array, start_pos, goal_pos))

    return maze_objects


class Maze:
    def __init__(self, maze, start_position, goal_position):
        self.maze = maze
        self.maze_height = maze.shape[0]
        self.maze_width = maze.shape[1]
        self.start_position = start_position
        self.goal_position = goal_position

    def show_maze(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.maze, cmap='gray')
        plt.text(self.start_position[1], self.start_position[0], 'S', ha='center', va='center', color='red', fontsize=20)
        plt.text(self.goal_position[1], self.goal_position[0], 'G', ha='center', va='center', color='green', fontsize=20)
        plt.xticks([]), plt.yticks([])
        plt.show()
