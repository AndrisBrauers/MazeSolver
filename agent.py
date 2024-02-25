import matplotlib.pyplot as plt
import numpy as np
from constants import *
import random

# Agent's possible moves
actions = [(-1, 0),
          (1, 0),
          (0, -1),
          (0, 1)]


class Agent:
    def __init__(self, maze, learning_rate=0.35, discount_factor=0.65, exploration_start=0.915, exploration_end=0.15, num_episodes=num_episodes_per_maze, num_mazes=num_of_mazes):
        self.maze = maze
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, len(actions)))
        self.learning_rate = learning_rate                                              # alpha - Determines how new information affects existing Q-values.
        self.discount_factor = discount_factor                                          # gamma - Weighs the importance of future rewards.
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes
        self.num_mazes = num_mazes


    def get_exploration_rate(self, current_episode, current_maze):
        '''
            Calculates the exploration rate for the current episode based on exponential decay from exploration_start to exploration_end.
            This encourages the agent to explore widely early on and gradually shift towards exploiting its knowledge.
            It not only takes into account the number of episode, but also the number of the maze in data set.
        '''
        exploration_rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (((current_episode / self.num_episodes) + (current_maze / self.num_mazes)) / 2 )
        return exploration_rate


    def set_maze(self, maze):
        '''
            Reinitializing the Q-table to match the new maze's dimensions
        '''
        self.maze = maze
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, len(actions)))

    def inspect_one_field_ahead(self, current_position):
        '''
            Checks the immediate surroundings of the agent's current position for obstacles (walls) or free space.
            This function helps in making informed decisions about valid actions.
        '''
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visibility = []
        for dx, dy in directions:
            next_x, next_y = current_position[0] + dx, current_position[1] + dy
            if 0 <= next_x < self.maze.maze_height and 0 <= next_y < self.maze.maze_width:
                visibility.append(self.maze.maze[next_x, next_y] == 0)  # True if free space, False if wall
            else:
                visibility.append(False)  # Treat out-of-bounds as walls
        return visibility

    def get_action(self, state, current_episode, current_maze):
        '''
            Decides which action to take based on the current state and episode number.
            It uses the exploration rate to choose between exploring (taking a random action) and exploiting (choosing the best-known action based on the Q-table).
            It also considers only valid actions that do not lead into walls, based on visibility information.
        '''

        exploration_rate = self.get_exploration_rate(current_episode, current_maze)
        visibility_info = self.inspect_one_field_ahead(state)

        if np.random.rand() < exploration_rate:
            action = np.random.randint(len(actions))  # Random action
        else:
            # Filter out actions leading into a wall based on visibility
            valid_actions = [i for i, visible in enumerate(visibility_info) if visible]
            if valid_actions:
                action = max(valid_actions, key=lambda a: self.q_table[state][a])
            else:
                action = np.argmax(self.q_table[state])

        return action

    def update_q_table(self, state, action, next_state, reward):
        '''
             Updates the Q-values in the Q-table using the Q-Learning formula : Q(S,A) = Q(S,A) + alpha * (reward + gamma * (max(Q(S′,anyActions)) − Q(S,A)))

             Over time and with enough exploration, the Q-values converge to stable values that represent the expected rewards for taking each action in each state,
             allowing the agent to make informed decisions about how to act in various situations to achieve its goals.
        '''
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][np.argmax(self.q_table[next_state])] - self.q_table[state][action])
