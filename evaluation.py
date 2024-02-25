from simulation import finish_episode
import matplotlib.pyplot as plt
from constants import *


def test_agent(agent, maze, current_episode=num_episodes_per_maze, current_maze=num_of_mazes):
    '''
        Testes the agent with the current Q-table values and plots the maze with visited fields.
    '''
    # Simulate the agent's behavior in the maze
    episode_reward, episode_step, path = finish_episode(agent, maze, current_episode, current_maze, train=False)

    print("Number of steps:", episode_step)
    print("Total reward:", episode_reward)

    # Clear the existing plot if any
    if plt.gcf().get_axes():
        plt.cla()

    # Visualize the maze using matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(maze.maze, cmap='gray')

    # Convert path to set of unique positions to ensure each cell is marked only once
    unique_positions = set(path)

    # Mark the agent's unique path with blue '#' symbols
    for position in unique_positions:
        plt.text(position[1], position[0], "x", ha="center", va='center', color='yellow', fontsize=10)

    # Mark the start position (red 'S') and goal position (green 'G') in the maze
    plt.text(maze.start_position[1], maze.start_position[0], 'S', ha='center', va='center', color='red',
             fontsize=20)
    plt.text(maze.goal_position[1], maze.goal_position[0], 'G', ha='center', va='center', color='green',
             fontsize=20)

    # Remove axis ticks and grid lines for a cleaner visualization
    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.show()

    return episode_step, episode_reward
