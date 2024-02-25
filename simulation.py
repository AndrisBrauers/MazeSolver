from agent import actions
from maze import Maze
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from constants import *
import random


def finish_episode(agent, maze, current_episode, current_maze, train=True):
    '''
        Simulates one agents attempt to go through the maze and updates the Q-table
    '''
    current_state = maze.start_position     # Initialize start position
    last_action = None                      # Initialize last_action to None
    is_done = False
    episode_reward = 0
    episode_step = 0
    path = [current_state]

    while not is_done:
        action = agent.get_action(current_state, current_episode, current_maze)
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])

        # Initialize reward with step_penalty
        reward = step_penalty

        # Check if the next state is out of bounds
        if next_state[0] < 0 or next_state[0] >= maze.maze_height or next_state[1] < 0 or next_state[1] >= maze.maze_width:
            reward = wall_penalty  # Override reward if hitting a wall
            next_state = current_state
        elif maze.maze[next_state[0]][next_state[1]] == 1:  # Adjusted to use correct indices
            reward = wall_penalty  # Override reward if hitting a wall
            next_state = current_state
        elif next_state == maze.goal_position:
            reward = goal_reward  # Override reward if reaching the goal
            is_done = True

        # Apply rotation penalty if the agent changes direction
        if last_action is not None and last_action != action:
            reward += rotation_penalty

        # Append the valid next state to the path
        if next_state != current_state:
            path.append(next_state)

        episode_reward += reward
        episode_step += 1

        if train:
            agent.update_q_table(current_state, action, next_state, reward)

        # Update current_state and last_action for the next iteration
        current_state = next_state
        last_action = action

    return episode_reward, episode_step, path


def train_agent(agent, maze, current_maze):
    '''
        Makes agent to go through the maze as many times as the episodes count
    '''
    # Lists to store the data for plotting
    episode_rewards = []
    episode_steps = []

    # Loop over the specified number of episodes
    for episode in range(num_episodes_per_maze):
        episode_reward, episode_step, path = finish_episode(agent, maze, episode, current_maze, train=True)

        # Store the episode's cumulative reward and the number of steps taken in their respective lists
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"The average reward is: {average_reward}")
    average_steps = sum(episode_steps) / len(episode_steps)
    print(f"The average steps is: {average_steps}")
    return average_reward, average_steps
