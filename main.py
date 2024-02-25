from maze import *
from agent import *
from simulation import train_agent
from evaluation import test_agent
import matplotlib.pyplot as plt
import numpy as np
from constants import *


def train_agent_on_multiple_mazes(agent, mazes):
    rewards = []
    steps = []
    for index, maze_data in enumerate(mazes):
        print(index)
        # Reset the maze in the agent to the new maze
        agent.set_maze(maze_data)

        # Train the agent on the current maze for a specified number of episodes
        reward, step = train_agent(agent, maze_data, index)
        rewards.append(reward)
        steps.append(step)

    # Plotting the data after training is completed
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Maze')
    plt.ylabel('Cumulative Reward')
    plt.title('Average reward per Maze')

    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.xlabel('Maze')
    plt.ylabel('Steps Taken')
    plt.ylim(0, 1000)
    plt.title('Average steps per maze')


    plt.tight_layout()
    plt.show()


training_mazes = parse_maze_from_file("./DataSet")
testing_mazes = parse_maze_from_file("./TestSet")

agent = Agent(training_mazes[0])
for maze in testing_mazes:
    test_agent(agent, maze, 1, 1)
train_agent_on_multiple_mazes(agent, training_mazes)
for maze in testing_mazes:
    test_agent(agent, maze, 1, 1)
