import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from typing import Tuple
from tqdm.auto import tqdm

# Neural network (Q-network) definition
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Maze class definition
class Maze:
    def __init__(
        self, level, goal_pos: Tuple[int, int], MAZE_HEIGHT=600, MAZE_WIDTH=600, SIZE=25
    ):
        self.goal = (23, 20)
        self.number_of_tiles = SIZE
        self.tile_size = MAZE_HEIGHT // self.number_of_tiles
        self.maze, self.walls = self.create_maze(level)
        self.goal_pos = goal_pos
        self.level = level
        self.state = self.get_init_state(self.level)

        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )
        self.action_values = np.zeros((self.number_of_tiles, self.number_of_tiles, 4))

        # Initialize the Q-network
        self.q_network = QNetwork(2, 4)
        self.q_network_optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def create_maze(self, level):
        maze = []
        walls = []
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == " ":
                    maze.append((row, col))
                elif level[row][col] == "X":
                    walls.append((row, col))
        return maze, walls

    def get_init_state(self, level):
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == "P":
                    return (row, col)
        return None

    def compute_reward(self, state: Tuple[int, int], action: int):
        next_state = self._get_next_state(state, action)
        return -float(state != self.goal_pos)

    def step(self, action):
        next_state = self._get_next_state(self.state, action)
        reward = self.compute_reward(self.state, action)
        done = next_state == self.goal
        self.state = next_state
        return next_state, reward, done

    def simulate_step(self, state, action):
        next_state = self._get_next_state(state, action)
        reward = self.compute_reward(state, action)
        done = next_state == self.goal
        return next_state, reward, done

    def _get_next_state(self, state: Tuple[int, int], action: int):
        if action == 0:
            next_state = (state[0], state[1] - 1)
        elif action == 1:
            next_state = (state[0] - 1, state[1])
        elif action == 2:
            next_state = (state[0], state[1] + 1)
        elif action == 3:
            next_state = (state[0] + 1, state[1])
        else:
            raise ValueError("Action value not supported:", action)
        if (next_state[0], next_state[1]) not in self.walls:
            return next_state
        return state

    def target_policy(self, state):
        q_values = self.q_network(torch.tensor([state]).float())
        return torch.argmax(q_values).item()

    
    def exploratory_policy(self, state, epsilon):
        if state is None:
            # Handle the case where state is None (initial state not found)
            print("Warning: Initial state not found.")
            return np.random.randint(4)

        if np.random.rand() < epsilon:
            return np.random.randint(4)
        else:
            q_values = self.q_network(torch.tensor([state]).float())
            return torch.argmax(q_values).item()


    def sarsa_with_dqn(self, gamma=0.99, alpha=0.2, epsilon=0.3, episodes=1000):
        init_state = self.state
        for _ in tqdm(range(episodes)):
            done = False
            state = init_state
            action = self.exploratory_policy(state, epsilon)
            while not done:
                next_state, reward, done = self.simulate_step(state, action)
                next_action = self.exploratory_policy(next_state, epsilon)

                q_values = self.q_network(torch.tensor([state]).float())
                next_q_values = self.q_network(torch.tensor([next_state]).float())

                qsa = q_values[0][action].item()
                next_qsa = next_q_values[0][next_action].item()
                q_values[0][action] = qsa + alpha * (reward + gamma * next_qsa - qsa)

                self.q_network_optimizer.zero_grad()
                loss = nn.MSELoss()(q_values, self.q_network(torch.tensor([state]).float()))
                loss.backward()
                self.q_network_optimizer.step()

                state = next_state
                action = next_action

    def reset_goal(self):
        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )
        self.goal_pos = random.sample(self.maze, 1)[0]

    def reset_state(self):
        self.state = self.get_init_state(self.level)
        return self.state


# Example usage
level_example = [
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
    "X XXXXXXXX          XXXXX",
    "X XXXXXXXX  XXXXXX  XXXXX",
    # ... (rest of the maze layout)
    "XXXX                    X",
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
]
env = Maze(level_example, goal_pos=(23, 20), MAZE_HEIGHT=600, MAZE_WIDTH=600, SIZE=25)
env.sarsa_with_dqn()
