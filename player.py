import numpy as np
import random
import torch
import torch.optim as optim
from typing import Tuple
from tqdm.auto import tqdm
from model import MazeSolverNetwork
import torch.nn.functional as F

class Maze:
    def __init__(
        self, level, goal_pos: Tuple[int, int], MAZE_HEIGHT=600, MAZE_WIDTH=600, SIZE=25
    ):
        """
        Maze class to represent a simple maze environment.

        Args:
            level (List[str]): A list of strings representing the maze layout.
            goal_pos (Tuple[int, int]): The goal position (row, col) in the maze.
            MAZE_HEIGHT (int, optional): Height of the maze in pixels. Defaults to 600.
            MAZE_WIDTH (int, optional): Width of the maze in pixels. Defaults to 600.
            SIZE (int, optional): Number of tiles per row/column in the maze. Defaults to 25.
        """
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

        self.network = MazeSolverNetwork(
            input_size=2,  # input size (row, col)
            hidden_size=64,  # adjust as needed
            output_size=4,  # 4 possible actions
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)


    def state_to_tensor(self, state):
        return torch.tensor([state[0] / self.number_of_tiles, state[1] / self.number_of_tiles], dtype=torch.float32)

    def create_maze(self, level):
        """
        Create a list of positions of walls and maze.

        Args:
            level (List[str]): A list of strings representing the maze layout.

        Returns:
            List[Tuple[int, int]]: A list of maze positions (row, col) that are not walls.
        """
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
        """
        Get the initial state (player's position) in the maze.

        Args:
            level (List[str]): A list of strings representing the maze layout.

        Returns:
            Tuple[int, int]: The initial state (row, col) in the maze.
        """
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == "P":
                    return (row, col)

    def compute_reward(self, state: Tuple[int, int], action: int):
        """
        Compute the reward for taking an action from the current state.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            float: The reward for taking the action from the current state.
        """
        next_state = self._get_next_state(state, action)
        return -float(state != self.goal_pos)

    def step(self, action):
        """
        Take a step in the maze environment.

        Args:
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            Tuple[Tuple[int, int], float, bool]: Tuple containing the next state, reward, and done flag.
        """
        next_state = self._get_next_state(self.state, action)
        reward = self.compute_reward(self.state, action)
        done = next_state == self.goal
        self.state = next_state
        return next_state, reward, done

    def simulate_step(self, state, action):
        """
        Simulate a step in the maze environment.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            Tuple[Tuple[int, int], float, bool]: Tuple containing the next state, reward, and done flag.
        """
        next_state = self._get_next_state(state, action)
        reward = self.compute_reward(state, action)
        done = next_state == self.goal
        return next_state, reward, done

    def _get_next_state(self, state: Tuple[int, int], action: int):
        """
        Get the next state based on the current state and action.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            Tuple[int, int]: The next state (row, col) after taking the action.
        """
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

    def solve(self, gamma=0.99, theta=1e-6):
        """
        Solve the maze environment using the value iteration algorithm.

        Args:
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            theta (float, optional): Threshold for convergence. Defaults to 1e-6.
        """
        delta = float("inf")

        while delta > theta:
            delta = 0
            for row in range(self.number_of_tiles):
                for col in range(self.number_of_tiles):
                    if (row, col) not in self.walls:
                        old_value = self.state_values[row, col]
                        q_max = float("-inf")

                        for action in range(4):
                            next_state, reward, done = self.simulate_step(
                                (row, col), action
                            )
                            value = reward + gamma * self.state_values[next_state]
                            if value > q_max:
                                q_max = value
                                action_probs = np.zeros(shape=(4))
                                action_probs[action] = 1

                        self.state_values[row, col] = q_max
                        self.policy_probs[row, col] = action_probs

                        delta = max(delta, abs(old_value - self.state_values[row, col]))

    def target_policy(self, state):
        av = self.action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))

    def exploratory_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)

        else:
            av = self.action_values[state]
            return np.random.choice(np.flatnonzero(av == av.max()))

    def sarsa(self, gamma=0.99, alpha=0.2, epsilon=0.3, episodes=1000):
        init_state = self.state
        self.action_values = np.zeros((self.number_of_tiles, self.number_of_tiles, 4))
        for _ in tqdm(range(episodes)):
            done = False
            state = init_state
            action = self.exploratory_policy(state, epsilon)
            while not done:
                next_state, reward, done = self.simulate_step(state, action)
                next_action = self.exploratory_policy(next_state, epsilon)
                qsa = self.action_values[state][action]
                next_qsa = self.action_values[next_state][next_action]
                self.action_values[state][action] = qsa + alpha * (
                    reward + gamma * next_qsa - qsa
                )
                state = next_state
                action = next_action

    def reset_goal(self):
        """Reset the goal position"""
        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )
        self.goal_pos = random.sample(self.maze, 1)[0]

    def reset_state(self):
        """Reset the maze environment."""
        self.state = self.get_init_state(self.level)

        return self.state


    def train_network(self, state, action, next_state, reward, gamma=0.99):
        self.optimizer.zero_grad()

        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)

        q_values = self.network(state_tensor)
        q_value = q_values[action]

        next_q_values = self.network(next_state_tensor)
        next_q_value = torch.max(next_q_values)

        target = reward + gamma * next_q_value
        loss = F.mse_loss(q_value, target)

        loss.backward()
        self.optimizer.step()


    def solve_with_neural_network(self, gamma=0.99, epsilon=0.3, episodes=1000):
        init_state = self.state
        for _ in tqdm(range(episodes)):
            done = False
            state = init_state
            while not done:
                state_tensor = self.state_to_tensor(state)
                q_values = self.network(state_tensor)
                
                action = self.exploratory_policy(state, epsilon)
                next_state, reward, done = self.simulate_step(state, action)

                next_state_tensor = self.state_to_tensor(next_state)
                next_q_values = self.network(next_state_tensor)
                next_q_value = torch.max(next_q_values)

                q_value = q_values[action]
                target = reward + gamma * next_q_value

                loss = F.mse_loss(q_value, target)
                loss.backward()
                self.optimizer.step()

                state = next_state




class LinearQTrainer:
    """Using the model to implement training, this method will be customisable in the future"""

    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
