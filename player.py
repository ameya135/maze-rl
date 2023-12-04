import numpy as np

from scipy.stats import norm
import random
import torch
import torch.optim as optim
from typing import Tuple
from tqdm.auto import tqdm
from model import MazeSolverNetwork, LinearQTrainer
import torch.nn.functional as F
import networkx as nx


class Maze:
    def __init__(
        self,
        level,
        goal_pos: Tuple[int, int],
        MAZE_HEIGHT=600,
        MAZE_WIDTH=600,
        SIZE=25,
        hidden_size=64,
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
            hidden_size=hidden_size,  # adjust as needed
            output_size=4,  # 4 possible actions
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.trainer = LinearQTrainer(
            model=self.network, learning_rate=0.00000001, gamma=0.99
        )

    def state_to_tensor(self, state):
        return torch.tensor(
            [state[0] / self.number_of_tiles, state[1] / self.number_of_tiles],
            dtype=torch.float32,
        )

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

    def is_collision(self, next_state) -> bool:
        return (next_state) in self.walls

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

        # Check if the next state is the goal
        if next_state == self.goal_pos:
            return 10.0  # A positive reward for reaching the goal

        # Check if the next state results in a collision
        if self.is_collision(next_state):
            return -5.0  # A negative reward for collisions

        # Calculate distance to the goal in the next state
        distance_to_goal = self.calculate_distance_to_goal(next_state)

        # Define a reward function based on distance (you can customize this)
        reward = max(0, 1.0 - 0.01 * distance_to_goal)
        print(f"reward:{reward}")
        return reward

    # def calculate_euclidean_distance(self, state, goal: list[int]):
    #     state_row, state_col = state
    #     goal_row, goal_col = goal

    #     euclidean_distance = (
    #         (state_row - goal_row) ** 2 + (state_col - goal_col) ** 2
    #     ) ** 0.5
    #     return euclidean_distance

    def calculate_distance_to_goal(self, state: Tuple[int, int]):
        # Replace this with your actual distance calculation to the goal
        # This is a placeholder function; you need to implement a meaningful distance calculation.
        goal_row, goal_col = self.goal_pos
        current_row, current_col = state
        distance = abs(goal_row - current_row) + abs(goal_col - current_col)
        return distance

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

    # def exploratory_policy(self, state, epsilon=0.3, wall_penalty=-0.5):
    #     if np.random.rand() < epsilon:
    #         return np.random.randint(4)
    #     else:
    #         av = self.action_values[state]

    #         # Calculate Bayesian distance for each act0.5n
    #         bayesian_distances = []
    #         for action in range(len(av)):
    #             next_state = self._get_next_state(state, action)
    #             if self.is_collision(next_state):
    #                 # Penalize the exploratory policy for hitting the wall
    #                 bayesian_distances.append(wall_penalty)
    #             else:
    #                 bayesian_distance = self.calculate_euclidean_distance(state, action)
    #                 bayesian_distances.append(bayesian_distance)

    #         # Choose the action based on Bayesian distances
    #         action_probs = np.exp(bayesian_distances - np.max(bayesian_distances))
    #         action_probs /= np.sum(action_probs)
    #         chosen_action = np.random.choice(range(len(av)), p=action_probs)

    #         return chosen_action

    def exploratory_policy(self, state, c=8):
        ucb_values = np.zeros(4)
        total_visits = np.sum(self.policy_probs[state]) + 1e-6  # Avoid division by zero

        for action in range(4):
            if self.is_collision(self._get_next_state(state, action)):
                ucb_values[action] = -np.inf
            else:
                action_visits = (
                    np.sum(self.policy_probs[state, action]) + 1e-6
                )  # Avoid division by zero
                exploitation_term = self.action_values[state][action]

                # Calculate Euclidean distance as a factor in exploration term
                euclidean_distance = self.calculate_euclidean_distance(state, action)
                exploration_term = (
                    c * np.sqrt(np.log(total_visits) / action_visits)
                    + euclidean_distance
                )

                ucb_values[action] = exploitation_term + exploration_term

        chosen_action = np.argmax(ucb_values)
        return chosen_action

    def calculate_euclidean_distance(self, state, action):
        # Get the coordinates of the current state and action
        state_row, state_col = state
        action_row, action_col = self._get_next_state(state, action)

        # Calculate Euclidean distance
        euclidean_distance = (
            (state_row - action_row) ** 2 + (state_col - action_col) ** 2
        ) ** 0.5
        return euclidean_distance

    def calculate_distance(self, state, action):
        # Replace this with your actual distance calculation
        # This is a placeholder function; you need to implement a meaningful distance calculation.
        return 0.0

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
        self.trainer.train_step(state, action, reward, next_state, done=False)

    def solve_with_neural_network(self, gamma=0.99, epsilon=0.3, episodes=1):
        init_state = self.state
        for _ in tqdm(range(episodes)):
            done = False
            state = init_state
            while not done:
                state_tensor = self.state_to_tensor(state)
                q_values = self.network(state_tensor)

                action = self.exploratory_policy(state, epsilon, wall_penalty=-1)
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
