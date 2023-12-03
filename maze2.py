
import pygame
import numpy as np
from player import Maze
import torch
from model import MazeSolverNetwork
from torch.distributions import Categorical

import torch.nn.functional as F

# Constants
GAME_HEIGHT = 600
GAME_WIDTH = 600
NUMBER_OF_TILES = 25
SCREEN_HEIGHT = 700
SCREEN_WIDTH = 700
TILE_SIZE = GAME_HEIGHT // NUMBER_OF_TILES

level = [
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
    "X XXXXXXXX          XXXXX",
    "X XXXXXXXX  XXXXXX  XXXXX",
    "X      XXX  XXXXXX  XXXXX",
    "X      XXX  XXX         X",
    "XXXXXX  XX  XXX        XX",
    "XXXXXX  XX  XXXXXX  XXXXX",
    "XXXXXX  XX  XXXXXX  XXXXX",
    "X  XXX      XXXXXXXXXXXXX",
    "X  XXX  XXXXXXXXXXXXXXXXX",
    "X         XXXXXXXXXXXXXXX",
    "X             XXXXXXXXXXX",
    "XXXXXXXXXXX      XXXXX  X",
    "XXXXXXXXXXXXXXX  XXXXX  X",
    "XXX XXXXXXXXXX         X",
    "XXX                     X",
    "XXX         XXXXXXXXXXXXX",
    "XXXXXXXXXX  XXXXXXXXXXXXX",
    "XXXXXXXXXX              X",
    "XX   XXXXX              X",
    "XX   XXXXXXXXXXXXX  XXXXX",
    "XX    XXXXXXXXXXXX  XXXXX",
    "XXP        XXXX          X",
    "XXXX                    X",
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
]


def create_env():
    env = Maze(
        level,
        goal_pos=(23, 20),
        MAZE_HEIGHT=GAME_HEIGHT,
        MAZE_WIDTH=GAME_WIDTH,
        SIZE=NUMBER_OF_TILES,
        hidden_size=64,
    )
    return env


env = create_env()
# Initialize Pygame
pygame.init()

# Create the game window
screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
pygame.display.set_caption("Maze Solver")

surface = pygame.Surface((GAME_HEIGHT, GAME_WIDTH))
clock = pygame.time.Clock()
running = 0

# Get the initial player and goal positions
treasure_pos = env.goal_pos
player_pos = env.state

# Create the neural network
neural_network = MazeSolverNetwork(input_size=2, hidden_size=64, output_size=4)
optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.001)


def train(model, env, learning_rate, hidden_size, num_cycles):
    for cycle in range(1):
        state = env.reset_state()
        done = False
        while not done:
            action_logits = model(env.state_to_tensor(state))
            action_probs = F.softmax(action_logits, dim=-1).squeeze()
            m = Categorical(action_probs)
            #action = m.sample().item()

            action = env.exploratory_policy(state, 0.5)
            next_state, reward, done = env.step(action)

            # Your training logic here
            # You may need to adapt the code based on your specific training requirements
            # Example: model.train_network(state, action, next_state, reward, learning_rate)

            state = next_state


def reset_goal():
    if env.state == env.goal_pos:
        env.reset_goal()
        env.solve()


# Game loop
while running <= 10:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    surface.fill((27, 64, 121))

    for row in range(len(level)):
        for col in range(len(level[row])):
            if level[row][col] == "X":
                pygame.draw.rect(
                    surface,
                    (241, 162, 8),
                    (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )

    pygame.draw.rect(
        surface,
        (255, 51, 102),
        pygame.Rect(
            player_pos[1] * TILE_SIZE,
            player_pos[0] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ).inflate(-TILE_SIZE / 3, -TILE_SIZE / 3),
        border_radius=3,
    )

    pygame.draw.rect(
        surface,
        "green",
        pygame.Rect(
            env.goal_pos[1] * TILE_SIZE,
            env.goal_pos[0] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ).inflate(-TILE_SIZE / 3, -TILE_SIZE / 3),
        border_radius=TILE_SIZE,
    )

    screen.blit(
        surface, ((SCREEN_HEIGHT - GAME_HEIGHT) / 2, (SCREEN_WIDTH - GAME_WIDTH) / 2)
    )
    pygame.display.flip()

    train(neural_network, env, learning_rate=-1.001, hidden_size=32, num_cycles=10)

    action_logits = neural_network(env.state_to_tensor(player_pos))
    action_probs = F.softmax(action_logits, dim=-1).squeeze()
    m = Categorical(action_probs)
    action = m.sample().item()

    if (
        action == 1
        and player_pos[0] > 0
        and (player_pos[0] - 1, player_pos[1]) not in env.walls
    ):
        player_pos = (player_pos[0] - 1, player_pos[1])
        env.state = player_pos
    elif (
        action == 3
        and player_pos[0] < NUMBER_OF_TILES - 1
        and (player_pos[0] + 1, player_pos[1]) not in env.walls
    ):
        player_pos = (player_pos[0] + 1, player_pos[1])
        env.state = player_pos
    elif (
        action == 0
        and player_pos[1] > 0
        and (player_pos[0], player_pos[1] - 1) not in env.walls
    ):
        player_pos = (player_pos[0], player_pos[1] - 1)
        env.state = player_pos
    elif (
        action == 2
        and player_pos[1] < NUMBER_OF_TILES - 1
        and (player_pos[0], player_pos[1] + 1) not in env.walls
    ):
        player_pos = (player_pos[0], player_pos[1] + 1)
        env.state = player_pos
    print(env.state)
    #reset_goal()
    clock.tick(100)
    

pygame.quit()
