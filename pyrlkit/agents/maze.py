import pygame
import numpy as np
from player import Maze
import torch
from model import MazeSolverNetwork
from torch.distributions import Categorical

from functools import partial
import torch.nn.functional as F

# Constants



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

def train(model, env, learning_rate, hidden_size, num_cycles):
    learning_rate = 0.000001
    hidden_size = 32
    num_cycles = 10


def reset_goal():
    if env.state == env.goal_pos:
        env.reset_goal()
        env.solve()

def reset_goal():
    if env.state == env.goal_pos:
        env.reset_goal()
        env.solve()

def train(model, env, learning_rate, hidden_size, num_cycles):
    learning_rate = 0.000001
    hidden_size = 32
    num_cycles = 10
    
    GAME_HEIGHT = 600
    GAME_WIDTH = 600
    NUMBER_OF_TILES = 25
    SCREEN_HEIGHT = 700
    SCREEN_WIDTH = 700
    TILE_SIZE = GAME_HEIGHT // NUMBER_OF_TILES


# Game loop
    env = create_env()
# Initialize Pygame
    pygame.init()

# Create the game window
    screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
    pygame.display.set_caption("Maze Solver")

    surface = pygame.Surface((GAME_HEIGHT, GAME_WIDTH))
    clock = pygame.time.Clock()
    running = True

# Get the initial player and goal positions
    treasure_pos = env.goal_pos
    player_pos = env.state

# Create the neural network
    neural_network = MazeSolverNetwork(input_size=2, hidden_size=512, output_size=4)
    optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.0001)


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
        "XXXP XXXXXXXXXX         X",
        "XXX                     X",
        "XXX         XXXXXXXXXXXXX",
        "XXXXXXXXXX  XXXXXXXXXXXXX",
        "XXXXXXXXXX              X",
        "XX   XXXXX              X",
        "XX   XXXXXXXXXXXXX  XXXXX",
        "XX    XXXXXXXXXXXX  XXXXX",
        "XX        XXXX          X",
        "XXXX                    X",
        "XXXXXXXXXXXXXXXXXXXXXXXXX",
    ]
    while running:
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

        env.exploratory_policy(env.state)
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

        # env.solve_with_neural_network(gamma=0.5, epsilon=0.3, episodes=10)

        env.train_network(
            player_pos,
            action,
            env._get_next_state(player_pos, action),
            env.compute_reward(player_pos, action),
        )
        optimizer.zero_grad()
        env.optimizer.step()

        print(env.state)
        reset_goal()
        clock.tick(60)
        running += 1

    pygame.quit()
