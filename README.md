# Frozen Lake Environment

This project implements a variety of reinforcement learning (RL) algorithms to find optimal policies in the Frozen Lake environment. The environment is a grid-based simulation where an agent navigates from a starting point to a goal, avoiding obstacles and slipping on icy surfaces.

## Features

- **Frozen Lake Environment**: Supports customizable lake configurations with variable slip probabilities.
- **Reinforcement Learning Algorithms**:
  - Policy Iteration
  - Value Iteration
  - SARSA
  - Q-Learning
  - Linear Function Approximation
  - Deep Q-Networks (DQN)

## Key Components

- **Environment**:
  - Fully customizable lake grids with absorbing states.
  - Actions: Move Up, Left, Down, Right.
  - Handles transitions, rewards, and rendering.

- **Algorithms**:
  - Tabular model-based methods (e.g., Policy Iteration, Value Iteration).
  - Model-free methods (e.g., SARSA, Q-Learning).
  - Advanced methods with linear and deep function approximation.

- **Interactive Play**: Test the environment interactively by controlling the agent manually.

## Requirements

- Python 3.7+
- Libraries: `numpy`, `torch`, `matplotlib`

## Usage

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd frozen-lake-environment

 
