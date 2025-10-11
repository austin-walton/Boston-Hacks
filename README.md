# Pac-Man Machine Learning Project 

# Boston Hacks Hackathon October 11 2025
# Austin Walton and August Amhed

A reinforcement learning implementation that trains Pac-Man to collect dots optimally using Deep Q-Networks (DQN) or Q-Learning.

## Project Overview

This project implements a Pac-Man game environment where an AI agent learns to collect all dots in the most efficient way possible. The training process is visualized in real-time, showing the agent's learning progress as it improves its strategy.


## How It Works

### Reinforcement Learning Approach
- **State Space**: Grid representation of the maze (walls, dots, pacman position)
- **Action Space**: 4 directions (up, down, left, right)
- **Reward System**:
  - +10 for collecting a dot
  - +100 bonus for completing the level
  - -0.1 for each step (encourages efficiency)
  - -1 for hitting walls

