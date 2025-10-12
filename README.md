# 🎮 Pac-Man AI: Reinforcement Learning Project

**Created by Austin Walton (apwalton@bu.edu) and August Ahmed (aulmer@bu.edu)**

**Boston Hacks Hackathon - Boston University, October 11, 2025**

## What We Built

We created an AI agent that learns to play Pac-Man using reinforcement learning. The AI starts by making random moves but gradually learns the optimal strategy to collect dots and complete levels efficiently.

## How It Works

- **The AI Agent**: Uses Q-Learning algorithm to learn from trial and error
- **The Game Environment**: Custom-built Pac-Man game with a classic maze layout
- **Learning Process**: The AI plays hundreds of games, and it's given rewards for collecting dots and penalties for inefficient moves
- **Visual Training**: You can watch the AI learn in real-time as it improves its strategy

## What You'll See

1. **Random Behavior**: Initially, the AI moves randomly and performs poorly
2. **Learning Phase**: Over time, it discovers better strategies and collects more dots
4. **Performance Graphs**: Visual charts showing the AI's improvement over time

## Technical Approach

The game environment was built from scratch using Pygame, creating a custom Pac-Man maze with walls, dots, and power pellets. Each game state is represented as a grid where the AI tracks its position and the remaining collectible items. We implemented Q-Learning, a reinforcement learning algorithm where the AI builds a "Q-table" that stores the expected rewards for each possible action in each game state. The reward system encourages efficient dot collection while penalizing wasted moves and revisiting cleared areas. During training, the AI starts with completely random actions but gradually learns optimal paths through trial and error, updating its Q-table after each game to improve future performance.

