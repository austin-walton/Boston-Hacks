# Pac-Man Machine Learning Project 🎮🤖

A reinforcement learning implementation that trains Pac-Man to collect dots optimally using Deep Q-Networks (DQN) or Q-Learning.

## 🎯 Project Overview

This project implements a Pac-Man game environment where an AI agent learns to collect all dots in the most efficient way possible. The training process is visualized in real-time, showing the agent's learning progress as it improves its strategy.

### Features
- **Real-time Training Visualization**: Watch the AI learn as it plays
- **Multiple Algorithms**: Both Deep Q-Network and Q-Learning implementations
- **Customizable Environment**: Easy to modify maze layouts and game parameters
- **Performance Metrics**: Detailed training statistics and visualizations
- **Interactive Testing**: Manual control mode for testing game mechanics

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Boston-Hacks
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Train the AI agent**
   ```bash
   python panner.py --episodes 500 --agent dqn
   ```

2. **Test manually (optional)**
   ```bash
   python run_game.py
   ```

3. **Generate sample level**
   ```bash
   python utils.py
   ```

## 🎮 Controls

### Manual Game (run_game.py)
- **Arrow Keys**: Move Pac-Man
- **R**: Reset the game
- **ESC/Close Window**: Quit

## 📊 Training Options

### Command Line Arguments
```bash
python panner.py [OPTIONS]

Options:
  --episodes EPISODES    Number of training episodes (default: 500)
  --agent {dqn,qlearning}  Agent type (default: dqn)
  --no-render           Disable visual rendering during training
  --test-only           Only run tests (requires trained model)
```

### Examples
```bash
# Quick training with Q-Learning (faster)
python panner.py --episodes 200 --agent qlearning

# Full DQN training with visualization
python panner.py --episodes 1000 --agent dqn

# Fast training without graphics
python panner.py --episodes 100 --no-render
```

## 🧠 How It Works

### Reinforcement Learning Approach
- **State Space**: Grid representation of the maze (walls, dots, pacman position)
- **Action Space**: 4 directions (up, down, left, right)
- **Reward System**:
  - `+10` for collecting a dot
  - `+100` bonus for completing the level
  - `-0.1` for each step (encourages efficiency)
  - `-1` for hitting walls

### Algorithm Options

1. **Deep Q-Network (DQN)**
   - Neural network with experience replay
   - Target network for stability
   - Better for complex environments
   - Slower but more powerful

2. **Q-Learning**
   - Tabular method with Q-table
   - Faster training for smaller state spaces
   - Good for understanding the basics

## 📁 Project Structure

```
Boston-Hacks/
├── pacman_ai/           # Main package
│   ├── __init__.py     # Package initialization
│   ├── game.py         # Game engine and environment
│   ├── agent.py        # ML agents (DQN, Q-Learning)
│   ├── assets/         # Game assets (images, sounds)
│   └── levels/         # Maze level files
├── train_rl.py         # Training script
├── run_game.py         # Manual game testing
├── panner.py           # Main demo script
├── utils.py            # Utility functions
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎯 Hackathon Demo

### For Your Presentation

1. **Start with manual mode** to show the game mechanics
2. **Run a quick training session** (100-200 episodes) to show the learning process
3. **Show the final trained agent** performing optimally
4. **Display training graphs** to show improvement over time

### Demo Commands
```bash
# 1. Show manual gameplay
python run_game.py

# 2. Quick training demo (5-10 minutes)
python panner.py --episodes 200 --agent qlearning

# 3. Show trained agent
# (The script automatically runs tests after training)
```

## 📈 Expected Results

- **Early episodes**: Random movement, low scores
- **Middle episodes**: Learning basic navigation, improving scores
- **Later episodes**: Efficient dot collection, high scores
- **Final performance**: Near-optimal pathfinding

## 🔧 Customization

### Modifying the Maze
Edit the `_create_maze()` function in `pacman_ai/game.py` to change the layout.

### Adjusting Rewards
Modify the reward values in the `step()` method of `PacmanGame` class.

### Training Parameters
Adjust hyperparameters in the agent classes:
- Learning rate
- Epsilon decay
- Network architecture (for DQN)

## 🐛 Troubleshooting

### Common Issues
1. **Pygame display errors**: Make sure you have a display available
2. **TensorFlow warnings**: Normal, can be ignored
3. **Slow training**: Reduce episodes or use Q-Learning for faster results

### Performance Tips
- Use `--no-render` for faster training
- Start with fewer episodes to test setup
- Use Q-Learning for quicker results on smaller mazes

## 🏆 Hackathon Tips

### What Makes This Impressive
- **Visual Learning**: Real-time training visualization
- **Multiple Algorithms**: Shows understanding of different RL approaches
- **Complete Implementation**: Full game engine + ML integration
- **Performance Metrics**: Professional-level training analysis

### Presentation Flow
1. **Problem**: "How can AI learn to play Pac-Man optimally?"
2. **Solution**: Show the training process
3. **Results**: Demonstrate the learned behavior
4. **Technical Details**: Explain the algorithms briefly

## 📚 Technical Details

### Dependencies
- `pygame`: Game engine and visualization
- `tensorflow`: Deep learning framework
- `numpy`: Numerical computations
- `matplotlib`: Training visualizations

### Key Files
- `pacman_ai/game.py`: Core game logic and environment
- `pacman_ai/agent.py`: ML algorithms implementation
- `train_rl.py`: Training loop and visualization
- `utils.py`: Helper functions and metrics

---

**Good luck with your hackathon! 🚀**

*This project demonstrates the power of reinforcement learning in game AI and provides a solid foundation for understanding how AI agents can learn to solve complex navigation and optimization problems.*  
