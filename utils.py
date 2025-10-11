import numpy as np
import matplotlib.pyplot as plt
import os

def create_level_file(filename, width=28, height=31):
    """
    Create a level file for Pac-Man
    
    Args:
        filename: Name of the level file
        width: Width of the maze
        height: Height of the maze
    """
    # Create a simple maze layout
    grid = np.zeros((height, width), dtype=int)
    
    # Walls around border
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    
    # Add some internal walls
    # Top section
    grid[3, 3:7] = 1
    grid[3, 9:13] = 1
    grid[3, 15:19] = 1
    grid[3, 21:25] = 1
    
    # Middle section
    grid[8, 3:25] = 1
    grid[8, 12] = 0  # Opening
    
    # Lower section
    grid[15, 3:7] = 1
    grid[15, 9:13] = 1
    grid[15, 15:19] = 1
    grid[15, 21:25] = 1
    
    # Add dots
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0:
                grid[y, x] = 2  # Dot
    
    # Save level
    os.makedirs('pacman_ai/levels', exist_ok=True)
    np.savetxt(f'pacman_ai/levels/{filename}', grid, fmt='%d')
    print(f"Level saved to pacman_ai/levels/{filename}")

def load_level(filename):
    """Load a level file"""
    return np.loadtxt(f'pacman_ai/levels/{filename}', dtype=int)

def visualize_training_progress(scores, avg_scores, epsilons, save_path='training_visualization.png'):
    """Create a visualization of training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Scores over time
    axes[0, 0].plot(scores, alpha=0.3, label='Episode Scores')
    axes[0, 0].plot(avg_scores, label='100-Episode Average')
    axes[0, 0].set_title('Training Scores')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Epsilon decay
    axes[0, 1].plot(epsilons)
    axes[0, 1].set_title('Exploration Rate (Epsilon)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].grid(True)
    
    # Plot 3: Score distribution
    axes[1, 0].hist(scores[-100:], bins=20, alpha=0.7)
    axes[1, 0].set_title('Score Distribution (Last 100 Episodes)')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    # Plot 4: Learning curve (smoothed)
    if len(avg_scores) > 50:
        window = 50
        smoothed = np.convolve(avg_scores, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(smoothed)
        axes[1, 1].set_title('Smoothed Learning Curve')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Score')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training visualization saved to {save_path}")

def calculate_efficiency_metrics(scores, steps_taken):
    """Calculate efficiency metrics for the agent"""
    if len(scores) == 0:
        return {}
    
    metrics = {
        'average_score': np.mean(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'score_std': np.std(scores),
        'average_steps': np.mean(steps_taken),
        'efficiency_ratio': np.mean(scores) / np.mean(steps_taken) if np.mean(steps_taken) > 0 else 0
    }
    
    return metrics

def print_training_summary(episodes, scores, steps_taken, agent_type):
    """Print a summary of training results"""
    metrics = calculate_efficiency_metrics(scores, steps_taken)
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Agent Type: {agent_type}")
    print(f"Total Episodes: {episodes}")
    print(f"Average Score: {metrics['average_score']:.2f}")
    print(f"Max Score: {metrics['max_score']:.2f}")
    print(f"Min Score: {metrics['min_score']:.2f}")
    print(f"Score Std Dev: {metrics['score_std']:.2f}")
    print(f"Average Steps per Episode: {metrics['average_steps']:.2f}")
    print(f"Efficiency Ratio: {metrics['efficiency_ratio']:.4f}")
    print("="*50)

if __name__ == "__main__":
    # Create a sample level
    create_level_file("level1.txt")
    print("Sample level created!")
