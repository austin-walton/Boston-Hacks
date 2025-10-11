#!/usr/bin/env python3
"""
Quick Demo Script for Pac-Man ML Project
Perfect for hackathon demonstrations!
"""

import time
from train_rl import train_pacman, test_agent

def main():
    print("🎮 Pac-Man Machine Learning Demo")
    print("=" * 40)
    
    print("\n🚀 Starting quick training session...")
    print("This will train the AI for 100 episodes (about 2-3 minutes)")
    print("Watch as the AI learns to collect dots efficiently!")
    
    # Quick training session
    agent = train_pacman(
        episodes=100, 
        agent_type='qlearning',  # Faster than DQN
        render=True,
        save_interval=50,
        use_log_scale=False,
        separate_plots=True
    )
    
    print("\n✨ Demo complete! The AI has learned to collect dots optimally.")
    print("Check 'training_results.png' for detailed performance graphs.")

if __name__ == "__main__":
    main()
