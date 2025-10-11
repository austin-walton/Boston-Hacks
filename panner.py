"""
Pac-Man ML Project - Main Demo Script
Run this to start the training and demo
"""

import argparse
from train_rl import train_pacman, test_agent

def main():
    parser = argparse.ArgumentParser(description='Pac-Man ML Training')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--agent', choices=['dqn', 'qlearning'], default='dqn', help='Agent type')
    parser.add_argument('--no-render', action='store_true', help='Disable visual rendering during training')
    parser.add_argument('--test-only', action='store_true', help='Only run tests (requires trained model)')
    
    args = parser.parse_args()
    
    if args.test_only:
        print("Testing mode - loading pre-trained agent...")
        # This would load a saved model
        print("Note: You need to train a model first using --episodes")
        return
    
    print(f"Starting Pac-Man ML Training")
    print(f"Episodes: {args.episodes}")
    print(f"Agent: {args.agent}")
    print(f"Rendering: {not args.no_render}")
    
    # Train the agent
    agent = train_pacman(
        episodes=args.episodes, 
        agent_type=args.agent, 
        render=not args.no_render
    )
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_agent(agent, agent_type=args.agent, episodes=3)

if __name__ == "__main__":
    main()
