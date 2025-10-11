import pygame
import numpy as np
import matplotlib.pyplot as plt
from pacman_ai.game import PacmanGame
from pacman_ai.agent import DQNAgent, SimpleQAgent
import time

def train_pacman(episodes=1000, agent_type='dqn', render=True, save_interval=100, use_log_scale=False, separate_plots=False):
    """
    Train Pac-Man agent with visual display
    
    Args:
        episodes: Number of training episodes
        agent_type: 'dqn' or 'qlearning'
        render: Whether to show the game during training
        save_interval: Save model every N episodes
        use_log_scale: If True, plot log-transformed scores for clearer trends
        separate_plots: If True, create separate plots for scores and averages
    """
    
    # Initialize game and agent
    game = PacmanGame()
    state_size = game.width * game.height
    action_size = game.get_action_space()
    
    if agent_type == 'dqn':
        agent = DQNAgent(state_size, action_size)
        print("Using Deep Q-Network agent")
    else:
        agent = SimpleQAgent(state_size, action_size)
        print("Using Q-Learning agent")
    
    # Initialize pygame for rendering
    if render:
        pygame.init()
        screen = pygame.display.set_mode((game.screen_width, game.screen_height))
        pygame.display.set_caption("Pac-Man ML Training")
        clock = pygame.time.Clock()
    
    # Training metrics
    scores = []
    avg_scores = []
    epsilons = []
    
    print(f"Starting training for {episodes} episodes...")
    
    for episode in range(episodes):
        game.reset()
        state = game.get_state()
        total_reward = 0
        steps = 0
        max_steps = 500  # Prevent infinite episodes
        
        done = False
        while not done and steps < max_steps:
            # Choose action
            action = agent.act(state, training=True)
            
            # Take step
            next_state, reward, done = game.step(action)
            total_reward += reward
            
            # Store experience and learn
            if agent_type == 'dqn':
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
            else:
                agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            steps += 1
            
            # Render game
            if render and episode % 10 == 0:  # Render every 10th episode
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                game.render(screen)
                pygame.display.flip()
                clock.tick(30)  # 30 FPS
        
        # Record metrics
        scores.append(total_reward)
        epsilons.append(agent.epsilon)
        
        # Calculate running average
        if episode >= 99:
            avg_scores.append(np.mean(scores[-100:]))
        else:
            avg_scores.append(np.mean(scores))
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {total_reward:.1f}, "
                  f"Avg Score: {avg_scores[-1]:.1f}, "
                  f"Steps: {steps}")
        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            if agent_type == 'dqn':
                agent.q_network.save(f'models/pacman_dqn_episode_{episode}.h5')
            print(f"Model saved at episode {episode}")
    
    # Update target network for DQN
    if agent_type == 'dqn':
        agent.update_target_network()
    
    # Clean up pygame BEFORE showing graphs
    if render:
        pygame.quit()
        pygame.display.quit()  # Force close any display windows
    
    # Plot training results - ONE window with TWO subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    episodes_range = range(len(scores))
    
    # Plot 1: Individual Episode Scores
    ax1.plot(episodes_range, scores, alpha=0.7, color='lightblue', linewidth=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Individual Episode Scores')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Scores
    ax2.plot(episodes_range, avg_scores, linewidth=2, color='darkblue')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Score')
    ax2.set_title('Average Score (100-episode rolling)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save final model
    if agent_type == 'dqn':
        agent.q_network.save('models/pacman_dqn_final.h5')
    
    print("Training completed!")
    return agent

def test_agent(agent, agent_type='dqn', episodes=5, render=False):
    """Test trained agent without exploration"""
    game = PacmanGame()
    
    if agent_type == 'dqn':
        agent.epsilon = 0  # No exploration during testing
    else:
        agent.epsilon = 0
    
    if render:
        # Initialize pygame for rendering
        pygame.init()
        screen = pygame.display.set_mode((game.screen_width, game.screen_height))
        pygame.display.set_caption("Trained Pac-Man AI Demo")
        clock = pygame.time.Clock()
    
    for episode in range(episodes):
        game.reset()
        state = game.get_state()
        total_reward = 0
        steps = 0
        
        done = False
        while not done and steps < 500:
            action = agent.act(state, training=False)
            state, reward, done = game.step(action)
            total_reward += reward
            steps += 1
            
            # Render if requested
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                game.render(screen)
                pygame.display.flip()
                clock.tick(15)  # Good speed for viewing
        
        print(f"Test Episode {episode + 1}: Score = {total_reward:.1f}, Steps = {steps}")
        if render and episode < episodes - 1:  # Don't pause after last episode
            time.sleep(0.5)  # Short pause between episodes
    
    if render:
        pygame.quit()


if __name__ == "__main__":
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    # Train the agent
    agent = train_pacman(episodes=500, agent_type='dqn', render=True)