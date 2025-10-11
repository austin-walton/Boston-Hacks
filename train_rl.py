import pygame
import numpy as np
import matplotlib.pyplot as plt
from pacman_ai.game import PacmanGame
from pacman_ai.agent import DQNAgent, SimpleQAgent
import time

def train_pacman(episodes=1000, agent_type='dqn', render=True, save_interval=100):
    """
    Train Pac-Man agent with visual display
    
    Args:
        episodes: Number of training episodes
        agent_type: 'dqn' or 'qlearning'
        render: Whether to show the game during training
        save_interval: Save model every N episodes
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
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Steps: {steps}")
        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            if agent_type == 'dqn':
                agent.q_network.save(f'models/pacman_dqn_episode_{episode}.h5')
            print(f"Model saved at episode {episode}")
    
    # Update target network for DQN
    if agent_type == 'dqn':
        agent.update_target_network()
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(['Score', '100-Episode Average'])
    
    plt.subplot(1, 3, 2)
    plt.plot(epsilons)
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.subplot(1, 3, 3)
    plt.plot(avg_scores)
    plt.title('Average Score (Last 100 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    # Save final model
    if agent_type == 'dqn':
        agent.q_network.save('models/pacman_dqn_final.h5')
    
    print("Training completed!")
    return agent

def test_agent(agent, agent_type='dqn', episodes=5):
    """Test trained agent without exploration"""
    game = PacmanGame()
    
    if agent_type == 'dqn':
        agent.epsilon = 0  # No exploration during testing
    else:
        agent.epsilon = 0
    
    # Initialize pygame for rendering
    pygame.init()
    screen = pygame.display.set_mode((game.screen_width, game.screen_height))
    pygame.display.set_caption("Pac-Man ML Testing")
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
            
            # Render
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            game.render(screen)
            pygame.display.flip()
            clock.tick(10)  # Slower for viewing
        
        print(f"Test Episode {episode + 1}: Score = {total_reward}, Steps = {steps}")
        time.sleep(1)  # Pause between episodes
    
    pygame.quit()

if __name__ == "__main__":
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    # Train the agent
    agent = train_pacman(episodes=500, agent_type='dqn', render=True)
    
    # Test the trained agent
    test_agent(agent, agent_type='dqn', episodes=3)
