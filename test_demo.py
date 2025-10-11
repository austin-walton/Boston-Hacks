"""
Quick test of the demo functionality
"""

from pacman_ai.agent import DQNAgent
from pacman_ai.game import PacmanGame
import pygame
import time

def test_demo():
    print("Creating a dummy agent for demo test...")
    
    # Create a simple agent (won't be trained, just for demo)
    agent = DQNAgent(state_size=868, action_size=4)
    
    print("Testing demo window...")
    
    # Initialize game
    game = PacmanGame()
    
    # Disable exploration for demo
    agent.epsilon = 0
    
    # Initialize pygame for demo window
    pygame.init()
    screen = pygame.display.set_mode((game.screen_width, game.screen_height))
    pygame.display.set_caption("Pac-Man AI Demo")
    clock = pygame.time.Clock()
    
    game.reset()
    state = game.get_state()
    total_reward = 0
    steps = 0
    start_time = time.time()
    
    print("Starting AI demo...")
    
    done = False
    while not done and steps < 1000:  # Allow more steps for demo
        action = agent.act(state, training=False)
        state, reward, done = game.step(action)
        total_reward += reward
        steps += 1
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Render the game
        game.render(screen)
        
        # Add some demo text
        font = pygame.font.Font(None, 24)
        demo_text = font.render("AI Demo", True, (255, 255, 0))
        screen.blit(demo_text, (10, game.screen_height - 30))
        
        pygame.display.flip()
        clock.tick(20)  # Good demo speed
    
    end_time = time.time()
    demo_duration = end_time - start_time
    
    print(f"Demo Complete!")
    print(f"Final Score: {total_reward}")
    print(f"Steps Taken: {steps}")
    print(f"Demo Duration: {demo_duration:.1f} seconds")
    print(f"Efficiency: {total_reward/steps:.2f} points per step")
    
    # Keep window open for a moment
    time.sleep(2)
    pygame.quit()

if __name__ == "__main__":
    test_demo()
