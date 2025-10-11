import pygame
from pacman_ai.game import PacmanGame
import time

def run_manual_game():
    """Run Pac-Man game with manual controls for testing"""
    pygame.init()
    
    game = PacmanGame()
    screen = pygame.display.set_mode((game.screen_width, game.screen_height))
    pygame.display.set_caption("Pac-Man Manual Test")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # Up
                elif event.key == pygame.K_DOWN:
                    action = 1  # Down
                elif event.key == pygame.K_LEFT:
                    action = 2  # Left
                elif event.key == pygame.K_RIGHT:
                    action = 3  # Right
                elif event.key == pygame.K_r:
                    game.reset()
                    continue
                else:
                    continue
                
                state, reward, done = game.step(action)
                print(f"Action: {action}, Reward: {reward}, Done: {done}, Score: {game.score}")
                
                if done:
                    print("Game Over! Press 'R' to reset.")
        
        game.render(screen)
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    print("Manual Pac-Man Game")
    print("Controls: Arrow Keys to move, 'R' to reset")
    run_manual_game()
