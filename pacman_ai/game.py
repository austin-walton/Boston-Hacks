import pygame
import numpy as np
from enum import Enum
from typing import Tuple, List

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    NONE = (0, 0)

class PacmanGame:
    def __init__(self, width=28, height=31, cell_size=20):
        """
        Initialize the Pac-Man game environment
        
        Args:
            width: Number of cells horizontally
            height: Number of cells vertically  
            cell_size: Size of each cell in pixels
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size
        
        # Game state
        self.grid = np.zeros((height, width), dtype=int)  # 0=empty, 1=wall, 2=dot
        self.pacman_pos = (14, 23)  # Starting position
        self.pacman_direction = Direction.NONE
        self.score = 0
        self.dots_collected = 0
        self.total_dots = 0
        self.game_over = False
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.WHITE = (255, 255, 255)
        
        # Initialize the maze
        self._create_maze()
        
    def _create_maze(self):
        """Create a simple Pac-Man maze layout"""
        # Create walls around the border
        self.grid[0, :] = 1  # Top wall
        self.grid[-1, :] = 1  # Bottom wall
        self.grid[:, 0] = 1  # Left wall
        self.grid[:, -1] = 1  # Right wall
        
        # Add some internal walls
        # Top section
        self.grid[3, 3:7] = 1
        self.grid[3, 9:13] = 1
        self.grid[3, 15:19] = 1
        self.grid[3, 21:25] = 1
        
        # Middle section
        self.grid[8, 3:25] = 1
        self.grid[8, 12] = 0  # Opening
        
        # Lower section
        self.grid[15, 3:7] = 1
        self.grid[15, 9:13] = 1
        self.grid[15, 15:19] = 1
        self.grid[15, 21:25] = 1
        
        # Add dots everywhere there's no wall
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:
                    self.grid[y, x] = 2
                    self.total_dots += 1
                    
        # Remove dot from starting position
        self.grid[self.pacman_pos[1], self.pacman_pos[0]] = 0
        
    def get_state(self) -> np.ndarray:
        """
        Get current game state as a flattened array
        Returns the grid state for ML training
        """
        state = self.grid.copy()
        # Mark pacman position
        state[self.pacman_pos[1], self.pacman_pos[0]] = 3
        return state.flatten()
    
    def get_action_space(self) -> int:
        """Return number of possible actions (4 directions)"""
        return 4
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take a step in the game
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            (new_state, reward, done)
        """
        if self.game_over:
            return self.get_state(), 0, True
            
        # Convert action to direction
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        if 0 <= action < len(directions):
            self.pacman_direction = directions[action]
        
        # Move pacman
        new_x = self.pacman_pos[0] + self.pacman_direction.value[0]
        new_y = self.pacman_pos[1] + self.pacman_direction.value[1]
        
        # Check bounds and walls
        if (0 <= new_x < self.width and 
            0 <= new_y < self.height and 
            self.grid[new_y, new_x] != 1):  # Not a wall
            
            self.pacman_pos = (new_x, new_y)
            
            # Check if pacman collected a dot
            if self.grid[new_y, new_x] == 2:
                self.grid[new_y, new_x] = 0
                self.score += 10
                self.dots_collected += 1
                reward = 10
            else:
                reward = -0.1  # Small penalty for not collecting dots
                
        else:
            reward = -1  # Penalty for hitting wall or going out of bounds
            
        # Check if game is over (all dots collected)
        done = self.dots_collected >= self.total_dots
        
        if done:
            self.game_over = True
            reward += 100  # Bonus for completing the level
            
        return self.get_state(), reward, done
    
    def reset(self):
        """Reset the game to initial state"""
        self.__init__(self.width, self.height, self.cell_size)
        
    def render(self, screen):
        """Render the game on pygame screen"""
        screen.fill(self.BLACK)
        
        # Draw grid
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                
                if self.grid[y, x] == 1:  # Wall
                    pygame.draw.rect(screen, self.BLUE, rect)
                elif self.grid[y, x] == 2:  # Dot
                    center = (x * self.cell_size + self.cell_size // 2,
                             y * self.cell_size + self.cell_size // 2)
                    pygame.draw.circle(screen, self.WHITE, center, 2)
        
        # Draw pacman
        pacman_rect = pygame.Rect(self.pacman_pos[0] * self.cell_size + 2,
                                 self.pacman_pos[1] * self.cell_size + 2,
                                 self.cell_size - 4, self.cell_size - 4)
        pygame.draw.circle(screen, self.YELLOW, pacman_rect.center, pacman_rect.width // 2)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        screen.blit(score_text, (10, 10))
