import pygame
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    NONE = (0, 0)

class PacmanGame:
    def __init__(self, width=28, height=31, cell_size=20, layout_path: Optional[str] = None):
        """
        Initialize the Pac-Man game environment
        
        Args:
            width: Number of cells horizontally (used if no layout is provided)
            height: Number of cells vertically (used if no layout is provided)
            cell_size: Size of each cell in pixels
            layout_path: Optional path to a layout file to load
        """
        self.cell_size = cell_size
        base_path = Path(__file__).resolve().parent
        layout = Path(layout_path) if layout_path is not None else base_path / "levels" / "original_pacman.txt"
        self.layout_path = layout if layout.is_absolute() else base_path / layout

        # Game state
        self.grid = np.zeros((height, width), dtype=int)  # Placeholder until layout loads
        self.pacman_pos = (width // 2, height // 2)  # Will be overwritten by layout parsing
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
        self.RED = (255, 0, 0)
        
        # Initialize the maze from the layout file
        self._create_maze()
        
    def _create_maze(self):
        """Load the Pac-Man maze layout from file."""
        if not self.layout_path.exists():
            raise FileNotFoundError(f"Layout file not found: {self.layout_path}")

        with self.layout_path.open("r", encoding="utf-8") as layout_file:
            lines = [line.rstrip("\n") for line in layout_file if line.rstrip("\n")]

        if not lines:
            raise ValueError(f"Layout file '{self.layout_path}' is empty.")

        row_width = max(len(row) for row in lines)
        # Layouts may have ragged edges; pad with spaces so the grid stays rectangular.
        if any(len(row) != row_width for row in lines):
            lines = [row.ljust(row_width) for row in lines]

        self.height = len(lines)
        self.width = row_width
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.total_dots = 0
        self.dots_collected = 0
        self.pacman_pos = None

        for y, row in enumerate(lines):
            for x, tile in enumerate(row):
                if tile == "%":
                    self.grid[y, x] = 1  # Wall
                elif tile == "=":
                    self.grid[y, x] = 1  # Ghost house door (treat as wall for now)
                elif tile == ".":
                    self.grid[y, x] = 2  # Dot
                    self.total_dots += 1
                elif tile in {"o", "O", "/"}:
                    self.grid[y, x] = 4  # Power pellet
                    self.total_dots += 1
                elif tile == "1":
                    self.grid[y, x] = 1  # Numeric wall fallback
                elif tile == "2":
                    self.grid[y, x] = 2
                    self.total_dots += 1
                elif tile in {" ", "0"}:
                    self.grid[y, x] = 0  # Empty space
                elif tile in {"P", "p", "3"}:
                    self.grid[y, x] = 0
                    self.pacman_pos = (x, y)
                else:
                    # Treat all other characters (e.g., ghost spawns) as empty
                    self.grid[y, x] = 0

        if self.pacman_pos is None:
            # Fallback to center if layout omitted Pac-Man start
            fallback_pos = (self.width // 2, self.height // 2)
            self.pacman_pos = fallback_pos
            if self.grid[fallback_pos[1], fallback_pos[0]] in (2, 4):
                self.grid[fallback_pos[1], fallback_pos[0]] = 0
                self.total_dots = max(0, self.total_dots - 1)

        self.pacman_start = self.pacman_pos
        self.screen_width = self.width * self.cell_size
        self.screen_height = self.height * self.cell_size
        self.visited_positions = {self.pacman_pos}
        
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

        # Wrap horizontally through tunnels
        if new_x < 0:
            new_x = self.width - 1
        elif new_x >= self.width:
            new_x = 0

        # Check bounds and walls
        if (0 <= new_y < self.height and
            self.grid[new_y, new_x] != 1):

            self.pacman_pos = (new_x, new_y)
            tile_key = self.pacman_pos
            
            # Check if pacman collected a dot
            tile_value = self.grid[new_y, new_x]
            if tile_value in (2, 4):
                self.grid[new_y, new_x] = 0
                self.dots_collected += 1
                reward = 10
            elif tile_key in self.visited_positions:
                reward = -0.1  # Penalty for revisiting a cleared spot
            else:
                reward = 0  # Neutral move on an unvisited empty space

            self.visited_positions.add(tile_key)
            self.score += reward
                
        else:
            reward = -0.1  # Small penalty for hitting wall or going out of bounds
            self.score += reward
            
        # Check if game is over (all dots collected)
        done = self.dots_collected >= self.total_dots
        
        if done:
            self.game_over = True
            reward += 100  # Bonus for completing the level
            self.score += 100
            
        return self.get_state(), reward, done
    
    def reset(self):
        """Reset the game to initial state"""
        self.score = 0
        self.dots_collected = 0
        self.game_over = False
        self.pacman_direction = Direction.NONE
        self._create_maze()
        
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
                elif self.grid[y, x] == 4:  # Power pellet
                    center = (x * self.cell_size + self.cell_size // 2,
                             y * self.cell_size + self.cell_size // 2)
                    radius = max(4, self.cell_size // 3)
                    pygame.draw.circle(screen, self.RED, center, radius)
        
        # Draw pacman
        pacman_rect = pygame.Rect(self.pacman_pos[0] * self.cell_size + 2,
                                 self.pacman_pos[1] * self.cell_size + 2,
                                 self.cell_size - 4, self.cell_size - 4)
        pygame.draw.circle(screen, self.YELLOW, pacman_rect.center, pacman_rect.width // 2)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score:.1f}", True, self.WHITE)
        screen.blit(score_text, (10, 10))
