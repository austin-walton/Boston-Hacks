#!/usr/bin/env python3
"""
Basic functionality test for Pac-Man ML Project
Tests core components without requiring display
"""

import numpy as np
from pacman_ai.game import PacmanGame, Direction
from pacman_ai.agent import SimpleQAgent

def test_game_basic():
    """Test basic game functionality"""
    print("Testing basic game functionality...")
    
    game = PacmanGame()
    print(f"[OK] Game created: {game.width}x{game.height} grid")
    
    # Test initial state
    state = game.get_state()
    print(f"[OK] Initial state shape: {state.shape}")
    print(f"[OK] Total dots to collect: {game.total_dots}")
    
    # Test movement
    reward1, reward2 = 0, 0
    state1, reward1, done1 = game.step(0)  # Try to move up
    state2, reward2, done2 = game.step(3)  # Try to move right
    
    print(f"[OK] Movement test - Rewards: {reward1}, {reward2}")
    print(f"[OK] Pacman position: {game.pacman_pos}")
    
    return True

def test_agent_basic():
    """Test basic agent functionality"""
    print("\nTesting agent functionality...")
    
    # Create a simple game for testing
    game = PacmanGame()
    state_size = game.width * game.height
    action_size = game.get_action_space()
    
    agent = SimpleQAgent(state_size, action_size)
    print(f"[OK] Agent created with state size: {state_size}, action size: {action_size}")
    
    # Test action selection
    state = game.get_state()
    action = agent.act(state, training=True)
    print(f"[OK] Agent selected action: {action}")
    
    # Test learning
    next_state, reward, done = game.step(action)
    agent.learn(state, action, reward, next_state, done)
    print(f"[OK] Agent learning successful")
    
    return True

def test_training_loop():
    """Test a short training loop"""
    print("\nTesting training loop...")
    
    game = PacmanGame()
    state_size = game.width * game.height
    action_size = game.get_action_space()
    
    agent = SimpleQAgent(state_size, action_size)
    
    # Run a few episodes
    scores = []
    for episode in range(5):
        game.reset()
        state = game.get_state()
        total_reward = 0
        steps = 0
        
        while steps < 50:  # Limit steps for quick test
            action = agent.act(state, training=True)
            next_state, reward, done = game.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        print(f"  Episode {episode + 1}: Score = {total_reward}, Steps = {steps}")
    
    avg_score = np.mean(scores)
    print(f"[OK] Training test complete - Average score: {avg_score:.2f}")
    
    return True

def main():
    print("Pac-Man ML Project - Basic Functionality Test")
    print("=" * 50)
    
    try:
        test_game_basic()
        test_agent_basic()
        test_training_loop()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("[SUCCESS] Project is ready for hackathon demo!")
        print("\nNext steps:")
        print("1. Run 'python quick_demo.py' for a visual demo")
        print("2. Run 'python run_game.py' for manual gameplay")
        print("3. Run 'python panner.py --help' to see training options")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
