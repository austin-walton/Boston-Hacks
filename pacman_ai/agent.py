import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        Deep Q-Network Agent for Pac-Man
        
        Args:
            state_size: Size of the state space (grid flattened)
            action_size: Number of possible actions (4)
            learning_rate: Learning rate for the neural network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Neural network parameters
        self.memory = []  # Experience replay buffer
        self.memory_size = 10000
        self.batch_size = 32
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # Build the neural network
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
    def _build_model(self):
        """Build the Deep Q-Network"""
        model = tf.keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current game state
            training: Whether we're in training mode (affects epsilon)
            
        Returns:
            Action to take
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SimpleQAgent:
    """Simpler Q-learning agent for faster training"""
    
    def __init__(self, state_size, action_size, learning_rate=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Simple Q-table (for smaller state spaces)
        self.q_table = {}
    
    def _state_to_key(self, state):
        """Convert state to hashable key"""
        return tuple(state)
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        state_key = self._state_to_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
