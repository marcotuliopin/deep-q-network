# Deep Q-Network (DQN) Implementation

A PyTorch implementation of Deep Q-Network for reinforcement learning tasks.

## Project Structure

```
├── agent.py              # DQN Agent implementation
├── dqn.py                # Deep Q-Network model definition
├── replay_buffer.py      # Experience replay buffer
├── main.py               # Main training/testing script
├── hyperparameters.yml   # Configuration parameters
├── env.yml               # Conda environment specification
```

## Features

- **Deep Q-Network**: Neural network-based Q-function approximation
- **Experience Replay**: Efficient storage and sampling of past experiences
- **Target Network**: Stabilized training with periodic target updates
- **Configurable Hyperparameters**: YAML-based configuration system
- **Model Persistence**: Save and load trained models

## Installation

1. Create the conda environment:
```bash
conda env create -f env.yml
conda activate dqn-env
```

2. Or install dependencies manually:
```bash
pip install torch numpy gymnasium stable-baselines3 pyyaml
```

## Usage

### Training

Run the main training script:
```bash
python main.py
```

The script will:
- Load hyperparameters from [`hyperparameters.yml`](hyperparameters.yml)
- Initialize the environment and agent
- Train the DQN agent
- Save the trained model to [`dqn_model.pth`](dqn_model.pth)

### Configuration

Modify [`hyperparameters.yml`](hyperparameters.yml) to adjust:
- Learning rate
- Batch size
- Replay buffer size
- Network architecture
- Training episodes
- Exploration parameters (epsilon decay)

## Algorithm Details

This implementation includes:

1. **Deep Q-Network**: Neural network that approximates Q-values for state-action pairs
2. **Experience Replay**: Stores transitions and samples random batches for training
3. **Target Network**: Separate network for computing target Q-values, updated periodically
4. **Epsilon-Greedy Exploration**: Balances exploration vs exploitation during training

## File Descriptions

- **[`agent.py`](agent.py)**: Contains the `DQNAgent` class with training and action selection logic
- **[`dqn.py`](dqn.py)**: Defines the neural network architecture for the Q-function
- **[`replay_buffer.py`](replay_buffer.py)**: Implements experience replay buffer for storing and sampling transitions
- **[`main.py`](main.py)**: Main script for training and evaluation

## Model Files

- **[`dqn_model.pth`](dqn_model.pth)**: Saved PyTorch model weights
- **[`dqn_sb3_model.zip`](dqn_sb3_model.zip)**: Stable Baselines3 model for comparison/benchmarking

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Gymnasium (OpenAI Gym)
- PyYAML
- Stable Baselines3 (optional, for comparison)

## Results

The trained model weights are saved in [`dqn_model.pth`](dqn_model.pth) and can be loaded for inference or further training.

## References

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Original DQN paper
