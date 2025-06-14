#!/bin/sh

# This script is an example of how to run the Deep Q-Network (DQN) training with a custom configuration.

# This script will modify the parameters gamma, batch_size, and epsilon_min, and maintain the default values for other parameters.
# The default values are located in the hyperparameters.yml file.
python main.py \
    --gamma 0.9 \
    --batch_size 32 \
    --epsilon_min 0.1 \