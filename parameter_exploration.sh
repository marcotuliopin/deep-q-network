#!/bin/sh

# Cria a pasta de resultados (se ainda não existir)
mkdir -p resultados

echo "Iniciando experimentos com Deep Q-Network..."

# Lista de experimentos com DQN customizado

# Experimento 1: Configuração padrão
python main.py \
  --run_name "default"

# Experimento 2: learning_rate
python main.py \
  --learning-rate 0.0005 \
  --run_name "lr0005"

python main.py \
  --learning-rate 0.005 \
  --run_name "lr005"

# Experimento 3: gamma
python main.py \
  --gamma 0.95 \
  --run_name "gamma095"

python main.py \
  --gamma 0.999 \
  --run_name "gamma0999"

# Experimento 4: epsilon
python main.py \
  --epsilon 0.5 \
  --run_name "eps05"

python main.py \
  --epsilon 0.9 \
  --run_name "eps09"

# Experimento 5: epsilon_decay
python main.py \
  --epsilon-decay 0.99 \
  --run_name "decay099"

python main.py \
  --epsilon-decay 0.98 \
  --run_name "decay098"

# Experimento 6: epsilon_min
python main.py \
  --epsilon-min 0.05 \
  --run_name "epsmin005"

python main.py \
  --epsilon-min 0.001 \
  --run_name "epsmin0001"

# Experimento 7: replay_buffer_capacity
python main.py \
  --replay-buffer-capacity 10000 \
  --run_name "buffer10k"

python main.py \
  --replay-buffer-capacity 100000 \
  --run_name "buffer100k"

# Experimento 8: batch_size
python main.py \
  --batch-size 32 \
  --run_name "batch32"

python main.py \
  --batch-size 128 \
  --run_name "batch128"

# Experimento 9: target_update_frequency
python main.py \
  --target-update-frequency 10 \
  --run_name "target10"

python main.py \
  --target-update-frequency 500 \
  --run_name "target500"

# ================================================
# Experimentos equivalentes usando Stable Baselines3
# ================================================

echo "Iniciando experimentos com Stable Baselines3..."

# SB3 Experimento 1: Padrão
python main.py \
  --use-baselines \
  --run_name "sb3_default"

# SB3 Experimento 2: learning_rate
python main.py \
  --use-baselines \
  --learning-rate 0.0005 \
  --run_name "sb3_lr0005"

python main.py \
  --use-baselines \
  --learning-rate 0.005 \
  --run_name "sb3_lr005"

# SB3 Experimento 3: gamma
python main.py \
  --use-baselines \
  --gamma 0.95 \
  --run_name "sb3_gamma095"

python main.py \
  --use-baselines \
  --gamma 0.999 \
  --run_name "sb3_gamma0999"

# SB3 Experimento 4: buffer size
python main.py \
  --use-baselines \
  --replay-buffer-capacity 10000 \
  --run_name "sb3_buffer10k"

python main.py \
  --use-baselines \
  --replay-buffer-capacity 100000 \
  --run_name "sb3_buffer100k"

# SB3 Experimento 5: batch_size
python main.py \
  --use-baselines \
  --batch-size 32 \
  --run_name "sb3_batch32"

python main.py \
  --use-baselines \
  --batch-size 128 \
  --run_name "sb3_batch128"

# SB3 Experimento 6: target_update_frequency
python main.py \
  --use-baselines \
  --target-update-frequency 10 \
  --run_name "sb3_target10"

python main.py \
  --use-baselines \
  --target-update-frequency 500 \
  --run_name "sb3_target500"
