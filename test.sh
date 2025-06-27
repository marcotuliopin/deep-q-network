#!/bin/sh

# Cria a pasta de resultados (se ainda não existir)
mkdir -p resultados

echo "Iniciando experimentos com Deep Q-Network..."

# Lista de experimentos com DQN customizado

python main.py \
  --run_name "best_combo" \
  --learning-rate 0.0005 \
  --epsilon 0.5 \
  --epsilon-min 0.05 \
  --replay-buffer-capacity 100000 \
  --batch-size 32 \

python main.py \
  --run_name "best_combo" \
  --learning-rate 0.0005 \
  --epsilon 0.5 \
  --epsilon-min 0.05 \
  --replay-buffer-capacity 100000 \
  --batch-size 32 \
  --test \
  --load-model-path ./resultados/dqn_model_best_combo.pth