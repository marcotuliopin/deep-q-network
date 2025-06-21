#!/bin/sh

# Cria a pasta de resultados (se ainda não existir)
mkdir -p resultados

echo "Iniciando experimentos com Deep Q-Network..."

# Experimento 1: Configuração padrão do YAML
python main.py \
  --run-name "default"

# Experimento 2: gamma = 0.9
python main.py \
  --gamma 0.9 \
  --run-name "gamma09"

# Experimento 3: batch_size = 32
python main.py \
  --batch-size 32 \
  --run-name "batch32"

# Experimento 4: epsilon_min = 0.1
python main.py \
  --epsilon-min 0.1 \
  --run-name "epsmin01"

# Experimento 5: combinação de parâmetros
python main.py \
  --gamma 0.95 \
  --batch-size 128 \
  --epsilon-min 0.05 \
  --run-name "combo1"

# Experimento 6: Stable Baselines3
python main.py \
  --use-baselines \
  --run-name "sb3"
