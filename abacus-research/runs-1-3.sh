#!/bin/bash
set -e

# Run 1: Remote-only — simple sampling
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --openai-only --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --exp-name final-remote-only-simple-sample --max-workers 4 --progress

# Run 2: Remote-only — best sampling (gamma-dominant)
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --openai-only --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --sampling-alpha 0.15 --sampling-beta 0.15 --sampling-gamma 0.70 \
  --exp-name final-remote-only-best-sample --max-workers 4 --progress

# Run 3: Local-only — simple sampling
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --local-only --use-ollama --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --exp-name final-local-only-simple-sample --max-workers 4 --progress
