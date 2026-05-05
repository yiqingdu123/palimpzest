#!/bin/bash
set -e

# Run 4: Local-only — best sampling (gamma-dominant)
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --local-only --use-ollama --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --sampling-alpha 0.15 --sampling-beta 0.15 --sampling-gamma 0.70 \
  --exp-name final-local-only-best-sample --max-workers 4 --progress

# Run 5: Hybrid — simple sampling
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --openai-only --use-ollama --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --exp-name final-hybrid-simple-sample --max-workers 4 --progress

# Run 6: Hybrid — best sampling (gamma-dominant)
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --openai-only --use-ollama --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --sampling-alpha 0.15 --sampling-beta 0.15 --sampling-gamma 0.70 \
  --exp-name final-hybrid-best-sample --max-workers 4 --progress
