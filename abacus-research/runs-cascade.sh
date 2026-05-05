#!/bin/bash
set -e

# Run 7: Hybrid + Cascade — simple sampling
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --openai-only --use-ollama --cascade --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --exp-name final-cascade-simple-sample --max-workers 4 --progress

# Run 8: Hybrid + Cascade — best sampling (gamma-dominant)
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --openai-only --use-ollama --cascade --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --sampling-alpha 0.15 --sampling-beta 0.15 --sampling-gamma 0.70 \
  --exp-name final-cascade-best-sample --max-workers 4 --progress
