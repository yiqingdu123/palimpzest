# BioDEX Demo — Setup and Usage Guide

This guide covers how to run `biodex-demo.py`, what each flag does, and how to apply the changes we made (diversity-aware sampling, local model support) to other Palimpzest demos.

---

## Prerequisites

### 1. Create and activate a virtual environment
From the repo root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Install and start Ollama (for local model runs)
Download and install Ollama from https://ollama.com, open the app, then pull the model:
```bash
ollama pull llama3.2:3b
```

### 3. Install additional dependencies
```bash
pip install chromadb datasets
```

### 4. Set API keys
```bash
export OPENAI_API_KEY=your_key_here
```

### 5. Build the ChromaDB embeddings index
After setting your OpenAI API key, generate the MedDRA index:
```bash
python "abacus-research/helper-scripts/biodex-gen-index.py"
```
This creates a `.chroma-biodex/` directory in the repo root containing the MedDRA vector index used by the retrieval step.

### 6. (Optional) Manually start the Ollama server
Ollama usually starts automatically when the app is open. If not:
```bash
ollama serve  # starts server at http://localhost:11434
```

---

## Running the Demo

All commands should be run from the **repo root** (not from `abacus-research/`).

### Remote-only (fastest, costs API credits)
```bash
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --openai-only --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --exp-name my-run --max-workers 4 --progress
```

### Local-only (free, slow without GPU)
```bash
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --local-only --use-ollama --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --exp-name my-local-run --max-workers 4 --progress
```

### Hybrid (Ollama + OpenAI, no cascade)
```bash
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --openai-only --use-ollama --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --exp-name my-hybrid-run --max-workers 4 --progress
```

### Hybrid with cascade
Add `--cascade` to any hybrid run. The cascade operator runs the local model first and escalates to remote only if the local model returns empty/None output fields.
```bash
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --openai-only --use-ollama --cascade \
  --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 100 --val-examples 25 --plan-examples 250 \
  --sentinel-execution-strategy mab --policy maxquality \
  --exp-name my-cascade-run --max-workers 4 --progress
```

### Fast test run (small scale)
Use small values for quick iteration:
```bash
PYTHONUNBUFFERED=1 python "abacus-research/biodex-demo.py" \
  --local-only --use-ollama --ollama-model "ollama/llama3.2:3b" --ollama-api-base "http://localhost:11434" \
  --seed 42 --sample-budget 20 --val-examples 5 --plan-examples 10 \
  --sentinel-execution-strategy mab --policy maxquality \
  --exp-name quick-test --max-workers 4 --progress
```

---

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--openai-only` | false | Use only GPT-4o and GPT-4o-mini |
| `--local-only` | false | Use only the Ollama model (requires `--use-ollama`) |
| `--use-ollama` | false | Make the Ollama model available to the optimizer |
| `--cascade` | false | Enable cascade: local model first, remote as fallback |
| `--ollama-model` | `ollama/llama3.2:3b` | Ollama model identifier |
| `--ollama-api-base` | `http://localhost:11434` | Ollama server URL |
| `--plan-examples` | 250 | Number of test records to run the final plan on |
| `--val-examples` | 25 | Number of training records for the MAB optimizer |
| `--sample-budget` | 100 | Total LLM calls the MAB optimizer can make |
| `--policy` | `maxquality` | Optimization policy: `maxquality`, `mincost`, `minlatency` |
| `--sentinel-execution-strategy` | `mab` | How the optimizer samples plans: `mab` or `random` |
| `--seed` | 42 | RNG seed for reproducibility |
| `--max-workers` | 8 | Parallel workers for execution |
| `--exp-name` | auto-generated | Name prefix for output files in `opt-profiling-data/` |
| `--progress` | false | Show live progress bars |
| `--sampling-alpha` | 0.7 | Weight for embedding diversity in sample reordering |
| `--sampling-beta` | 0.3 | Weight for document length diversity in sample reordering |
| `--sampling-gamma` | 0.0 | Weight for extraction difficulty in sample reordering |

> **Note:** `--quality` is only used with `--policy mincostatfixedquality` or `minlatencyatfixedquality`. It has no effect with `maxquality`.

---

## Output Files

After a run, `opt-profiling-data/` will contain four files named by `--exp-name`:

| File | Contents |
|------|----------|
| `{exp-name}-output.csv` | Full output dataframe (pmid, reactions, labels, ranked labels) |
| `{exp-name}-records.json` | Per-record predictions in JSON |
| `{exp-name}-profiling.json` | Full execution statistics |
| `{exp-name}-metrics.json` | Summary: `rp@5`, cost, time, and the selected `plan_str` |

To inspect which plan the optimizer chose:
```bash
python3 -c "
import json
d = json.load(open('opt-profiling-data/{exp-name}-metrics.json'))
print(d['plan_str'])
"
```

---

## Diversity-Aware Sampling

We added three flags to control how the MAB optimizer's training examples are reordered before optimization begins:

- **`--sampling-alpha`**: weight for semantic embedding diversity (farthest-point traversal over `text-embedding-3-small` embeddings)
- **`--sampling-beta`**: weight for document length distribution
- **`--sampling-gamma`**: weight for extraction difficulty (entity density proxy)

The three weights should sum to 1.0. Our experiments found `--sampling-alpha 0.15 --sampling-beta 0.15 --sampling-gamma 0.70` (gamma-dominant) gives the best results on BioDEX.

**Simple sampling** (default): alpha=0.7, beta=0.3, gamma=0.0

**Best sampling** (our result):
```bash
--sampling-alpha 0.15 --sampling-beta 0.15 --sampling-gamma 0.70
```

Embeddings for the training records are cached in `opt-profiling-data/sampling-embeddings-cache/` so subsequent runs with the same examples do not re-embed.

---

## Applying These Changes to Other Demos

The following changes were made to `biodex-demo.py` that can be ported to other demos like `cuad-demo.py` or `mmqa-demo.py`.

### 1. Add diversity-aware sampling flags

Add these arguments to the `argparse` block:
```python
parser.add_argument("--sampling-alpha", default=0.7, type=float,
    help="Weight for embedding diversity in embedding-hybrid sampling.")
parser.add_argument("--sampling-beta", default=0.3, type=float,
    help="Weight for length signal in embedding-hybrid sampling.")
parser.add_argument("--sampling-gamma", default=0.0, type=float,
    help="Weight for difficulty proxy in embedding-hybrid sampling.")
parser.add_argument("--sampling-embedding-batch-size", default=128, type=int,
    help="Embedding batch size for embedding-hybrid sampling.")
```

### 2. Add local model (Ollama) flags

```python
parser.add_argument("--use-ollama", default=False, action="store_true",
    help="Include a local Ollama model in available models.")
parser.add_argument("--cascade", default=False, action="store_true",
    help="Enable cascade: use local model as primary and remote as fallback.")
parser.add_argument("--ollama-model", default="ollama/llama3.2:3b", type=str,
    help="Local Ollama model identifier.")
parser.add_argument("--ollama-api-base", default="http://localhost:11434", type=str,
    help="Base URL for local Ollama server.")
parser.add_argument("--local-only", default=False, action="store_true",
    help="Use only local model(s) for LLM operators")
```

### 3. Wire the flags into QueryProcessorConfig

Pass the sampling and Ollama args into the config:
```python
sampling_cache_dir = os.path.join("opt-profiling-data", "sampling-embeddings-cache")
os.makedirs(sampling_cache_dir, exist_ok=True)

ollama_models_arg = [args.ollama_model] if args.use_ollama else None

config = pz.QueryProcessorConfig(
    # ... your existing config fields ...
    sampling_embedding_provider="openai",
    sampling_embedding_model="openai/text-embedding-3-small",
    sampling_alpha=args.sampling_alpha,
    sampling_beta=args.sampling_beta,
    sampling_gamma=args.sampling_gamma,
    sampling_embedding_batch_size=args.sampling_embedding_batch_size,
    sampling_cache_dir=sampling_cache_dir,
    use_ollama=args.use_ollama,
    ollama_models=ollama_models_arg,
    ollama_api_base=args.ollama_api_base,
    allow_cascade=args.cascade,
)
```
