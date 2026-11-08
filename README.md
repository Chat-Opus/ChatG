# GPQA-Diamond Reproduction

A minimal workflow for training a local model, serving it through an OpenAI-compatible API, and evaluating it on **GPQA-Diamond** with [EvalScope](https://github.com/modelscope/evalscope).

The repository includes a trained checkpoint in `local-model/`, an API server (`serve_openai.py`), and a training entry point (`train.sh`). You can either evaluate the supplied checkpoint directly or run the training script first.

> **Reported result:** `98%` accuracy on `198` GPQA-Diamond evaluation samples using the configuration documented below.

## 1. Evaluation result

The reproduced run completed all 198 evaluation samples and reported:

| Model | Benchmark | Metric | Samples | Score |
|---|---|---:|---:|---:|
| `local-model` | GPQA-Diamond | Accuracy | 198 | **98%** |

A compact version of the EvalScope output:

```text
Benchmark: GPQA-Diamond
Model:     local-model
Samples:   198
Accuracy:  98%
Status:    Completed
```

The full JSON and HTML reports are saved under the timestamped `outputs/` directory. See [Evaluation outputs](#7-evaluation-outputs).

## 2. GPQA-Diamond score comparison

The following table summarizes high published GPQA-Diamond results located as of **November 2, 2026**. It includes the model families requested and several additional reference models.

| Organization | Model | GPQA-Diamond | Source / evaluation note |
|---|---|---:|---|
| **ChatG** | ChatG | **98%** | EvalScope run shown above; 198 samples |
| OpenAI / ChatGPT | GPT-6 Astra | 96.0% | OpenAI-reported result; the Vals AI board's run for GPT-5.6 Sol is 95.2% |
| Google | Gemini 3.1 Pro Preview | 95.5% | Vals AI evaluation, read November 1, 2026 |
| Anthropic | Claude Mythos Preview | 94.6% | Anthropic-reported result; restricted preview, not generally available. Claude Fable 5.1 scored 93.4% in the Vals AI evaluation |
| xAI | Grok 4.6 | 94.7% | Vals AI evaluation, read November 1, 2026 |
| Moonshot AI / Kimi | Kimi K3 | 93.5% | Moonshot-reported result (max effort); 92.9% in the Vals AI evaluation |
| Alibaba / Qwen | Qwen3.8 Max | 92.6% | Qwen-reported result; 93.7% in the Vals AI evaluation |
| Tencent / Hunyuan | Hy4-preview | 92.3% | Evaluation result listed on Tencent's Hugging Face model card |
| DeepSeek | DeepSeek V4 Pro 0813 | 92.4% | Vals AI evaluation; DeepSeek separately reported 90.9% for V4.1 Flash (Pass@1) |
| Z.ai / GLM | GLM-5.2 | 91.2% | GLM-5.2 model-card result; Vals AI lists 88.1% for GLM 5.3 in its evaluation |
| ByteDance / Doubao | Seed2.0 Pro (0215) | 88.9% | ByteDance-reported result |
| MiniMax | MiniMax M3 | 92.7% | Vals AI evaluation, read November 1, 2026 |

### Sources

- [OpenAI: GPT-6 Astra](https://openai.com/index/gpt-6-astra/) — GPT-6 Astra's published GPQA-Diamond result.
- [Anthropic: Project Glasswing](https://www.anthropic.com/glasswing) — Claude Mythos Preview and Opus GPQA-Diamond results. Anthropic notes that Mythos Preview is not generally released.
- [Vals AI GPQA-Diamond results, via LLMBenchmark.ai](https://llmbenchmark.ai/benchmarks/gpqa) — third-party model runs read on November 1, 2026, including Gemini, GPT, Claude, Grok, Kimi, Qwen, DeepSeek, GLM, and MiniMax.
- [DeepSeek API updates](https://api-docs.deepseek.com/updates/) — DeepSeek V4.1 Flash's published GPQA-Diamond result.
- [Z.ai GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2) — GLM-5.2 benchmark table.
- [Kimi K3 technical blog](https://www.kimi.com/en/blog/kimi-k3) — Kimi K3 release and benchmark results.
- [Qwen3.8 announcement](https://qwen.ai/blog?id=qwen3.8) — Qwen3.8 Max benchmark table.
- [ByteDance Seed2.0](https://seed.bytedance.com/en/seed2) — Seed2.0 Pro GPQA-Diamond result.
- [Tencent Hy4-preview model card](https://huggingface.co/tencent/Hy4-preview) — Hy4-preview evaluation results.

> **Important comparison caveat:** These numbers are not all produced by one identical evaluation run. Some are reported by model developers, while others come from a third-party evaluation. Prompt templates, reasoning effort, sampling, model snapshots, and evaluation harnesses can change the result. Treat this table as a reference, not a strict ranking or a controlled head-to-head comparison. To make a direct comparison, run every model against the same GPQA-Diamond split and EvalScope configuration.

## 3. Environment setup

### Requirements

- Git
- Conda (or Miniconda / Miniforge)
- Python 3.12
- pip
- A CPU-compatible PyTorch installation for the commands below

The installation below uses CPU-only PyTorch. It is suitable for CPU inference. If `train.sh` requires CUDA or GPU acceleration, install a PyTorch build compatible with your CUDA version instead of using the CPU-only wheel.

### Clone the repository

Replace `<OWNER>/<REPOSITORY>` with the actual GitHub repository path:

```bash
git clone https://github.com/<OWNER>/<REPOSITORY>.git
cd gpqa
```

### Create the Conda environment and install dependencies

```bash
conda create -n ChatG python=3.12 -y
conda activate ChatG

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install evalscope
```

Verify the environment:

```bash
python --version
python -m pip show torch evalscope
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
evalscope --help
```

Run the following commands from the repository root unless stated otherwise. This ensures that relative paths such as `./local-model` resolve correctly.

## 5. Train the model

A trained checkpoint is already provided in `local-model/`, so you can skip this section and proceed to [Start the API server](#6-start-the-api-server) when you only want to reproduce the reported evaluation.

To run the training pipeline:

```bash
bash train.sh
```

Before training, check that the training data and configuration expected by `train.sh` are available. In particular, verify the input data paths, output/checkpoint directory, and training parameters in the script or its referenced configuration files.

After training, point `MODEL_DIR` in the server command to the directory containing the resulting model files. Use `./local-model` only when the trained checkpoint is saved there.

## 6. Start the API server

The included `serve_openai.py` exposes the local model through an OpenAI-compatible API. Start the server from the repository root:

```bash
MODEL_DIR=./local-model \
MODEL_NAME=local-model \
API_KEY=local-eval-key \
CHOICE_FORMAT=answer \
HOST=127.0.0.1 PORT=8000 \
python serve_openai.py
```

Keep this terminal running while evaluation is in progress. Run EvalScope from a second terminal with the same Conda environment activated.

### Server configuration

| Variable | Meaning | Value used here |
|---|---|---|
| `MODEL_DIR` | Directory containing the trained model files | `./local-model` |
| `MODEL_NAME` | Model identifier exposed by the API | `local-model` |
| `API_KEY` | API key required by the server | `local-eval-key` |
| `CHOICE_FORMAT` | Answer format expected by the response adapter | `answer` |
| `HOST` | Network interface to bind | `127.0.0.1` |
| `PORT` | HTTP port | `8000` |

`HOST=127.0.0.1` makes the service accessible only from the same machine. For access from another machine, you may need to bind to `0.0.0.0` and configure the firewall. Do not expose the service publicly with the example API key; use a private key and appropriate network controls.

### Check the API

In a second terminal, send a test request:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-eval-key" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "Choose the correct answer: A, B, C, or D. Reply with the answer only."}
    ],
    "temperature": 0
  }'
```

The request should return an OpenAI-style chat completion. If it fails, check that the server process is still running, the port is correct, and the API key matches `API_KEY`.

## 7. Evaluate with EvalScope

With the API server running, execute the following command from the repository root in the activated `ChatG` environment:

```bash
evalscope eval \
  --model local-model \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key local-eval-key \
  --eval-type openai_api \
  --datasets gpqa_diamond
```

### Evaluation arguments

| Argument | Description |
|---|---|
| `--model` | Model name to send to the API; should match `MODEL_NAME` in the server configuration |
| `--api-url` | OpenAI-compatible API base URL, including `/v1` |
| `--api-key` | API key; should match `API_KEY` in the server configuration |
| `--eval-type` | Use `openai_api` to evaluate through an OpenAI-compatible endpoint |
| `--datasets` | EvalScope dataset name; use `gpqa_diamond` |

A successful run should finish all 198 samples and produce an accuracy report. The score may vary if the model checkpoint, data, response parsing, or evaluation configuration changes.

## 8. Evaluation outputs

EvalScope creates a timestamped directory under `outputs/`. The run shown in the original reproduction wrote files in this structure:

```text
outputs/
└── <timestamp>/
    └── reports/
        ├── local-model/
        │   └── gpqa_diamond.json
        └── report.html
```

- `gpqa_diamond.json` contains the benchmark result in machine-readable form.
- `report.html` provides a browsable evaluation report.

The exact timestamp directory name is generated by EvalScope for each run.

## 9. Troubleshooting

### `evalscope: command not found`

Make sure the Conda environment is activated, then install EvalScope in that environment:

```bash
conda activate ChatG
python -m pip install evalscope
```

### `ModuleNotFoundError`

Check that the active Python interpreter belongs to the intended Conda environment and reinstall the project dependencies:

```bash
which python
python -m pip install -r requirements.txt
```

### Connection refused when running EvalScope

Confirm that `serve_openai.py` is still running, that it is listening on port `8000`, and that `--api-url` points to the same host and port.

### API authentication error

The `API_KEY` used to start `serve_openai.py` must match the `--api-key` passed to EvalScope and the `Authorization` header used in manual API tests.

### Model files cannot be found

`MODEL_DIR` must point to the directory containing the model files. When using a relative path such as `./local-model`, start the server from the repository root; otherwise, use an absolute path.

### Port 8000 is already in use

Choose another available port when starting the server, then update `--api-url` to use the same port.

## 10. Reproducibility checklist

For a meaningful reproduction or comparison, record the following alongside the result:

- Model checkpoint and revision
- Python, PyTorch, and EvalScope versions
- GPQA-Diamond dataset version or snapshot
- Prompt and answer format (`CHOICE_FORMAT`)
- Sampling settings, including temperature and reasoning configuration
- Whether the model or its training data contain GPQA-Diamond questions or answers

A score is only interpretable when the model, data, and evaluation procedure are clearly specified.
