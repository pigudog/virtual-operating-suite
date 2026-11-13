# Virtual operating suite

OR-GPT workflow simulation and coordinating agents.

## Install and run the example

Place [OR-GPT](https://github.com/pigudog/OR-GPT) beside this repository and create its `orgpt` environment following its README. Run from this directory:

```bash
conda activate orgpt
python -m pip install --no-deps -e ../OR-GPT
python -m pip install --no-deps -e .
CUDA_VISIBLE_DEVICES=0 make smoke
```

The example uses the companion repository's synthetic data and runs on CUDA without an LLM server.

## Download Qwen weights

Use a separate environment for vLLM:

```bash
conda create -n orgpt-vllm python=3.11 -y
conda activate orgpt-vllm
python -m pip install vllm==0.10.2 transformers==4.55.2 huggingface_hub==0.34.4
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir ../models/Qwen3-30B-A3B-Instruct-2507
```

Weights: [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507).

## Start vLLM

Example settings for one 80-GB NVIDIA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ../models/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 16384 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

For multiple GPUs, set `CUDA_VISIBLE_DEVICES` and `--tensor-parallel-size` accordingly. Keep tool calling enabled for the coordinator; see the [vLLM documentation](https://docs.vllm.ai/en/v0.10.2/features/tool_calling.html).

## Connect the coordinator

Keep the server running. In a second terminal, activate the simulation environment and check the connection:

```bash
conda activate orgpt
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
python -m virtual_suite.llm_client
```

The check verifies a model response and a tool call. The coordinating agent is available through `virtual_suite.autonomous_coordinator.autonomous_agent`.
