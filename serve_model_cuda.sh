#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: bash serve_model_cuda.sh <merged_model_path> [port]"
  echo "Example: bash serve_model_cuda.sh ./merged_model/exp2 8080"
  exit 1
fi

MERGED_DIR=$1
PORT=${2:-8080}

vllm serve "$MERGED_DIR" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 4096