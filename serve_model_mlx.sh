#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: bash serve_model_mlx.sh <merged_model_path> [port]"
  echo "Example: bash serve_model_mlx.sh ./merged_model/exp2 8080"
  exit 1
fi

export KMP_DUPLICATE_LIB_OK=TRUE
mlx_lm.server --model "$1" --port "${2:-8080}"