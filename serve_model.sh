#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
mlx_lm.server --model ./merged_model --port 8080
