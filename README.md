## Setup

```bash
conda env create -f environment.yaml
conda activate sft
```

## Explaination 

1. **prepare_data.py** downloads and saves the UltraFeedback dataset locally
2. **train.py** fine-tunes the model with LoRA
3. **merge.py** merges the LoRA adapter into the base model
4. **eval.py** runs LM Evaluation Harness and logs results to MLflow
+ `experiment.py` to run everything end to end.

## Configs!

```bash
python experiment.py --config configs/exp2_lower_lr.yaml
```
This will:
- Evaluate the base model (baseline)
- Train with the config's hyperparameters
- Merge the adapter
- Evaluate the fine-tuned model
- Log a comparison (delta) to MLflow

## Configs i defined so far

in `configs/`:

| Config | LR | LoRA r | Epochs | Samples |
|--------|-----|--------|--------|---------|
| exp1_baseline | 2e-4 | 16 | 1 | 5000 |
| exp2_lower_lr | 5e-5 | 16 | 3 | 5000 |
| exp3_high_rank | 1e-4 | 32 | 3 | 5000 |

## Serving

On Mac (MLX):
```bash
bash serve_model_mlx.sh ./merged_model/exp2
```

On a CUDA cluster:
```bash
bash serve_model_cuda.sh ./merged_model/exp2
```

## Bugs 
- Training on MPS uses fp32 because fp16 is not supported and i got zero gradients
- On CUDA the scripts automatically switch to bf16