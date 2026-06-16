## Setup

**Mac**
```bash
conda create -n sft python=3.12 -y && conda activate sft
pip install torch torchvision torchaudio
pip install -r requirements-mac.txt
```

**CUDA cluster (tested on CUDA 12.0 / RTX 3090)**
```bash
conda create -n sft python=3.12 -y && conda activate sft
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## First-time data prep (run once)

```bash
python prepare_data.py        # downloads UltraFeedback (5000 train / 500 test)
python prepare_eval_data.py   # downloads MLQA + MathQA for eval tasks
```

## Running an experiment

```bash
python experiment.py --exp 1
```

This runs in one shot:
1. Eval base model (baseline)
2. LoRA fine-tune
3. Merge adapter
4. Eval merged model (finetuned)
5. Log delta to MLflow

Or individually:
```bash
python train.py  --config configs/exp1_baseline.yaml
python merge.py  --config configs/exp1_baseline.yaml
python eval.py   --exp 1 --tag baseline
python eval.py   --exp 1 --tag finetuned
```

## MLflow

MLflow logs to `mlruns.db` (SQLite file)

To view results locally:
```bash
bash serve_mlflow.sh    # opens http://localhost:5000
```

## Configs

| Config | LR | LoRA r | Epochs |
|--------|-----|--------|--------|
| exp1_baseline | 2e-4 | 16 | 1 |
| exp2_lower_lr | 5e-5 | 16 | 3 |
| exp3_high_rank | 1e-4 | 32 | 3 |

All use 5000 training samples, batch_size=2, grad_accum=4.

## Serving a merged model

On Mac (MLX):
```bash
bash serve_model_mlx.sh ./merged_model/exp1
```

On CUDA cluster:
```bash
bash serve_model_cuda.sh ./merged_model/exp1
```

## Notes
- MPS (Mac) uses fp32. fp16 causes NaN gradients
- CUDA automatically uses bf16
- eval tasks: mmlu, commonsense_qa, mlqa_en_en, mathqa