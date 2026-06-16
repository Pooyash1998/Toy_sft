import argparse
import json
import subprocess
import sys
import time
import yaml
import mlflow

import glob

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True, type=int)
args = parser.parse_args()

config_path = glob.glob(f"configs/exp{args.exp}_*.yaml")[0]
with open(config_path) as f:
    cfg = yaml.safe_load(f)

MERGED_DIR  = cfg["data"].get("merged_dir", "./merged_model")
EXP_NAME    = cfg.get("experiment", {}).get("name", "experiment")
RUN_NAME    = f"{EXP_NAME}-{time.strftime('%Y%m%d-%H%M%S')}"
BASE_MODEL  = cfg["model"]["base_model_id"]

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def load_eval_results(key):
    path = f"eval_results/{key}.json"
    with open(path) as f:
        return json.load(f)


with mlflow.start_run(run_name=RUN_NAME) as parent:
    mlflow.log_params({
        "config":        config_path,
        "base_model":    BASE_MODEL,
        "lora_r":        cfg["lora"]["r"],
        "lora_alpha":    cfg["lora"]["alpha"],
        "learning_rate": cfg["training"]["learning_rate"],
        "epochs":        cfg["training"]["epochs"],
        "num_samples":   5000,
    })

    exp_num = str(args.exp)

    # Step 1 — eval base model
    print("\n========== STEP 1: Baseline eval ==========")
    run([sys.executable, "eval.py", "--exp", exp_num, "--tag", "baseline"])
    base_results = load_eval_results(f"exp{exp_num}_baseline")

    # Step 2 — train
    print("\n========== STEP 2: Training ==========")
    run([sys.executable, "train.py", "--config", config_path])

    # Step 3 — merge
    print("\n========== STEP 3: Merging ==========")
    run([sys.executable, "merge.py", "--config", config_path])

    # Step 4 — eval merged model
    print("\n========== STEP 4: Finetuned eval ==========")
    run([sys.executable, "eval.py", "--exp", exp_num, "--tag", "finetuned"])
    ft_results = load_eval_results(f"exp{exp_num}_finetuned")

    # Step 5 — log delta metrics to parent run
    print("\n========== STEP 5: Comparison ==========")
    print(f"\n{'Task':<40} {'Baseline':>10} {'Finetuned':>10} {'Delta':>10}")
    print("-" * 72)
    for task in base_results:
        base_acc = base_results[task].get("acc,none")
        ft_acc   = ft_results.get(task, {}).get("acc,none")
        if base_acc is not None and ft_acc is not None:
            delta = ft_acc - base_acc
            mlflow.log_metric(f"baseline_{task}_acc",  base_acc)
            mlflow.log_metric(f"finetuned_{task}_acc", ft_acc)
            mlflow.log_metric(f"delta_{task}_acc",     delta)
            print(f"{task:<40} {base_acc:>10.4f} {ft_acc:>10.4f} {delta:>+10.4f}")

    print(f"\nExperiment '{RUN_NAME}' complete.")