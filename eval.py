import argparse
import glob
import os
import json
import time
import yaml
import mlflow
import lm_eval
from utils import setup_env, get_device

setup_env()

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True, type=int)
parser.add_argument("--tag", required=True, choices=["baseline", "finetuned"], help="Which model to eval")
args = parser.parse_args()

config_path = glob.glob(f"configs/exp{args.exp}_*.yaml")[0]

with open(config_path) as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = cfg["model"]["base_model_id"] if args.tag == "baseline" else cfg["data"]["merged_dir"]
TASKS      = ["arc_easy", "mmlu", "winogrande"]
NUM_FEWSHOT = 0
LIMIT       = 2000 # number of samples lm_eval evaluates per task
device      = get_device()
dtype_str   = "bfloat16" if device == "cuda" else "float32"
RUN_NAME    = f"lm-eval-exp{args.exp}-{args.tag}-{time.strftime('%Y%m%d-%H%M%S')}"

print(f"[exp{args.exp}] Evaluating {args.tag}: {MODEL_PATH} on {device}")

results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={MODEL_PATH},dtype={dtype_str},device={device}",
    tasks=TASKS,
    num_fewshot=NUM_FEWSHOT,
    batch_size=1,
    limit=LIMIT,
)

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

os.makedirs("eval_results", exist_ok=True)
results_path = f"eval_results/exp{args.exp}_{args.tag}.json"
with open(results_path, "w") as f:
    json.dump(results["results"], f, indent=2)

print(f"\n=== Results (exp{args.exp} {args.tag}) ===")
with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("exp",        args.exp)
    mlflow.log_param("tag",        args.tag)
    mlflow.log_param("model_path", MODEL_PATH)
    mlflow.log_param("tasks",      TASKS)
    mlflow.log_artifact(results_path)

    for task, task_results in results["results"].items():
        print(f"\n{task}:")
        for metric, value in task_results.items():
            if isinstance(value, (int, float)) and "_stderr" not in metric:
                mlflow.log_metric(metric.replace(",", "_"), value)
                print(f"  {metric.replace(',', '_')}: {value:.4f}")

print(f"\nSaved to {results_path}")
