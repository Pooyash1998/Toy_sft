import argparse
import glob
import os
import json
import time
import yaml
import mlflow
import lm_eval
from lm_eval.tasks import TaskManager
from utils import setup_env, get_device, EVAL_TASKS

setup_env()

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True, type=int)
parser.add_argument("--tag", required=True, choices=["baseline", "finetuned"], help="Which model to eval")
args = parser.parse_args()

config_path = glob.glob(f"configs/exp{args.exp}_*.yaml")[0]

with open(config_path) as f:
    cfg = yaml.safe_load(f)

MODEL_PATH  = cfg["model"]["base_model_id"] if args.tag == "baseline" else cfg["data"]["merged_dir"]
TASKS       = EVAL_TASKS
NUM_FEWSHOT = 0
LIMIT       = 200
device      = get_device()
dtype_str   = "bfloat16" if device == "cuda" else "float32"

print(f"\n[exp{args.exp}] Evaluating {args.tag}: {MODEL_PATH}")
print(f"  Device : {device}")
print(f"  Tasks  : {TASKS}")
print(f"  Limit  : {LIMIT} samples per task\n")

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

os.makedirs("eval_results", exist_ok=True)

task_manager = TaskManager(include_path="./custom_tasks") #Workaround for the datasets problem

for i, task in enumerate(TASKS, 1):
    print(f"\n[{i}/{len(TASKS)}] ► Starting task: {task} ...")

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={MODEL_PATH},dtype={dtype_str},device={device}",
        tasks=[task],
        num_fewshot=NUM_FEWSHOT,
        batch_size=1,
        limit=LIMIT,
        task_manager=task_manager,
        apply_chat_template=True,
        log_samples=True,
        confirm_run_unsafe_code=True,
    )

    results_path = f"eval_results/exp{args.exp}_{args.tag}_{task}.json"
    with open(results_path, "w") as f:
        json.dump(results["results"], f, indent=2)

    # qualitative dump: actual prompt sent + model's generated answer, for manual inspection
    samples_path = f"eval_results/exp{args.exp}_{args.tag}_{task}_samples.json"
    qualitative = []
    for ex in results.get("samples", {}).get(task, []):
        qualitative.append({
            "doc_id":     ex["doc_id"],
            "prompt":     ex["arguments"][0][0] if ex.get("arguments") else None,
            "target":     ex["target"],
            "prediction": ex["filtered_resps"][0] if ex.get("filtered_resps") else None,
            "metrics":    {m: ex[m] for m in ex.get("metrics", [])},
        })
    with open(samples_path, "w") as f:
        json.dump(qualitative, f, indent=2)

    run_name = f"lm-eval-exp{args.exp}-{args.tag}-{task}-{time.strftime('%Y%m%d-%H%M%S')}"

    print(f"[{i}/{len(TASKS)}] ✓ Done: {task} — logging to MLflow as '{run_name}'")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("exp",        args.exp)
        mlflow.log_param("tag",        args.tag)
        mlflow.log_param("model_path", MODEL_PATH)
        mlflow.log_param("task",       task)
        mlflow.log_artifact(results_path)
        mlflow.log_artifact(samples_path)

        for task_key, task_data in results["results"].items():
            for metric, value in task_data.items():
                if isinstance(value, (int, float)) and "_stderr" not in metric:
                    clean_metric = metric.replace(",", "_")
                    mlflow.log_metric(clean_metric, value)
                    print(f"  {clean_metric}: {value:.4f}")

print(f"\n[exp{args.exp}] All tasks complete.")
