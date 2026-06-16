import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import yaml
import mlflow
import lm_eval

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = "./merged_model"
TASKS = ["arc_easy", "mmlu", "gsm8k", "winogrande"]
NUM_FEWSHOT = 0
RUN_NAME = f"lm-eval-{'_'.join(TASKS)}-{time.strftime('%Y%m%d-%H%M%S')}"

print(f"Running lm-eval on {MODEL_PATH}, tasks: {TASKS}")

results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={MODEL_PATH},dtype=float32,device=mps",
    tasks=TASKS,
    num_fewshot=NUM_FEWSHOT,
    batch_size=1,
    limit=2000,
)

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("tasks", TASKS)
    mlflow.log_param("num_fewshot", NUM_FEWSHOT)
    mlflow.log_param("model", MODEL_PATH)

    for task, task_results in results["results"].items():
        for metric, value in task_results.items():
            if isinstance(value, (int, float)) and "_stderr" not in metric:
                mlflow.log_metric(f"{task}/{metric.replace(',', '_')}", value)

    results_path = "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    mlflow.log_artifact(results_path)

    print("\n=== Results ===")
    for task, task_results in results["results"].items():
        print(f"\n{task}:")
        for metric, value in task_results.items():
            if isinstance(value, (int, float)) and "_stderr" not in metric:
                print(f"  {metric.replace(',', '_')}: {value:.4f}")

print(f"\nResults logged to MLflow run '{RUN_NAME}' and saved to eval_results.json")
