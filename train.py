import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import mlflow
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer
import time 
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

BASE_MODEL_ID = cfg["model"]["base_model_id"]
MODEL_PATH = cfg["model"]["model_path"]
DATA_PATH = cfg["data"]["data_path"]
OUTPUT_DIR = cfg["data"]["output_dir"]
LORA_R = cfg["lora"]["r"]
LORA_ALPHA = cfg["lora"]["alpha"]
LORA_DROPOUT = cfg["lora"]["dropout"]
EPOCHS = cfg["training"]["epochs"]
BATCH_SIZE = cfg["training"]["batch_size"]
GRAD_ACCUM = cfg["training"]["grad_accum"]
LEARNING_RATE = cfg["training"]["learning_rate"]
MAX_LENGTH = cfg["training"]["max_length"]
MLFLOW_URI = cfg["mlflow"]["tracking_uri"]
EXPERIMENT_NAME = cfg["mlflow"]["experiment_name"]


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, cache_dir=MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Loaded Tokenizer\n")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float32,
    trust_remote_code=True,
    device_map={"": device},
    cache_dir =MODEL_PATH,
)
print("Loaded model\n")

model.config.use_cache = False

dataset = load_from_disk(DATA_PATH)
train_ds = dataset["train"]
eval_ds = dataset["test"]
print(f"Train size: {len(train_ds)}")
print(f"Eval size:  {len(eval_ds)}")

def format_messages(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_ds = train_ds.map(format_messages, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(format_messages, remove_columns=eval_ds.column_names)

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

RUN_NAME = f"lora-sft-lr{LEARNING_RATE}-r{LORA_R}-{time.strftime('%Y%m%d-%H%M%S')}"

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params({
        "model_id": BASE_MODEL_ID,
        "device": device,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
    }) 
    num_update_steps = (len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    warmup_steps = max(1, int(0.05 * num_update_steps))

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        max_length=MAX_LENGTH,
        dataset_text_field="text",
        loss_type="chunked_nll",
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        report_to="mlflow",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting training ...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="model")
    print(f"Done. Model saved to {OUTPUT_DIR}")